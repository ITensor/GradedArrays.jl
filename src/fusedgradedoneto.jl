using Dictionaries: Dictionaries, AbstractDictionary, Dictionary

"""
    FusedGradedOneTo{S<:Sector}

A graded axis whose sectors are fused and sorted: each sector appears once and the
sectors are in sorted order. This is the canonical form of the coupled-sector axes of a
[`FusedGradedMatrix`](@ref), and it also matches the sorted-and-merged convention TensorKit
uses for a `GradedSpace`.

Stores the sectors and their data lengths (multiplicities) as sorted parallel vectors — the
same layout as a TensorKit `GradedSpace` — plus a single `isdual` flag. The sectors carry no
arrow of their own, so the flag is the axis's entire duality.
"""
struct FusedGradedOneTo{S <: Sector} <: AbstractGradedOneTo{S}
    sectors::Vector{S}
    datalengths::Vector{Int}
    isdual::Bool
    function FusedGradedOneTo(
            sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
        ) where {S <: Sector}
        length(sectors) == length(datalengths) ||
            throw(ArgumentError("sectors and datalengths must have the same length"))
        issortedunique(sectors) || throw(
            ArgumentError(
                "FusedGradedOneTo sectors must be sorted and unique: $(sectors)"
            )
        )
        return new{S}(sectors, datalengths, isdual)
    end
end

issortedunique(v) = all(((a, b),) -> isless(a, b), zip(v, Iterators.drop(v, 1)))

# Arrow defaults to non-dual.
function FusedGradedOneTo(sectors::Vector{<:Sector}, datalengths::Vector{Int})
    return FusedGradedOneTo(sectors, datalengths, false)
end

# Bare TensorKitSectors labels, as they come back from a TensorKit space.
function FusedGradedOneTo(
        labels::Vector{<:TKS.Sector}, datalengths::Vector{Int}, isdual::Bool = false
    )
    return FusedGradedOneTo(map(Sector, labels), datalengths, isdual)
end

# Dictionary convenience (e.g. a `map` over `sectordata`); the keys must already be in
# canonical fused form.
function FusedGradedOneTo(
        sector_datalengths::AbstractDictionary{S, Int}, isdual::Bool = false
    ) where {S <: Sector}
    return FusedGradedOneTo(
        collect(keys(sector_datalengths)), collect(sector_datalengths), isdual
    )
end

# ========================  primitive accessors  ========================

# `sectors` and `datalengths` hand back the stored vectors themselves, and `sectordatalengths`
# is a zero-copy dictionary view keyed by them (lookups binary-search the sorted keys).
# Callers must not mutate them. The remaining range-interface methods are shared via
# `AbstractGradedOneTo`.
TensorAlgebra.isdual(g::FusedGradedOneTo) = g.isdual
sectors(g::FusedGradedOneTo) = g.sectors
datalengths(g::FusedGradedOneTo) = g.datalengths
sectordatalengths(g::FusedGradedOneTo) = SortedArrayDictionary(sectors(g), datalengths(g))

# Lenient per-sector length: the `get`-prefixed name marks the fallback to length 0 for a sector
# the axis does not carry.
getsectordatalengths(g::FusedGradedOneTo, c) = get(sectordatalengths(g), c, 0)

# Position of sector `c` among the sorted sectors (its block index in the axis); the axis
# stores each sector once, so the position is unique. Throws for an absent sector.
function sectorindex(g::FusedGradedOneTo, c)
    (hassector, t) = gettoken(sectordatalengths(g), c)
    hassector || throw(ArgumentError("sector $c is not in the axis"))
    return t
end

# ========================  setsectors  ========================

# Sectors of `ss` that `g` already has keep their lengths, the ones it lacks get length zero.
# The walk that fills `lens` doubles as the coverage check: every stored sector has to be
# matched, which the test after the loop confirms. `ss` is stored as is, so axes set from the
# same vector share it.
function setsectors(g::FusedGradedOneTo{S}, ss::Vector{S}) where {S}
    gl, gd = sectors(g), datalengths(g)
    (ss === gl || ss == gl) && return g
    lens = Vector{Int}(undef, length(ss))
    i = 1
    for k in eachindex(ss)
        if i <= length(gl) && isequal(gl[i], ss[k])
            lens[k] = gd[i]
            i += 1
        else
            lens[k] = 0
        end
    end
    i > length(gl) || throw(
        ArgumentError("sectors $(ss) do not cover the axis support $(gl)")
    )
    return FusedGradedOneTo(ss, lens, isdual(g))
end

# ========================  dual, flip  ========================

# `dual` flips the arrow only; the stored sectors and their order are unchanged, so the
# fused+sorted invariant is preserved.
function TensorAlgebra.dual(g::FusedGradedOneTo)
    return FusedGradedOneTo(sectors(g), datalengths(g), !isdual(g))
end

# `flip` conjugates the sectors and flips the arrow (matching `GradedOneTo`), leaving the block
# sectors unchanged. Conjugation generally reorders the sectors, so re-sort to restore the
# canonical fused form.
function flip(g::FusedGradedOneTo)
    flipped = map(charge_conjugate, sectors(g))
    perm = sortperm(flipped)
    return FusedGradedOneTo(flipped[perm], datalengths(g)[perm], !isdual(g))
end

# ========================  show  ========================

# Factor the `dual` to the outside — `dual(fusedgradedrange([...]))` — so the printed form is
# compact and round-trips through the constructor.
function Base.show(io::IO, g::FusedGradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "fusedgradedrange([")
    join(io, (s => m for (s, m) in zip(sectors(g), datalengths(g))), ", ")
    print(io, "])")
    isdual(g) && print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", g::FusedGradedOneTo)
    summary(io, g)
    isempty(g) && return nothing
    print(io, ":\n  sectors: ")
    isdual(g) && print(io, "dual.(")
    print(io, "[")
    join(io, sectors(g), ", ")
    print(io, "]")
    isdual(g) && print(io, ")")
    println(io)
    Base.print_array(io, g)
    return nothing
end

# ========================  fusedgradedrange constructors  ========================

"""
    fusedgradedrange(xs::AbstractVector{<:Pair{<:Sector, <:Integer}})

Construct a non-dual [`FusedGradedOneTo`](@ref) from `sector => multiplicity` pairs. The sectors
must already be in canonical fused form (each once, in sorted order); non-canonical input is
rejected by the constructor. Wrap the result in `dual` for a dual axis.
"""
function fusedgradedrange(xs::AbstractVector{<:Pair{S, <:Integer}}) where {S <: Sector}
    return FusedGradedOneTo(S[first(p) for p in xs], Int[last(p) for p in xs], false)
end

# Generic fallback mirroring `gradedrange`: converts keys through `Sector`, which accepts
# NamedTuple keys (for sector products) and rejects an arrow-carrying key with a message
# pointing at the axis.
function fusedgradedrange(xs::AbstractVector{<:Pair})
    isempty(xs) && throw(
        ArgumentError("Cannot create FusedGradedOneTo from empty vector without type info")
    )
    return fusedgradedrange([Sector(first(p)) => last(p) for p in xs])
end

# ========================  conversions between graded-axis types  ========================

FusedGradedOneTo(g::FusedGradedOneTo) = g

# Fuse any graded axis into canonical form. This is value-preserving: the constructor rejects
# unsorted input rather than silently re-sorting. (`GradedOneTo` has a cached fast path in
# `gradedoneto.jl`.)
function FusedGradedOneTo(g::AbstractGradedOneTo)
    return FusedGradedOneTo(collect(sectors(g)), datalengths(g), isdual(g))
end

# `convert` is a thin delegator to the constructor (the worker); `convert(::Type{T}, ::T)` from Base
# gives the no-op on an already-fused axis.
Base.convert(::Type{FusedGradedOneTo}, g::AbstractGradedOneTo) = FusedGradedOneTo(g)

# ========================  mergesectors  ========================

# Merge repeated sectors (summing their data lengths) and sort. The sectors carry no arrow, so
# merging them is exact and the axis-level arrow plays no part. Returns the sorted sectors, each
# appearing once, and the summed data lengths.
function mergesectors(
        sectors::AbstractVector{S}, datalengths::AbstractVector{Int}
    ) where {S <: Sector}
    perm = sortperm(sectors)
    merged_sectors = Vector{S}(undef, 0)
    merged_datalengths = Vector{Int}(undef, 0)
    for p in perm
        s = sectors[p]
        if !isempty(merged_sectors) && isequal(last(merged_sectors), s)
            merged_datalengths[end] += datalengths[p]
        else
            push!(merged_sectors, s)
            push!(merged_datalengths, datalengths[p])
        end
    end
    return (merged_sectors, merged_datalengths)
end

# ========================  fusesectors  ========================

# An already-fused axis is its own fused form.
fusesectors(g::FusedGradedOneTo) = g
