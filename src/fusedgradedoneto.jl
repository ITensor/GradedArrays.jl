using Dictionaries: Dictionaries, AbstractDictionary, Dictionary
using MappedArrays: mappedarray

# A graded axis whose sectors are fused and sorted: each sector appears once and the
# sectors are in sorted order. This is the canonical form of the coupled-sector axes of a
# `FusedGradedMatrix`, and it also matches the sorted-and-merged convention TensorKit
# uses for a `GradedSpace`.
#
# Stores the sectors and their data lengths (multiplicities) as sorted parallel vectors plus a
# single `isdual` flag. The sectors carry no arrow of their own, so the flag is the axis's entire
# duality.
#
# The sectors are held in their TensorKitSectors form, so the storage is exactly a
# `GradedSpace`'s and crossing into TensorKit hands over these vectors rather than rebuilding
# them, which means a space built from an axis aliases it. `I` is that stored type, a parameter
# only because a field type cannot be computed from `S`. Treat it as an implementation detail of
# the TensorKit conversion, liable to change.
struct FusedGradedOneTo{S <: Sector, I <: TKS.Sector} <: AbstractGradedOneTo{S}
    tensorkit_sectors::Vector{I}
    datalengths::Vector{Int}
    isdual::Bool
    function FusedGradedOneTo(
            tensorkit_sectors::Vector{I}, datalengths::Vector{Int}, isdual::Bool
        ) where {I <: TKS.Sector}
        length(tensorkit_sectors) == length(datalengths) ||
            throw(ArgumentError("sectors and datalengths must have the same length"))
        issortedunique(tensorkit_sectors) || throw(
            ArgumentError(
                "FusedGradedOneTo sectors must be sorted and unique: $(tensorkit_sectors)"
            )
        )
        return new{gradedarrays_sectortype(I), I}(tensorkit_sectors, datalengths, isdual)
    end
end

issortedunique(v) = all(((a, b),) -> isless(a, b), zip(v, Iterators.drop(v, 1)))

# The conversion happens once here rather than on every crossing into TensorKit.
function FusedGradedOneTo(
        sectors::AbstractVector{S}, datalengths::AbstractVector{<:Integer},
        isdual::Bool = false
    ) where {S <: Sector}
    return FusedGradedOneTo(
        tensorkit_sectortype(S)[TKS.Sector(s) for s in sectors],
        collect(Int, datalengths), isdual
    )
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

# `tensorkit_sectors` and `datalengths` hand back the stored vectors themselves, `sectors` maps
# the stored sectors lazily, and `sectordatalengths` is a zero-copy dictionary view keyed by them
# (lookups binary-search the sorted keys). Callers must not mutate the stored vectors: a
# `GradedSpace` built from this axis aliases them. The remaining range-interface methods are
# shared via `AbstractGradedOneTo`.
TensorAlgebra.isdual(g::FusedGradedOneTo) = g.isdual
tensorkit_sectors(g::FusedGradedOneTo) = g.tensorkit_sectors
# Wrapped in a closure because `mappedarray` reads a bare type argument as the element type of
# the result rather than as the function producing it, and `Sector` is abstract.
sectors(g::FusedGradedOneTo) = mappedarray(c -> Sector(c), tensorkit_sectors(g))
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
# matched, which the test after the loop confirms.
function setsectors(g::FusedGradedOneTo{S}, ss::Vector{S}) where {S}
    return setsectors(g, tensorkit_sectortype(S)[TKS.Sector(s) for s in ss])
end

# `cs` is stored as is, so axes set from the same vector share it.
function setsectors(g::FusedGradedOneTo{S, I}, cs::Vector{I}) where {S, I}
    gl, gd = tensorkit_sectors(g), datalengths(g)
    (cs === gl || cs == gl) && return g
    lens = Vector{Int}(undef, length(cs))
    i = 1
    for k in eachindex(cs)
        if i <= length(gl) && isequal(gl[i], cs[k])
            lens[k] = gd[i]
            i += 1
        else
            lens[k] = 0
        end
    end
    i > length(gl) || throw(
        ArgumentError("sectors $(cs) do not cover the axis support $(gl)")
    )
    return FusedGradedOneTo(cs, lens, isdual(g))
end

# ========================  dual, flip  ========================

# `dual` flips the arrow only; the stored sectors and their order are unchanged, so the
# fused+sorted invariant is preserved.
function TensorAlgebra.dual(g::FusedGradedOneTo)
    return FusedGradedOneTo(tensorkit_sectors(g), datalengths(g), !isdual(g))
end

# `flip` conjugates the sectors and flips the arrow (matching `GradedOneTo`), leaving the block
# sectors unchanged. Conjugation generally reorders the sectors, so re-sort to restore the
# canonical fused form.
function flip(g::FusedGradedOneTo)
    flipped = map(TKS.dual, tensorkit_sectors(g))
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

# Construct a non-dual `FusedGradedOneTo` from `sector => multiplicity` pairs, keyed by anything
# `Sector` accepts. The sectors must already be in canonical fused form (each once, in sorted
# order); non-canonical input is rejected by the constructor. Wrap the result in `dual` for a
# dual axis.
function fusedgradedrange(xs::AbstractVector{<:Pair})
    return FusedGradedOneTo(map(p -> Sector(first(p)), xs), Int[last(p) for p in xs], false)
end

# ========================  conversions between graded-axis types  ========================

FusedGradedOneTo(g::FusedGradedOneTo) = g

# Fuse any graded axis into canonical form. This is value-preserving: the constructor rejects
# unsorted input rather than silently re-sorting. (`GradedOneTo` has a cached fast path in
# `gradedoneto.jl`.)
function FusedGradedOneTo(g::AbstractGradedOneTo)
    return FusedGradedOneTo(sectors(g), datalengths(g), isdual(g))
end

# `convert` is a thin delegator to the constructor (the worker); `convert(::Type{T}, ::T)` from Base
# gives the no-op on an already-fused axis.
Base.convert(::Type{FusedGradedOneTo}, g::AbstractGradedOneTo) = FusedGradedOneTo(g)

# ========================  fusesectors  ========================

# An already-fused axis is its own fused form.
fusesectors(g::FusedGradedOneTo) = g

# The repairing counterpart to the constructor, which rejects unsorted or repeated sectors rather
# than fixing them: sort the sectors, sum the data lengths of the repeats, and build the axis from
# the result. The sectors carry no arrow, so the merge is exact and the axis-level arrow plays no
# part beyond being carried through. Called by the generic `fusesectors` and by the `GradedOneTo`
# constructor, which fuses eagerly and so has no axis to hand `fusesectors` yet.
function sortmergesectors(
        sectors::AbstractVector{S}, datalengths::AbstractVector{Int}, isdual::Bool
    ) where {S <: Sector}
    merged_sectors = S[]
    merged_datalengths = Int[]
    for p in sortperm(sectors)
        if !isempty(merged_sectors) && isequal(last(merged_sectors), sectors[p])
            merged_datalengths[end] += datalengths[p]
        else
            push!(merged_sectors, sectors[p])
            push!(merged_datalengths, datalengths[p])
        end
    end
    return FusedGradedOneTo(merged_sectors, merged_datalengths, isdual)
end
