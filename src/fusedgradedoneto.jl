using Dictionaries: Dictionaries, AbstractDictionary, Dictionary
using MappedArrays: ReadonlyMappedArray, mappedarray

"""
    FusedGradedOneTo{S<:SectorRange}

A graded axis whose sectors are fused and sorted: each sector appears once and the
sectors are in sorted order. This is the canonical form of the coupled-sector axes of a
[`FusedGradedMatrix`](@ref), and it also matches the sorted-and-merged convention TensorKit
uses for a `GradedSpace`.

Stores the bare (non-dual) sector labels and their data lengths (multiplicities) as sorted
parallel vectors — the same layout as a TensorKit `GradedSpace`, so the space conversion is
a transcription of the stored vectors — plus a single `isdual` flag.
"""
struct FusedGradedOneTo{S <: SectorRange} <: AbstractGradedOneTo{S}
    # The label vector is `Vector{labeltype(S)}`; a field type cannot be computed from a
    # type parameter, so the field is loosely typed and `sectorlabels` recovers the concrete
    # type via a typeassert.
    labels::Vector
    datalengths::Vector{Int}
    isdual::Bool
    function FusedGradedOneTo(
            labels::Vector{I}, datalengths::Vector{Int}, isdual::Bool
        ) where {I <: TKS.Sector}
        length(labels) == length(datalengths) ||
            throw(ArgumentError("sectors and datalengths must have the same length"))
        issortedunique(labels) || throw(
            ArgumentError(
                "FusedGradedOneTo sectors must be sorted and unique: $(labels)"
            )
        )
        return new{SectorRange{I}}(labels, datalengths, isdual)
    end
end

issortedunique(v) = all(i -> isless(v[i - 1], v[i]), 2:length(v))

# Arrow defaults to non-dual.
function FusedGradedOneTo(labels::Vector{<:TKS.Sector}, datalengths::Vector{Int})
    return FusedGradedOneTo(labels, datalengths, false)
end

# `SectorRange` convenience: strip the sectors to their bare labels after checking they
# carry no arrow of their own (the arrow is axis-level, passed via `isdual`).
function FusedGradedOneTo(
        sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
    ) where {S <: SectorRange}
    all(s -> !TensorAlgebra.isdual(s), sectors) || throw(
        ArgumentError(
            "FusedGradedOneTo stores non-dual sectors; pass the arrow via `isdual`"
        )
    )
    labels = labeltype(S)[label(s) for s in sectors]
    return FusedGradedOneTo(labels, datalengths, isdual)
end
# Arrow defaults to non-dual.
function FusedGradedOneTo(
        sectors::Vector{S},
        datalengths::Vector{Int}
    ) where {S <: SectorRange}
    return FusedGradedOneTo(sectors, datalengths, false)
end

# Dictionary convenience (e.g. a `map` over `sectordata`); the keys must already be in
# canonical fused form.
function FusedGradedOneTo(
        sector_datalengths::AbstractDictionary{S, Int}, isdual::Bool
    ) where {S <: SectorRange}
    return FusedGradedOneTo(
        collect(keys(sector_datalengths)), collect(sector_datalengths), isdual
    )
end
# Arrow defaults to non-dual.
function FusedGradedOneTo(
        sector_datalengths::AbstractDictionary{S, Int}
    ) where {S <: SectorRange}
    return FusedGradedOneTo(sector_datalengths, false)
end

# ========================  zero-copy views over the stored vectors  ========================
# The lazy read-only views over the stored sorted label vector: the positional `sectors(g)` vector
# (a `mappedarray` whose entries materialize as `SectorRange`s on access, with the bare labels
# reachable through `parent`) and, keyed by it, the `SortedArrayDictionary` returned by
# `sectordatalengths` (sector → data length; lookups binary-search the sorted keys). Callers must
# not mutate the vectors they wrap.

# The concrete type of the lazy sector view over a bare-label vector, for typeasserts (the type
# involves `labeltype(S)`, so a struct field cannot spell it).
function sectorstype(::Type{SectorRange{I}}) where {I}
    return ReadonlyMappedArray{SectorRange{I}, 1, Vector{I}, Type{SectorRange{I}}}
end

# Strip a vector of (non-dual) sectors to the lazy bare-label view form, for the generic
# `SortedArrayDictionary` canonicalization.
function Base.convert(
        ::Type{ReadonlyMappedArray{SectorRange{I}, 1, Vector{I}, Type{SectorRange{I}}}},
        v::AbstractVector{SectorRange{I}}
    ) where {I}
    all(s -> !TensorAlgebra.isdual(s), v) ||
        throw(ArgumentError("sector keys must be non-dual"))
    return mappedarray(SectorRange{I}, I[label(s) for s in v])
end

# ========================  primitive accessors  ========================

# `sectors`/`datalengths`/`sectordatalengths` are zero-copy views over the stored parallel
# vectors (see above); `sectorlabels` is the internal bare-label primitive. The remaining
# range-interface methods are shared via `AbstractGradedOneTo`.
TensorAlgebra.isdual(g::FusedGradedOneTo) = g.isdual
sectorlabels(g::FusedGradedOneTo{SectorRange{I}}) where {I} = g.labels::Vector{I}
function sectors(g::FusedGradedOneTo{SectorRange{I}}) where {I}
    return mappedarray(SectorRange{I}, sectorlabels(g))
end
datalengths(g::FusedGradedOneTo) = g.datalengths
sectordatalengths(g::FusedGradedOneTo) = SortedArrayDictionary(sectors(g), datalengths(g))

# Per-sector length accessors following the strict/lenient convention: the bare 2-arg form is
# strict (throws on an absent sector), the `get`-prefixed form falls back to length 0.
sectordatalengths(g::FusedGradedOneTo, c) = sectordatalengths(g)[c]
getsectordatalengths(g::FusedGradedOneTo, c) = get(sectordatalengths(g), c, 0)

# Position of sector `c` among the sorted sectors (its block index in the axis); the axis
# stores each sector once, so the position is unique. Throws for an absent sector.
function findsectorindex(g::FusedGradedOneTo, c)
    (hassector, t) = gettoken(sectordatalengths(g), c)
    hassector || throw(ArgumentError("sector $c is not in the axis"))
    return t
end

# ========================  setsectors  ========================

# The bare label vector behind a sector vector: zero-copy for the lazy sector view over a label
# vector (the `sectors` form), one strip per call otherwise.
function sectorlabelvector(
        cs::ReadonlyMappedArray{SectorRange{I}, 1, <:Vector, Type{SectorRange{I}}}
    ) where {I}
    return parent(cs)
end
function sectorlabelvector(cs::AbstractVector{S}) where {S <: SectorRange}
    all(s -> !TensorAlgebra.isdual(s), cs) ||
        throw(ArgumentError("sectors must be non-dual"))
    return labeltype(S)[label(s) for s in cs]
end

# The support-set axis: sector support exactly `ls`, keeping `g`'s per-sector lengths (length
# zero for the added sectors) and arrow. `ls` must be sorted, unique, and cover `g`'s support
# (`ArgumentError` otherwise); the allocation-free walk that gathers the lengths doubles as the
# covering check, and the constructor rejects an unsorted or non-unique `ls`. The result stores
# `ls` itself, so every axis set from one vector shares it. An `ls` equal to the stored support
# returns `g` itself, so callers can detect the identity by `===` and skip rebuilding anything
# derived from the axis.
function setsectors(g::FusedGradedOneTo{SectorRange{I}}, ls::Vector{I}) where {I}
    gl, gd = sectorlabels(g), datalengths(g)
    (ls === gl || ls == gl) && return g
    lens = Vector{Int}(undef, length(ls))
    i = 1
    for k in eachindex(ls)
        if i <= length(gl) && isequal(gl[i], ls[k])
            lens[k] = gd[i]
            i += 1
        else
            lens[k] = 0
        end
    end
    i > length(gl) || throw(
        ArgumentError("sectors $(ls) do not cover the axis support $(gl)")
    )
    return FusedGradedOneTo(ls, lens, isdual(g))
end

function setsectors(g::FusedGradedOneTo{S}, cs::AbstractVector{S}) where {S <: SectorRange}
    return setsectors(g, sectorlabelvector(cs))
end

# ========================  dual, flip  ========================

# `dual` flips the arrow only; the stored (non-dual) labels and their order are unchanged,
# so the fused+sorted invariant is preserved.
function TensorAlgebra.dual(g::FusedGradedOneTo)
    return FusedGradedOneTo(sectorlabels(g), datalengths(g), !isdual(g))
end

# `flip` conjugates the sector labels and flips the arrow (matching `GradedOneTo`), leaving
# the block sectors unchanged. Dualizing the labels generally reorders them, so re-sort to
# restore the canonical fused form.
function flip(g::FusedGradedOneTo)
    flipped = map(dual, sectorlabels(g))
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
    fusedgradedrange(xs::AbstractVector{<:Pair{<:SectorRange, <:Integer}})

Construct a non-dual [`FusedGradedOneTo`](@ref) from `sector => multiplicity` pairs. The sectors
must be non-dual and already in canonical fused form (each once, in sorted order); non-canonical
or dual input is rejected by the constructor. Wrap the result in `dual` for a dual axis.
"""
function fusedgradedrange(xs::AbstractVector{<:Pair{S, <:Integer}}) where {S <: SectorRange}
    return FusedGradedOneTo(S[first(p) for p in xs], Int[last(p) for p in xs], false)
end

# ========================  conversions between graded-axis types  ========================

FusedGradedOneTo(g::FusedGradedOneTo) = g

# Fuse any graded axis into canonical form. This is value-preserving: the constructor rejects
# unsorted/dual input rather than silently re-sorting. (`GradedOneTo` has a cached fast path
# in `gradedoneto.jl`.)
function FusedGradedOneTo(g::AbstractGradedOneTo)
    return FusedGradedOneTo(sectors(g), datalengths(g), isdual(g))
end

# `convert` is a thin delegator to the constructor (the worker); `convert(::Type{T}, ::T)` from Base
# gives the no-op on an already-fused axis.
Base.convert(::Type{FusedGradedOneTo}, g::AbstractGradedOneTo) = FusedGradedOneTo(g)

# ========================  sectormergesort  ========================

# Merge repeated sectors (summing their data lengths) and sort into canonical fused form,
# building the sorted label/length vectors directly. The stored sectors are non-dual and the
# arrow is axis-level, so merging their labels is exact. The vector-level worker lets
# `GradedOneTo` fuse at construction time, before the axis object exists.
function sectormergesort(
        sectors::AbstractVector{S}, datalengths::AbstractVector{Int}, isdual::Bool
    ) where {S <: SectorRange}
    perm = sortperm(sectors)
    labels = Vector{labeltype(S)}(undef, 0)
    merged_datalengths = Vector{Int}(undef, 0)
    for p in perm
        l = label(sectors[p])
        if !isempty(labels) && isequal(last(labels), l)
            merged_datalengths[end] += datalengths[p]
        else
            push!(labels, l)
            push!(merged_datalengths, datalengths[p])
        end
    end
    return FusedGradedOneTo(labels, merged_datalengths, isdual)
end

# An already-fused axis is its own fused form.
sectormergesort(g::FusedGradedOneTo) = g
