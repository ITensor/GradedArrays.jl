"""
    FusedGradedOneTo{S<:SectorRange}

A graded axis whose sectors are fused and sorted: each sector appears once and the
sectors are in sorted order. This is the canonical form of the coupled-sector axes of a
[`FusedGradedMatrix`](@ref), and it also matches the sorted-and-merged convention TensorKit
uses for a `GradedSpace`.

Stores a `Dictionary` mapping each (non-dual) `SectorRange` to its data length
(multiplicity), plus a single `isdual` flag.
"""
struct FusedGradedOneTo{S <: SectorRange} <: AbstractGradedOneTo{S}
    sector_datalengths::Dictionary{S, Int}
    isdual::Bool
    function FusedGradedOneTo(
            sector_datalengths::Dictionary{S, Int}, isdual::Bool
        ) where {S <: SectorRange}
        all(s -> !TensorAlgebra.isdual(s), keys(sector_datalengths)) || throw(
            ArgumentError(
                "FusedGradedOneTo stores non-dual sectors; pass the arrow via `isdual`"
            )
        )
        issorted(keys(sector_datalengths)) || throw(
            ArgumentError(
                "FusedGradedOneTo sectors must be sorted: $(keys(sector_datalengths))"
            )
        )
        return new{S}(sector_datalengths, isdual)
    end
end

# Arrow defaults to non-dual (sectors assumed non-dual, checked by the inner constructor).
function FusedGradedOneTo(sector_datalengths::Dictionary{S, Int}) where {S <: SectorRange}
    return FusedGradedOneTo(sector_datalengths, false)
end

# Vector convenience; the `Dictionary` requires unique (merged) sectors, and the inner
# constructor checks sorted and non-dual.
function FusedGradedOneTo(
        sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
    ) where {S <: SectorRange}
    length(sectors) == length(datalengths) ||
        throw(ArgumentError("sectors and datalengths must have the same length"))
    return FusedGradedOneTo(Dictionary{S, Int}(sectors, datalengths), isdual)
end
# Arrow defaults to non-dual (sectors assumed non-dual, checked by the inner constructor).
function FusedGradedOneTo(
        sectors::Vector{S},
        datalengths::Vector{Int}
    ) where {S <: SectorRange}
    return FusedGradedOneTo(sectors, datalengths, false)
end

# Primitive accessors. `sectors`/`datalengths` return vectors (in fused/sorted order) to
# match the `GradedOneTo` interface; `sectordatalengths` exposes the underlying sector-to-length
# `Dictionary` for keyed per-sector lookups. The remaining range-interface methods are shared via
# `AbstractGradedOneTo`.
TensorAlgebra.isdual(g::FusedGradedOneTo) = g.isdual
sectors(g::FusedGradedOneTo) = collect(keys(g.sector_datalengths))
datalengths(g::FusedGradedOneTo) = collect(values(g.sector_datalengths))
sectordatalengths(g::FusedGradedOneTo) = g.sector_datalengths

# Per-sector length/axis accessors following the strict/lenient convention: the bare 2-arg form is
# strict (throws on an absent sector), the `get`-prefixed form falls back to length 0. Only the
# lenient axis accessor exists, since a data axis is needed only to size an absent (zero) block.
sectordatalengths(g::FusedGradedOneTo, c) = sectordatalengths(g)[c]
getsectordatalengths(g::FusedGradedOneTo, c) = get(sectordatalengths(g), c, 0)
getsectordataaxis(g::FusedGradedOneTo, c) = Base.OneTo(getsectordatalengths(g, c))

# ========================  dual, flip  ========================

# `dual` flips the arrow only; the stored (non-dual) sectors and their order are unchanged,
# so the fused+sorted invariant is preserved.
TensorAlgebra.dual(g::FusedGradedOneTo) = FusedGradedOneTo(g.sector_datalengths, !isdual(g))

# `flip` conjugates the sector labels and flips the arrow (matching `GradedOneTo`), leaving
# the block sectors unchanged. Dualizing the labels generally reorders them, so re-sort to
# restore the canonical fused form.
function flip(g::FusedGradedOneTo)
    new_nondual = [SectorRange(dual(label(s))) for s in sectors(g)]
    perm = sortperm(new_nondual)
    return FusedGradedOneTo(new_nondual[perm], datalengths(g)[perm], !isdual(g))
end

# ========================  show  ========================

# Factor the `dual` to the outside — `dual(fusedgradedrange([...]))` — so the printed form is
# compact and round-trips through the constructor.
function Base.show(io::IO, g::FusedGradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "fusedgradedrange([")
    join(io, (s => m for (s, m) in pairs(g.sector_datalengths)), ", ")
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
    join(io, keys(g.sector_datalengths), ", ")
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
# unsorted/dual input rather than silently re-sorting.
function FusedGradedOneTo(g::AbstractGradedOneTo)
    return FusedGradedOneTo(sectors(g), datalengths(g), isdual(g))
end

# `convert` is a thin delegator to the constructor (the worker); `convert(::Type{T}, ::T)` from Base
# gives the no-op on an already-fused axis.
Base.convert(::Type{FusedGradedOneTo}, g::AbstractGradedOneTo) = FusedGradedOneTo(g)

GradedOneTo(g::GradedOneTo) = g
GradedOneTo(g::AbstractGradedOneTo) = GradedOneTo(sectors(g), datalengths(g), isdual(g))
Base.convert(::Type{GradedOneTo}, g::AbstractGradedOneTo) = GradedOneTo(g)
