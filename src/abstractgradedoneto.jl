"""
    AbstractGradedOneTo{S<:SectorRange} <: AbstractBlockedUnitRange{Int,Vector{Int}}

Supertype for graded axes — a blocked unit range carved into sectors, each with a data length
(multiplicity), plus a range-level `isdual` arrow. Concrete subtypes differ only in storage
and invariants:

  - [`GradedOneTo`](@ref) stores parallel `sectors`/`datalengths` vectors and may hold
    repeated or unsorted sectors (the intermediate state of a not-yet-merged fusion).
  - [`FusedGradedOneTo`](@ref) stores a sector-to-length `Dictionary` and is always fused and
    sorted (each sector once, in sorted order).

Subtypes must provide the primitive accessors `sectors`, `datalengths`, and `isdual`, plus
`dual` and `flip` (which return the same concrete type). Everything below is derived from
those.
"""
abstract type AbstractGradedOneTo{S <: SectorRange} <:
AbstractBlockedUnitRange{Int, Vector{Int}} end

# ========================  derived accessors  ========================

sectorlengths(g::AbstractGradedOneTo) = length.(sectors(g))
Base.first(::AbstractGradedOneTo) = 1
# An `AbstractGradedOneTo` is 1-based and acts as its own axis, like `Base.OneTo`. Without
# this, `axes` falls back to the `AbstractBlockedUnitRange` default, which returns a plain
# `BlockedOneTo` and drops the sectors.
Base.axes(g::AbstractGradedOneTo) = (g,)
BlockArrays.blocklasts(g::AbstractGradedOneTo) = cumsum(blocklengths(g))
BlockArrays.blocklength(g::AbstractGradedOneTo) = length(sectors(g))
BlockArrays.eachblockaxes1(g::AbstractGradedOneTo) = eachblockaxis(g)

# blocklengths: total length of each block (length(sector) * multiplicity).
function BlockArrays.blocklengths(g::AbstractGradedOneTo)
    return [length(s) * m for (s, m) in zip(sectors(g), datalengths(g))]
end

# sectortype, FusionStyle
sectortype(::Type{<:AbstractGradedOneTo{S}}) where {S} = S
TKS.FusionStyle(g::AbstractGradedOneTo) = TKS.FusionStyle(typeof(g))
TKS.FusionStyle(::Type{<:AbstractGradedOneTo{S}}) where {S} = TKS.FusionStyle(S)
dataaxistype(::Type{<:AbstractGradedOneTo}) = Base.OneTo{Int}

# ========================  BlockSparseArrays interface  ========================

function eachblockaxis(g::AbstractGradedOneTo)
    block_sectors = isdual(g) ? dual.(sectors(g)) : sectors(g)
    return [SectorOneTo(s, m) for (s, m) in zip(block_sectors, datalengths(g))]
end
eachdataaxis(g::AbstractGradedOneTo) = data.(eachblockaxis(g))
eachsectoraxis(g::AbstractGradedOneTo) = sector.(eachblockaxis(g))

# ========================  conj, flip_dual  ========================
# `dual` and `flip` are concrete-type-specific (they return the same concrete type); `conj`
# and `flip_dual` derive from them.
Base.conj(g::AbstractGradedOneTo) = dual(g)
flip_dual(g::AbstractGradedOneTo) = isdual(g) ? flip(g) : g

# Bounds checking (needed for AbstractArray scalar indexing).
Base.checkindex(::Type{Bool}, g::AbstractGradedOneTo, i::Int) = 1 <= i <= length(g)
