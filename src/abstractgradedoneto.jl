# A `GradedOneTo` may hold repeated or unsorted sectors, while a `FusedGradedOneTo` holds each
# sector once, in sorted order. Subtypes provide `sectors`, `datalengths`, `isdual`, `dual` and
# `flip`.
"""
    AbstractGradedOneTo{S <: Sector}

Supertype of graded axes, such as [`GradedOneTo`](@ref): a unit range split into blocks, each
labelled by a sector of type `S` with a multiplicity, plus an arrow for the whole range that
`isdual` reports. Use [`sectors`](@ref) and [`datalengths`](@ref) to read the blocks.
"""
abstract type AbstractGradedOneTo{S <: Sector} <: AbstractUnitRange{Int} end

"""
    sectors(g)

The sector of each block of a graded axis, in block order.
"""
function sectors end

"""
    datalengths(g)

The multiplicity of each block of a graded axis, in block order. A block's length is its
multiplicity times the dimension of its sector.

# Examples

```jldoctest
julia> using GradedArrays: SU2, datalengths, gradedrange

julia> g = gradedrange([SU2(0) => 2, SU2(1 // 2) => 3]);

julia> datalengths(g)
2-element Vector{Int64}:
 2
 3

julia> length(g)
8
```
"""
function datalengths end

# ========================  derived accessors  ========================

sectorlengths(g::AbstractGradedOneTo) = length.(sectors(g))
Base.first(::AbstractGradedOneTo) = 1
Base.length(g::AbstractGradedOneTo) = sum(blocklengths(g))
Base.last(g::AbstractGradedOneTo) = length(g)
# Like `Base.OneTo`, a graded axis is its own axis. The generic `axes` would drop the sectors.
Base.axes(g::AbstractGradedOneTo) = (g,)
BlockArrays.blocklasts(g::AbstractGradedOneTo) = cumsum(blocklengths(g))
BlockArrays.blocklength(g::AbstractGradedOneTo) = length(sectors(g))
BlockArrays.blockaxes(g::AbstractGradedOneTo) = (Block.(Base.OneTo(blocklength(g))),)
BlockArrays.eachblockaxes1(g::AbstractGradedOneTo) = eachblockaxis(g)
function BlockArrays.findblock(g::AbstractGradedOneTo, i::Integer)
    @boundscheck i in g || throw(BoundsError(g, i))
    return Block(searchsortedfirst(blocklasts(g), i))
end

# Lazy, since `length` sums it and `length` is hit by every hash and comparison of an index.
function BlockArrays.blocklengths(g::AbstractGradedOneTo)
    return mappedarray((s, m) -> length(s) * m, sectors(g), datalengths(g))
end

# sectortype, FusionStyle
sectortype(::Type{<:AbstractGradedOneTo{S}}) where {S} = S
TKS.FusionStyle(g::AbstractGradedOneTo) = TKS.FusionStyle(typeof(g))
TKS.FusionStyle(::Type{<:AbstractGradedOneTo{S}}) where {S} = TKS.FusionStyle(S)
dataaxistype(::Type{<:AbstractGradedOneTo}) = Base.OneTo{Int}

# ========================  BlockSparseArrays interface  ========================

function eachblockaxis(g::AbstractGradedOneTo)
    # The stored sectors carry no arrow, so each block axis takes the range's own.
    return [SectorOneTo(s, m, isdual(g)) for (s, m) in zip(sectors(g), datalengths(g))]
end
eachdataaxis(g::AbstractGradedOneTo) = data.(eachblockaxis(g))
eachstructureaxis(g::AbstractGradedOneTo) = structure.(eachblockaxis(g))

# ========================  conj, flip_dual  ========================
Base.conj(g::AbstractGradedOneTo) = dual(g)
flip_dual(g::AbstractGradedOneTo) = isdual(g) ? flip(g) : g

Base.checkindex(::Type{Bool}, g::AbstractGradedOneTo, i::Int) = 1 <= i <= length(g)

# ========================  equality, hashing  ========================
# Compared by content, so equal axes compare equal across the concrete subtypes.
function Base.isequal(a::AbstractGradedOneTo, b::AbstractGradedOneTo)
    return isequal(sectors(a), sectors(b)) &&
        isequal(datalengths(a), datalengths(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::AbstractGradedOneTo, b::AbstractGradedOneTo) = isequal(a, b)
function Base.hash(g::AbstractGradedOneTo, h::UInt)
    return hash(
        :AbstractGradedOneTo,
        hash(sectors(g), hash(datalengths(g), hash(isdual(g), h)))
    )
end

# ========================  show  ========================
# `showarg` rather than `summary`, so Base still prints the element count and the indices.
function Base.showarg(io::IO, g::AbstractGradedOneTo, toplevel::Bool)
    toplevel || print(io, "::")
    print(io, summary_typename(io, typeof(g)))
    return nothing
end
