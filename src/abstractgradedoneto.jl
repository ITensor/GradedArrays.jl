# Supertype for graded axes — a unit range carved into sectors (its blocks), each with a data length
# (multiplicity), plus a range-level `isdual` arrow. Concrete subtypes differ only in storage
# and invariants:
#
#   - `GradedOneTo` stores parallel `sectors`/`datalengths` vectors and may hold
#     repeated or unsorted sectors (the intermediate state of a not-yet-merged fusion), plus a
#     cached fused form of itself.
#   - `FusedGradedOneTo` stores sorted parallel label/length vectors and is always
#     fused and sorted (each sector once, in sorted order).
#
# Subtypes must provide the primitive accessors `sectors`, `datalengths`, and `isdual`, plus
# `dual` and `flip` (which return the same concrete type). Everything below is derived from
# those.
abstract type AbstractGradedOneTo{S <: Sector} <: AbstractUnitRange{Int} end

# A `SectorOneTo` gives the one sector it carries. A `GradedOneTo` may repeat a sector or leave
# them unsorted, where a `FusedGradedOneTo` holds each once and in order.
"""
    sectors(g)

The sector of each block of a graded axis, in block order.
"""
function sectors end

# ========================  derived accessors  ========================

sectorlengths(g::AbstractGradedOneTo) = length.(sectors(g))
Base.first(::AbstractGradedOneTo) = 1
Base.length(g::AbstractGradedOneTo) = sum(blocklengths(g))
Base.last(g::AbstractGradedOneTo) = length(g)
# An `AbstractGradedOneTo` is 1-based and acts as its own axis, like `Base.OneTo`. Without
# this, `axes` falls back to the generic default, which returns a plain `OneTo` and drops the
# sectors.
Base.axes(g::AbstractGradedOneTo) = (g,)
BlockArrays.blocklasts(g::AbstractGradedOneTo) = cumsum(blocklengths(g))
BlockArrays.blocklength(g::AbstractGradedOneTo) = length(sectors(g))
BlockArrays.blockaxes(g::AbstractGradedOneTo) = (Block.(Base.OneTo(blocklength(g))),)
BlockArrays.eachblockaxes1(g::AbstractGradedOneTo) = eachblockaxis(g)
function BlockArrays.findblock(g::AbstractGradedOneTo, i::Integer)
    @boundscheck i in g || throw(BoundsError(g, i))
    return Block(searchsortedfirst(blocklasts(g), i))
end

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
    # The stored sectors carry no arrow, so each block axis takes the range's own.
    return [SectorOneTo(s, m, isdual(g)) for (s, m) in zip(sectors(g), datalengths(g))]
end
eachdataaxis(g::AbstractGradedOneTo) = data.(eachblockaxis(g))
eachstructureaxis(g::AbstractGradedOneTo) = structure.(eachblockaxis(g))

# ========================  conj, flip_dual  ========================
# `dual` and `flip` are concrete-type-specific (they return the same concrete type); `conj`
# and `flip_dual` derive from them.
Base.conj(g::AbstractGradedOneTo) = dual(g)
flip_dual(g::AbstractGradedOneTo) = isdual(g) ? flip(g) : g

# Bounds checking (needed for AbstractArray scalar indexing).
Base.checkindex(::Type{Bool}, g::AbstractGradedOneTo, i::Int) = 1 <= i <= length(g)

# ========================  equality, hashing  ========================
# Compared by content (sectors, data lengths, arrow), so equal-content axes compare equal
# across the concrete subtypes. `sectors` holds the non-dual sectors, so `isdual` carries the
# arrow and is compared and hashed alongside them.
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
# `showarg` rather than `summary`, so Base still supplies the element count and the `with indices`
# tail and only the type name is ours.
function Base.showarg(io::IO, g::AbstractGradedOneTo, toplevel::Bool)
    toplevel || print(io, "::")
    print(io, summary_typename(io, typeof(g)))
    return nothing
end
