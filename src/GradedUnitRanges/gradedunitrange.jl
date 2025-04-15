using BlockArrays:
  BlockArrays,
  AbstractBlockVector,
  AbstractBlockedUnitRange,
  Block,
  BlockIndex,
  BlockIndexRange,
  BlockRange,
  BlockSlice,
  BlockVector,
  BlockedOneTo,
  BlockedUnitRange,
  block,
  blockedrange,
  blockfirsts,
  blockisequal,
  blocklasts,
  blocklength,
  blocklengths,
  blocks,
  blockindex,
  combine_blockaxes,
  mortar,
  sortedunion
using BlockSparseArrays:
  BlockSparseArrays,
  _blocks,
  blockedunitrange_findblock,
  blockedunitrange_findblockindex,
  blockedunitrange_getindices
using Compat: allequal
using FillArrays: Fill

abstract type AbstractGradedUnitRange{T,BlockLasts} <:
              AbstractBlockedUnitRange{T,BlockLasts} end

struct GradedUnitRange{
  T,BlockLasts,BR<:AbstractBlockedUnitRange{T,BlockLasts},SUR<:SectorOneTo{T}
} <: AbstractGradedUnitRange{T,BlockLasts}
  sectors::Vector{SUR}
  range::BR

  function GradedUnitRange(
    sectors::AbstractVector, range::AbstractBlockedUnitRange{T,BlockLasts}
  ) where {T,BlockLasts}
    @assert length.(sectors) == blocklengths(range)
    return new{T,BlockLasts,typeof(range),eltype(sectors)}(sectors, range)
  end
end

const GradedOneTo{T,BlockLasts,BR,SUR} =
  GradedUnitRange{T,BlockLasts,BR,SUR} where {BR<:BlockedOneTo}

# Accessors
sectors(g::GradedUnitRange) = g.sectors
unlabel_blocks(g::GradedUnitRange) = g.range

#
# Constructors
#

#= TBD remove?
# assume that lasts is sorted, no checks carried out here
function GradedOneTo(lasts::BlockLasts) where {T<:Integer,BlockLasts<:AbstractVector{T}}
Base.require_one_based_indexing(lasts)
isempty(lasts) || first(lasts) >= 0 || throw(ArgumentError("blocklasts must be >= 0"))
return new{T,BlockLasts}(lasts)
end
function GradedOneTo(lasts::BlockLasts) where {T<:Integer,BlockLasts<:Tuple{T,Vararg{T}}}
first(lasts) >= 0 || throw(ArgumentError("blocklasts must be >= 0"))
return new{T,BlockLasts}(lasts)
end
=#

sector_multiplicity(g::GradedUnitRange) = sector_multiplicity.(sectors(g))

sector_type(x) = sector_type(typeof(x))
sector_type(::Type) = error("Not implemented")
sector_type(::Type{<:GradedUnitRange{<:Any,<:Any,<:Any,SUR}}) where {SUR} = sector_type(SUR)

function rangemortar(sectors::Vector{<:SectorOneTo})
  brange = blockedrange(length.(sectors))
  return GradedUnitRange(sectors, brange)
end

function gradedrange(
  lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}}; dual::Bool=false
)
  sectors = sectorunitrange.(lblocklengths, dual)
  return rangemortar(sectors)
end

dual(g::GradedUnitRange) = GradedUnitRange(dual.(sectors(g)), unlabel_blocks(g))

function Base.show(io::IO, ::MIME"text/plain", g::AbstractGradedUnitRange)
  println(io, typeof(g))
  return print(io, join(repr.(blocks(g)), '\n'))
end

function Base.show(io::IO, g::AbstractGradedUnitRange)
  v = blocklabels(g) .=> blocklengths(g)
  return print(io, nameof(typeof(g)), '[', join(repr.(v), ", "), ']')
end

# == is just a range comparison that ignores labels. Need dedicated function to check equality.
struct NoLabel end
blocklabels(r::AbstractUnitRange) = Fill(NoLabel(), blocklength(r))

function space_isequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return (isdual(a1) == isdual(a2)) &&
         blocklabels(a1) == blocklabels(a2) &&
         blockisequal(a1, a2)
end

# needed in BlockSparseArrays
function Base.AbstractUnitRange{T}(a::AbstractGradedUnitRange{T}) where {T}
  return unlabel_blocks(a)
end

Base.last(a::AbstractGradedUnitRange) = last(unlabel_blocks(a))

# TODO: Use `TypeParameterAccessors`.
Base.eltype(::Type{<:GradedUnitRange{T}}) where {T} = T

function labelled_blocks(a::BlockedOneTo, labels)
  # TODO: Use `blocklasts(a)`? That might
  # cause a recursive loop.
  return GradedOneTo(labelled.(a.lasts, labels))
end
function labelled_blocks(a::BlockedUnitRange, labels)
  # TODO: Use `first(a)` and `blocklasts(a)`? Those might
  # cause a recursive loop.
  return GradedUnitRange(labelled(a.first, labels[1]), labelled.(a.lasts, labels))
end

function BlockArrays.findblock(a::AbstractGradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function BlockSparseArrays.blockedunitrange_findblock(
  a::AbstractGradedUnitRange, index::Integer
)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function BlockSparseArrays.blockedunitrange_findblockindex(
  a::AbstractGradedUnitRange, index::Integer
)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

function BlockArrays.findblockindex(a::AbstractGradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

## Block label interface

# Internal function
function get_label(a::AbstractUnitRange, index::Block{1})
  return label(blocklasts(a)[Int(index)])
end

# Internal function
function get_label(a::AbstractUnitRange, index::Integer)
  return get_label(a, blockedunitrange_findblock(a, index))
end

function blocklabels(a::AbstractBlockVector)
  return map(BlockRange(a)) do block
    return label(@view(a[block]))
  end
end

function blocklabels(a::AbstractBlockedUnitRange)
  return map(sr -> only(blocklabels(sr)), sectors(a))
end

## BlockedUnitRange interface

function Base.axes(ga::AbstractGradedUnitRange)
  return (GradedUnitRange(sectors(ga), blockedrange(blocklengths(ga))),)
end

function gradedunitrange_blockfirsts(a::AbstractGradedUnitRange)
  return blockfirsts(unlabel_blocks(a))
end
function BlockArrays.blockfirsts(a::AbstractGradedUnitRange)
  return gradedunitrange_blockfirsts(a)
end

function BlockArrays.blocklasts(a::AbstractGradedUnitRange)
  return blocklasts(unlabel_blocks(a))
end

function BlockArrays.blocklengths(a::AbstractGradedUnitRange)
  return blocklengths(unlabel_blocks(a))
end

function Base.first(a::AbstractGradedUnitRange)
  return first(unlabel_blocks(a))
end

Base.iterate(a::AbstractGradedUnitRange) = iterate(unlabel_blocks(a))
Base.iterate(a::AbstractGradedUnitRange, i) = iterate(unlabel_blocks(a), i)

function firstblockindices(a::AbstractGradedUnitRange)
  return labelled.(firstblockindices(unlabel_blocks(a)), blocklabels(a))
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, index::Block{1}
)
  sr = sectors(a)[Int(index)]
  return sectorunitrange(nondual_sector(sr), unlabel_blocks(a)[index], isdual(sr))
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::Vector{<:Integer}
)
  return map(index -> a[index], indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange,
  indices::BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},
)
  return mortar(map(b -> a[b], blocks(indices)))
end

function BlockSparseArrays.blockedunitrange_getindices(a::AbstractGradedUnitRange, index)
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockIndexRange{1}
)
  return a[block(indices)][only(indices.indices)]
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::AbstractVector{<:Union{Block{1},BlockIndexRange{1}}}
)
  # Without converting `indices` to `Vector`,
  # mapping `indices` outputs a `BlockVector`
  # which is harder to reason about.
  blocks = map(index -> a[index], Vector(indices))
  # We pass `length.(blocks)` to `mortar` in order
  # to pass block labels to the axes of the output,
  # if they exist. This makes it so that
  # `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`, for example.
  return mortar(blocks, length.(blocks))
end

# The block labels of the corresponding slice.
function blocklabels(a::AbstractUnitRange, indices)
  return map(_blocks(a, indices)) do block
    return label(a[block])
  end
end

function BlockSparseArrays.blockedunitrange_getindices(
  ga::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  return blockedunitrange_getindices(unlabel_blocks(ga), indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockSlice
)
  return a[indices.block]
end

function BlockSparseArrays.blockedunitrange_getindices(
  ga::AbstractGradedUnitRange, indices::BlockRange
)
  return labelled_blocks(unlabel_blocks(ga)[indices], blocklabels(ga, indices))
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockIndex{1}
)
  return a[block(indices)][blockindex(indices)]
end

function Base.getindex(a::AbstractGradedUnitRange, index::Integer)
  return unlabel_blocks(a)[index]
end

function Base.getindex(a::AbstractGradedUnitRange, index::Block{1})
  return blockedunitrange_getindices(a, index)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::BlockIndexRange{1})
  return blockedunitrange_getindices(a, indices)
end

#getindex(::GradedUnitRanges.AbstractGradedUnitRange, ::BlockArrays.BlockIndexRange{1, R, I} where {R<:Tuple{AbstractUnitRange{<:Integer}}, I<:Tuple{Integer}})

# fix ambiguities
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockArrays.BlockRange{1,<:Tuple{Base.OneTo}}
)
  return blockedunitrange_getindices(a, indices)
end
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return blockedunitrange_getindices(a, indices)
end

# Fix ambiguity error with BlockArrays.jl.
function Base.getindex(a::AbstractGradedUnitRange, indices::BlockIndex{1})
  return blockedunitrange_getindices(a, indices)
end

# Fixes ambiguity issues with:
# ```julia
# getindex(::BlockedUnitRange, ::BlockSlice)
# getindex(::GradedUnitRange, ::AbstractUnitRange{<:Integer})
# getindex(::GradedUnitRange, ::Any)
# getindex(::AbstractUnitRange, ::AbstractUnitRange{<:Integer})
# ```
function Base.getindex(a::AbstractGradedUnitRange, indices::BlockSlice)
  return blockedunitrange_getindices(a, indices)
end

# Fix ambiguity error with BlockArrays.jl.
function Base.getindex(a::AbstractGradedUnitRange, indices::AbstractVector{<:Block{1}})
  return blockedunitrange_getindices(a, indices)
end

# Fix ambiguity error with BlockArrays.jl.
function Base.getindex(
  a::AbstractGradedUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  return blockedunitrange_getindices(a, indices)
end

# Fix ambiguity error with BlockArrays.jl.
function Base.getindex(a::AbstractGradedUnitRange, indices::AbstractVector{<:BlockIndex{1}})
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::AbstractGradedUnitRange, indices)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer})
  return blockedunitrange_getindices(a, indices)
end

# This fixes an issue that `combine_blockaxes` was promoting
# the element type of the axes to `Integer` in broadcasting operations
# that mixed dense and graded axes.
# TODO: Maybe come up with a more general solution.
function BlockArrays.combine_blockaxes(
  a1::AbstractGradedUnitRange{T}, a2::AbstractUnitRange{T}
) where {T<:Integer}
  combined_blocklasts = sort!(union(unlabel.(blocklasts(a1)), blocklasts(a2)))
  return BlockedOneTo(combined_blocklasts)
end
function BlockArrays.combine_blockaxes(
  a1::AbstractUnitRange{T}, a2::AbstractGradedUnitRange{T}
) where {T<:Integer}
  return BlockArrays.combine_blockaxes(a2, a1)
end

# preserve labels inside combine_blockaxes
function BlockArrays.combine_blockaxes(a::GradedOneTo, b::GradedOneTo)
  return GradedUnitRange(sortedunion(blocklasts(a), blocklasts(b)))
end
function BlockArrays.combine_blockaxes(a::GradedUnitRange, b::GradedUnitRange)
  new_blocklasts = sortedunion(blocklasts(a), blocklasts(b))
  new_first = labelled(oneunit(eltype(new_blocklasts)), label(first(new_blocklasts)))
  return GradedUnitRange(new_first, new_blocklasts)
end

# preserve axes in SubArray
Base.axes(S::Base.Slice{<:AbstractGradedUnitRange}) = (S.indices,)

# Version of length that checks that all blocks have the same label
# and returns a labelled length with that label.
function labelled_length(a::AbstractBlockVector{<:Integer})
  blocklabels = label.(blocks(a))
  @assert allequal(blocklabels)
  return labelled(unlabel(length(a)), first(blocklabels))
end

# TODO: Make sure this handles block labels (AbstractGradedUnitRange) correctly.
# TODO: Make a special case for `BlockedVector{<:Block{1},<:BlockRange{1}}`?
# For example:
# ```julia
# blocklengths = map(bs -> sum(b -> length(a[b]), bs), blocks(indices))
# return blockedrange(blocklengths)
# ```
function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::AbstractBlockVector{<:Block{1}}
)
  blks = map(bs -> mortar(map(b -> a[b], bs)), blocks(indices))
  # We pass `length.(blks)` to `mortar` in order
  # to pass block labels to the axes of the output,
  # if they exist. This makes it so that
  # `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`, for example.
  return mortar(blks, labelled_length.(blks))
end

map_blocklabels(::Any, a::AbstractUnitRange) = a
function map_blocklabels(f, g::AbstractGradedUnitRange)
  # use labelled_blocks to preserve GradedUnitRange
  return GradedUnitRange(map_blocklabels.(f, sectors(g)), unlabel_blocks(g))
end
