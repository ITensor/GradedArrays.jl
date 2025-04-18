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
  block,
  blockedrange,
  blocklasts,
  blocklengths,
  blocks,
  blockindex,
  findblock,
  mortar
using BlockSparseArrays:
  BlockSparseArrays,
  blockedunitrange_findblock,
  blockedunitrange_findblockindex,
  blockedunitrange_getindices

abstract type AbstractGradedUnitRange{T,BlockLasts} <:
              AbstractBlockedUnitRange{T,BlockLasts} end

struct GradedUnitRange{T,SUR<:SectorOneTo{T},BR<:AbstractUnitRange{T},BlockLasts} <:
       AbstractGradedUnitRange{T,BlockLasts}
  sector_axes::Vector{SUR}
  full_range::BR

  function GradedUnitRange{T,SUR,BR,BlockLasts}(
    sector_axes::AbstractVector{SUR}, full_range::AbstractUnitRange{T}
  ) where {T,SUR,BR,BlockLasts}
    length.(sector_axes) == blocklengths(full_range) ||
      throw(ArgumentError("sectors and range are not compatible"))
    typeof(blocklasts(full_range)) == BlockLasts ||
      throw(TypeError(:BlockLasts, "", blocklasts(full_range)))
    return new{T,SUR,BR,BlockLasts}(sector_axes, full_range)
  end
end

const GradedOneTo{T,SUR,BR,BlockLasts} =
  GradedUnitRange{T,SUR,BR,BlockLasts} where {BR<:BlockedOneTo}

function GradedUnitRange(sector_axes::AbstractVector, full_range::AbstractUnitRange)
  return GradedUnitRange{
    eltype(full_range),eltype(sector_axes),typeof(full_range),typeof(blocklasts(full_range))
  }(
    sector_axes, full_range
  )
end

# Accessors
sector_axes(g::GradedUnitRange) = g.sector_axes
unlabel_blocks(g::GradedUnitRange) = g.full_range  # TBD use full_range?

sector_multiplicities(g::GradedUnitRange) = sector_multiplicity.(sector_axes(g))

sector_type(::Type{<:GradedUnitRange{<:Any,SUR}}) where {SUR} = sector_type(SUR)

#
# Constructors
#

function axis_cat(sectors::AbstractVector{<:SectorOneTo})
  brange = blockedrange(length.(sectors))
  return GradedUnitRange(sectors, brange)
end

function axis_cat(gaxes::AbstractVector{<:GradedOneTo})
  return axis_cat(mapreduce(sector_axes, vcat, gaxes))
end

function gradedrange(
  lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}}; dual::Bool=false
)
  sectors = sectorrange.(lblocklengths, dual)
  return axis_cat(sectors)
end

### GradedUnitRange interface
dual(g::GradedUnitRange) = GradedUnitRange(dual.(sector_axes(g)), unlabel_blocks(g))

isdual(g::AbstractGradedUnitRange) = isdual(first(sector_axes(g)))  # crash for empty. Should not be an issue.

function blocklabels(g::AbstractGradedUnitRange)
  nondual_blocklabels = nondual_sector.(sector_axes(g))
  return isdual(g) ? dual.(nondual_blocklabels) : nondual_blocklabels
end

function map_blocklabels(f, g::GradedUnitRange)
  return GradedUnitRange(map_blocklabels.(f, sector_axes(g)), unlabel_blocks(g))
end

### Base interface

# needed in BlockSparseArrays
function Base.AbstractUnitRange{T}(a::AbstractGradedUnitRange{T}) where {T}
  return unlabel_blocks(a)
end

function Base.axes(ga::AbstractGradedUnitRange)
  return (axis_cat(sector_axes(ga)),)
end

# preserve axes in SubArray
Base.axes(S::Base.Slice{<:AbstractGradedUnitRange}) = (S.indices,)

function Base.show(io::IO, ::MIME"text/plain", g::AbstractGradedUnitRange)
  println(io, typeof(g))
  return print(io, join(repr.(blocks(g)), '\n'))
end

function Base.show(io::IO, g::AbstractGradedUnitRange)
  v = blocklabels(g) .=> blocklengths(g)
  return print(io, nameof(typeof(g)), '[', join(repr.(v), ", "), ']')
end

Base.first(a::AbstractGradedUnitRange) = first(unlabel_blocks(a))

### BlockArrays interface

function BlockArrays.blocklasts(a::AbstractGradedUnitRange)
  return blocklasts(unlabel_blocks(a))
end

### BlockSparseArrays interface

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

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, index::Block{1}
)
  sr = sector_axes(a)[Int(index)]
  return sectorrange(nondual_sector(sr), unlabel_blocks(a)[index], isdual(sr))
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

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockIndexRange{1}
)
  return a[block(indices)][only(indices.indices)]
end

function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::AbstractVector{<:Block{1}}
)
  # Without converting `indices` to `Vector`,
  # mapping `indices` outputs a `BlockVector`
  # which is harder to reason about.
  gblocks = map(index -> g[index], Vector(indices))
  # pass block labels to the axes of the output,
  # such that `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`
  newg = axis_cat(sectorrange.(nondual_sector.(gblocks) .=> length.(gblocks), isdual(g)))
  return mortar(gblocks, (newg,))
end

# TBD dispacth on symmetry style?
function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  return blockedunitrange_getindices(unlabel_blocks(g), indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  ::AbelianStyle, g::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  r = blockedunitrange_getindices(NotAbelianStyle(), g, indices)
  i = Int(findblock(g, first(indices)))
  j = Int(findblock(g, last(indices)))
  labels = nondual_sector.(sector_axes(g))[i:j]
  new_axes = sectorrange.(labels .=> Base.oneto.(blocklengths(r)), isdual(g))
  return GradedUnitRange(new_axes, r)
end

function BlockSparseArrays.blockedunitrange_getindices(
  ::NotAbelianStyle, g::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  return blockedunitrange_getindices(unlabel_blocks(g), indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockSlice
)
  return a[indices.block]
end

function BlockSparseArrays.blockedunitrange_getindices(
  ga::AbstractGradedUnitRange, indices::BlockRange
)
  return GradedUnitRange(sector_axes(ga)[Int.(indices)], unlabel_blocks(ga)[indices])
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockIndex{1}
)
  return a[block(indices)][blockindex(indices)]
end

### Slicing
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

# fix ambiguity
Base.getindex(g::AbstractGradedUnitRange, ::Colon) = g

function Base.getindex(a::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer})
  return blockedunitrange_getindices(SymmetryStyle(a), a, indices)
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
