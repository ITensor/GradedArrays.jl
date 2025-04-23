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
  combine_blockaxes,
  findblock,
  mortar
using BlockSparseArrays:
  BlockSparseArrays,
  blockedunitrange_findblock,
  blockedunitrange_findblockindex,
  blockedunitrange_getindices
using Compat: allequal

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
    allequal(isdual.(sector_axes)) ||
      throw(ArgumentError("all blocks must have same duality"))
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
unlabel_blocks(g::GradedUnitRange) = g.full_range  # TBD rename full_range?

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
  lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}}; isdual::Bool=false
)
  sectors = sectorrange.(lblocklengths, isdual)
  return axis_cat(sectors)
end

### GradedUnitRange interface
dual(g::GradedUnitRange) = GradedUnitRange(dual.(sector_axes(g)), unlabel_blocks(g))

isdual(g::AbstractGradedUnitRange) = isdual(first(sector_axes(g)))  # crash for empty. Should not be an issue.

# TBD remove? No use if change blocklabels convention
nondual(g::AbstractGradedUnitRange) = isdual(g) ? dual(g) : g

# TBD: change convention to nondual sectors?
function blocklabels(g::AbstractGradedUnitRange)
  nondual_blocklabels = nondual_sector.(sector_axes(g))
  return isdual(g) ? dual.(nondual_blocklabels) : nondual_blocklabels
end

function map_blocklabels(f, g::GradedUnitRange)
  return GradedUnitRange(map_blocklabels.(f, sector_axes(g)), unlabel_blocks(g))
end

### GradedUnitRange specific slicing
function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  gblocks = map(index -> g[index], Vector(indices))
  # pass block labels to the axes of the output,
  # such that `only(axes(g[indices])) isa `GradedOneTo`
  newg = axis_cat(sectorrange.(nondual_sector.(gblocks) .=> length.(gblocks), isdual(g)))
  return mortar(gblocks, (newg,))
end

function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::AbstractUnitRange{<:Integer}
)
  new_range = blockedunitrange_getindices(unlabel_blocks(g), indices)
  bf = findblock(g, first(indices))
  bl = findblock(g, last(indices))
  labels = blocklabels(nondual(g)[bf:bl])
  new_sector_axes = sectorrange.(labels .=> Base.oneto.(blocklengths(new_range)), isdual(g))
  return GradedUnitRange(new_sector_axes, new_range)
end

function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::BlockVector{<:BlockIndex{1}}
)
  blks = blocks(indices)
  newg = gradedrange(
    map(b -> nondual_sector(g[b]), block.(blks)) .=> length.(blks); isdual=isdual(g)
  )
  v = mortar(map(b -> g[b], blks), (newg,))
  return v
end

# need to drop label in some non-abelian slicing
function gradedunitrange_getindices(::NotAbelianStyle, g::AbstractUnitRange, indices)
  return blockedunitrange_getindices(unlabel_blocks(g), indices)
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

function BlockArrays.combine_blockaxes(a::GradedUnitRange, b::GradedUnitRange)
  # avoid mixing different labels
  # better to throw explicit error than silently dropping labels
  !space_isequal(a, b) && throw(ArgumentError("axes are not compatible"))
  #!space_isequal(a, b) && return combine_blockaxes(unlabel_blocks(a), unlabel_blocks(b))
  # preserve BlockArrays convention for BlockedUnitRange / BlockedOneTo
  return GradedUnitRange(
    sector_axes(a), combine_blockaxes(unlabel_blocks(a), unlabel_blocks(b))
  )
end

# preserve BlockedOneTo when possible
function BlockArrays.combine_blockaxes(a::AbstractGradedUnitRange, b::AbstractUnitRange)
  return combine_blockaxes(unlabel_blocks(a), b)
end
function BlockArrays.combine_blockaxes(a::AbstractUnitRange, b::AbstractGradedUnitRange)
  return combine_blockaxes(a, unlabel_blocks(b))
end

### BlockSparseArrays interface

# BlockSparseArray explicitly calls blockedunitrange_getindices, both Base.getindex
# and blockedunitrange_getindices must be defined

# fix ambiguity
function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockSlice
)
  return a[indices.block]
end

# used in BlockSparseArrays
function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange,
  indices::BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},
)
  return gradedunitrange_getindices(SymmetryStyle(g), g, indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  return gradedunitrange_getindices(SymmetryStyle(a), a, indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  return gradedunitrange_getindices(SymmetryStyle(g), g, indices)
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, index::Block{1}
)
  sr = sector_axes(a)[Int(index)]
  return sectorrange(nondual_sector(sr), unlabel_blocks(a)[index], isdual(sr))
end

function BlockSparseArrays.blockedunitrange_getindices(
  ga::GradedUnitRange, indices::BlockRange
)
  return GradedUnitRange(sector_axes(ga)[Int.(indices)], unlabel_blocks(ga)[indices])
end

function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::AbstractVector{<:Block{1}}
)
  # full block slicing is always possible for any fusion category

  # Without converting `indices` to `Vector`,
  # mapping `indices` outputs a `BlockVector`
  # which is harder to reason about.
  gblocks = map(index -> g[index], Vector(indices))
  # pass block labels to the axes of the output,
  # such that `only(axes(a[indices])) isa `GradedUnitRange`
  # if `a isa `GradedUnitRange`
  new_sectoraxes = sectorrange.(
    nondual_sector.(gblocks), Base.oneto.(length.(gblocks)), isdual(g)
  )
  newg = axis_cat(new_sectoraxes)
  return mortar(gblocks, (newg,))
end

# used in BlockSparseArray slicing
function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::AbstractBlockVector{<:Block{1}}
)
  blks = map(bs -> mortar(map(b -> g[b], bs)), blocks(indices))
  new_labels = map(b -> blocklabels(nondual(g))[Int.(b)], blocks(indices))
  @assert all(allequal.(new_labels))
  new_lengths = length.(blks)
  new_sector_axes = sectorrange.(first.(new_labels), Base.oneto.(new_lengths), isdual(g))
  newg = axis_cat(new_sector_axes)
  return mortar(blks, (newg,))
end

### Slicing
function Base.getindex(a::AbstractGradedUnitRange, index::Block{1})
  return blockedunitrange_getindices(a, index)
end

# impose Base.getindex and blockedunitrange_getindices to return the same output
# this version of blockedunitrange_getindices is used in BlockSparseArray slicing
function Base.getindex(
  g::AbstractGradedUnitRange,
  indices::BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}},
)
  return blockedunitrange_getindices(g, indices)
end

function Base.getindex(a::AbstractGradedUnitRange, indices::AbstractUnitRange{<:Integer})
  # BlockSparseArray explicitly calls blockedunitrange_getindices, both Base.getindex
  # and blockedunitrange_getindices must be defined
  return blockedunitrange_getindices(a, indices)
end

# fix ambiguities
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockRange{1,<:Tuple{Base.OneTo}}
)
  return blockedunitrange_getindices(a, indices)
end
function Base.getindex(
  a::AbstractGradedUnitRange, indices::BlockRange{1,<:Tuple{AbstractUnitRange{<:Integer}}}
)
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

function Base.getindex(
  a::AbstractGradedUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  return blockedunitrange_getindices(a, indices)
end
