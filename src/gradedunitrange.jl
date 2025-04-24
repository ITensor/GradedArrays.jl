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

# ====================================  Definitions  =======================================

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

# =====================================  Accessors  ========================================

sector_axes(g::GradedUnitRange) = g.sector_axes
ungrade(g::GradedUnitRange) = g.full_range

sector_multiplicities(g::GradedUnitRange) = sector_multiplicity.(sector_axes(g))

sector_type(::Type{<:GradedUnitRange{<:Any,SUR}}) where {SUR} = sector_type(SUR)

# ====================================  Constructors  ======================================

function axis_cat(sectors::AbstractVector{<:SectorOneTo})
  brange = blockedrange(length.(sectors))
  return GradedUnitRange(sectors, brange)
end

function axis_cat(gaxes::AbstractVector{<:GradedOneTo})
  return axis_cat(mapreduce(sector_axes, vcat, gaxes))
end

function gradedrange(
  sectors_lengths::AbstractVector{<:Pair{<:Any,<:Integer}}; isdual::Bool=false
)
  gsector_axes = sectorrange.(sectors_lengths, isdual)
  return axis_cat(gsector_axes)
end

# =============================  GradedUnitRanges interface  ===============================
dual(g::GradedUnitRange) = GradedUnitRange(dual.(sector_axes(g)), ungrade(g))

isdual(g::AbstractGradedUnitRange) = isdual(first(sector_axes(g)))  # crash for empty. Should not be an issue.

function sectors(g::AbstractGradedUnitRange)
  return sector.(sector_axes(g))
end

function map_sectors(f, g::GradedUnitRange)
  return GradedUnitRange(map_sectors.(f, sector_axes(g)), ungrade(g))
end

### GradedUnitRange specific slicing
function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  gblocks = map(index -> g[index], Vector(indices))
  # pass block labels to the axes of the output,
  # such that `only(axes(g[indices])) isa `GradedOneTo`
  newg = axis_cat(sectorrange.(sector.(gblocks) .=> length.(gblocks), isdual(g)))
  return mortar(gblocks, (newg,))
end

function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::AbstractUnitRange{<:Integer}
)
  new_range = blockedunitrange_getindices(ungrade(g), indices)
  bf = findblock(g, first(indices))
  bl = findblock(g, last(indices))
  new_sectors = sectors(g)[Int.(bf:bl)]
  new_sector_axes = sectorrange.(
    new_sectors .=> Base.oneto.(blocklengths(new_range)), isdual(g)
  )
  return GradedUnitRange(new_sector_axes, new_range)
end

function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::BlockVector{<:BlockIndex{1}}
)
  blks = blocks(indices)
  newg = gradedrange(
    map(b -> sector(g[b]), block.(blks)) .=> length.(blks); isdual=isdual(g)
  )
  v = mortar(map(b -> g[b], blks), (newg,))
  return v
end

# need to drop label in some non-abelian slicing
function gradedunitrange_getindices(::NotAbelianStyle, g::AbstractUnitRange, indices)
  return blockedunitrange_getindices(ungrade(g), indices)
end

# ==================================  Base interface  ======================================

# needed in BlockSparseArrays
function Base.AbstractUnitRange{T}(a::AbstractGradedUnitRange{T}) where {T}
  return ungrade(a)
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
  v = sectors(g) .=> blocklengths(g)
  return print(io, nameof(typeof(g)), '[', join(repr.(v), ", "), ']')
end

Base.first(a::AbstractGradedUnitRange) = first(ungrade(a))

# BlockSparseArray explicitly calls blockedunitrange_getindices, both Base.getindex
# and blockedunitrange_getindices must be defined.
# Also impose Base.getindex and blockedunitrange_getindices to return the same output
for T in [
  :(AbstractUnitRange{<:Integer}),
  :(AbstractVector{<:Block{1}}),
  :(AbstractVector{<:BlockIndexRange{1}}),
  :(Block{1}),
  :(BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}),
  :(BlockRange{1,<:Tuple{Base.OneTo}}),
  :(BlockRange{1,<:Tuple{AbstractUnitRange{<:Integer}}}),
  :(BlockSlice),
]
  @eval Base.getindex(g::AbstractGradedUnitRange, indices::$T) = blockedunitrange_getindices(
    g, indices
  )
end

# ================================  BlockArrays interface  =================================

function BlockArrays.blocklasts(a::AbstractGradedUnitRange)
  return blocklasts(ungrade(a))
end

function BlockArrays.combine_blockaxes(a::GradedUnitRange, b::GradedUnitRange)
  # avoid mixing different labels
  # better to throw explicit error than silently dropping labels
  !space_isequal(a, b) && throw(ArgumentError("axes are not compatible"))
  #!space_isequal(a, b) && return combine_blockaxes(ungrade(a), ungrade(b))
  # preserve BlockArrays convention for BlockedUnitRange / BlockedOneTo
  return GradedUnitRange(sector_axes(a), combine_blockaxes(ungrade(a), ungrade(b)))
end

# preserve BlockedOneTo when possible
function BlockArrays.combine_blockaxes(a::AbstractGradedUnitRange, b::AbstractUnitRange)
  return combine_blockaxes(ungrade(a), b)
end
function BlockArrays.combine_blockaxes(a::AbstractUnitRange, b::AbstractGradedUnitRange)
  return combine_blockaxes(a, ungrade(b))
end

# ============================  BlockSparseArrays interface  ===============================

# BlockSparseArray explicitly calls blockedunitrange_getindices, both Base.getindex
# and blockedunitrange_getindices must be defined

# fix ambiguity
function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::BlockSlice
)
  return a[indices.block]
end

for T in [
  :(AbstractUnitRange{<:Integer}),
  :(AbstractVector{<:BlockIndexRange{1}}),
  :(BlockVector{<:BlockIndex{1},<:Vector{<:BlockIndexRange{1}}}),
]
  @eval BlockSparseArrays.blockedunitrange_getindices(g::AbstractGradedUnitRange, indices::$T) = gradedunitrange_getindices(
    SymmetryStyle(g), g, indices
  )
end

function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, index::Block{1}
)
  sr = sector_axes(a)[Int(index)]
  return sectorrange(sector(sr), ungrade(a)[index], isdual(sr))
end

function BlockSparseArrays.blockedunitrange_getindices(
  ga::GradedUnitRange, indices::BlockRange
)
  return GradedUnitRange(sector_axes(ga)[Int.(indices)], ungrade(ga)[indices])
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
  new_sectoraxes = sectorrange.(sector.(gblocks), Base.oneto.(length.(gblocks)), isdual(g))
  newg = axis_cat(new_sectoraxes)
  return mortar(gblocks, (newg,))
end

# used in BlockSparseArray slicing
function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::AbstractBlockVector{<:Block{1}}
)
  #TODO use one map
  blks = map(bs -> mortar(map(b -> g[b], bs)), blocks(indices))
  new_sectors = map(b -> sectors(g)[Int.(b)], blocks(indices))
  @assert all(allequal.(new_sectors))
  new_lengths = length.(blks)
  new_sector_axes = sectorrange.(first.(new_sectors), Base.oneto.(new_lengths), isdual(g))
  newg = axis_cat(new_sector_axes)
  return mortar(blks, (newg,))
end
