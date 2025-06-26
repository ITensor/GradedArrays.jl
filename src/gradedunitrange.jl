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
  blockfirsts,
  blockedrange,
  blocklasts,
  blocklengths,
  blocks,
  combine_blockaxes,
  findblock,
  mortar
using BlockSparseArrays:
  BlockSparseArrays, blockedunitrange_getindices, eachblockaxis, mortar_axis
using Compat: allequal

# ====================================  Definitions  =======================================

abstract type AbstractGradedUnitRange{T,BlockLasts} <:
              AbstractBlockedUnitRange{T,BlockLasts} end

struct GradedUnitRange{T,SUR<:SectorOneTo{T},BR<:AbstractUnitRange{T},BlockLasts} <:
       AbstractGradedUnitRange{T,BlockLasts}
  eachblockaxis::Vector{SUR}
  full_range::BR

  function GradedUnitRange{T,SUR,BR,BlockLasts}(
    eachblockaxis::AbstractVector{SUR}, full_range::AbstractUnitRange{T}
  ) where {T,SUR,BR,BlockLasts}
    length.(eachblockaxis) == blocklengths(full_range) ||
      throw(ArgumentError("sectors and range are not compatible"))
    allequal(isdual.(eachblockaxis)) ||
      throw(ArgumentError("all blocks must have same duality"))
    typeof(blocklasts(full_range)) == BlockLasts ||
      throw(TypeError(:BlockLasts, "", blocklasts(full_range)))
    return new{T,SUR,BR,BlockLasts}(eachblockaxis, full_range)
  end
end

const GradedOneTo{T,SUR,BR,BlockLasts} =
  GradedUnitRange{T,SUR,BR,BlockLasts} where {BR<:BlockedOneTo}

function GradedUnitRange(eachblockaxis::AbstractVector, full_range::AbstractUnitRange)
  return GradedUnitRange{
    eltype(full_range),
    eltype(eachblockaxis),
    typeof(full_range),
    typeof(blocklasts(full_range)),
  }(
    eachblockaxis, full_range
  )
end

# =====================================  Accessors  ========================================

BlockSparseArrays.eachblockaxis(g::GradedUnitRange) = g.eachblockaxis
ungrade(g::GradedUnitRange) = g.full_range

sector_multiplicities(g::GradedUnitRange) = sector_multiplicity.(eachblockaxis(g))

sector_type(::Type{<:GradedUnitRange{<:Any,SUR}}) where {SUR} = sector_type(SUR)

# ====================================  Constructors  ======================================

function BlockSparseArrays.mortar_axis(geachblockaxis::AbstractVector{<:SectorOneTo})
  brange = blockedrange(length.(geachblockaxis))
  return GradedUnitRange(geachblockaxis, brange)
end

function BlockSparseArrays.mortar_axis(gaxes::AbstractVector{<:GradedOneTo})
  return mortar_axis(mapreduce(eachblockaxis, vcat, gaxes))
end

function gradedrange(
  sectors_lengths::AbstractVector{<:Pair{<:Any,<:Integer}}; isdual::Bool=false
)
  geachblockaxis = sectorrange.(sectors_lengths, isdual)
  return mortar_axis(geachblockaxis)
end

function gradedrange(
  f::Integer, sectors_lengths::AbstractVector{<:Pair{<:Any,<:Integer}}; isdual::Bool=false
)
  geachblockaxis = sectorrange.(sectors_lengths, isdual)
  brange = blockedrange(f, length.(geachblockaxis))
  return GradedUnitRange(geachblockaxis, brange)
end

# =============================  GradedUnitRanges interface  ===============================
dual(g::GradedUnitRange) = GradedUnitRange(dual.(eachblockaxis(g)), ungrade(g))

isdual(g::AbstractGradedUnitRange) = isdual(first(eachblockaxis(g)))  # crash for empty. Should not be an issue.

function sectors(g::AbstractGradedUnitRange)
  return sector.(eachblockaxis(g))
end

flux(a::AbstractBlockedUnitRange, I::Block{1}) = flux(a[I])

function map_sectors(f, g::GradedUnitRange)
  return GradedUnitRange(map_sectors.(f, eachblockaxis(g)), ungrade(g))
end

### GradedUnitRange specific slicing
function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::AbstractVector{<:BlockIndexRange{1}}
)
  gblocks = map(index -> g[index], Vector(indices))
  # pass block labels to the axes of the output,
  # such that `only(axes(g[indices])) isa `GradedOneTo`
  newg = mortar_axis(sectorrange.(sector.(gblocks) .=> length.(gblocks), isdual(g)))
  return mortar(gblocks, (newg,))
end

function gradedunitrange_getindices(
  ::AbelianStyle, g::AbstractUnitRange, indices::AbstractUnitRange{<:Integer}
)
  new_range = blockedunitrange_getindices(ungrade(g), indices)
  bf = findblock(g, first(indices))
  bl = findblock(g, last(indices))
  new_sectors = sectors(g)[Int.(bf:bl)]
  new_eachblockaxis = sectorrange.(
    new_sectors .=> Base.oneto.(blocklengths(new_range)), isdual(g)
  )
  return GradedUnitRange(new_eachblockaxis, new_range)
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

# do not overload BlockArrays.findblock to avoid ambiguity for findblock(g, 1)
function findfirstblock_sector(g::AbstractGradedUnitRange, s)
  i = findfirst(==(s), sectors(g))
  isnothing(i) && return nothing
  return Block(i)
end

# ==================================  Base interface  ======================================

# needed in BlockSparseArrays
function Base.AbstractUnitRange{T}(a::AbstractGradedUnitRange{T}) where {T}
  return ungrade(a)
end

function Base.axes(ga::AbstractGradedUnitRange)
  return (mortar_axis(eachblockaxis(ga)),)
end

# preserve axes in SubArray
Base.axes(S::Base.Slice{<:AbstractGradedUnitRange}) = (S.indices,)

function Base.show(io::IO, ::MIME"text/plain", g::AbstractGradedUnitRange)
  println(io, typeof(g))
  return print(io, join(repr.(blocks(g)), '\n'))
end

function Base.show(io::IO, g::AbstractGradedUnitRange)
  v = sectors(g) .=> blocklengths(g)
  s = isdual(g) ? " dual " : ""
  return print(io, nameof(typeof(g)), s, '[', join(repr.(v), ", "), ']')
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
  :(Tuple{Colon,<:Any}),  # TODO replace with Kronecker range
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
  isdual(a) == isdual(b) || throw(ArgumentError("axes duality are not compatible"))
  r = combine_blockaxes(ungrade(a), ungrade(b))
  sector_axes = map(zip(blocklengths(r), blocklasts(r))) do (blength, blast)
    s_a = sector(a[findblock(a, blast)])
    if s_a != sector(b[findblock(b, blast)])  # forbid conflicting sectors
      throw(ArgumentError("sectors are not compatible"))
    end
    return sectorrange(s_a, Base.oneto(blength), isdual(a))
  end
  # preserve BlockArrays convention for BlockedUnitRange / BlockedOneTo
  return GradedUnitRange(sector_axes, r)
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
  sr = eachblockaxis(a)[Int(index)]
  return sectorrange(sector(sr), ungrade(a)[index], isdual(sr))
end

function BlockSparseArrays.blockedunitrange_getindices(
  ga::GradedUnitRange, indices::BlockRange
)
  return GradedUnitRange(eachblockaxis(ga)[Int.(indices)], ungrade(ga)[indices])
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
  newg = mortar_axis(new_sectoraxes)
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
  new_eachblockaxis = sectorrange.(first.(new_sectors), Base.oneto.(new_lengths), isdual(g))
  newg = mortar_axis(new_eachblockaxis)
  return mortar(blks, (newg,))
end

# TODO use Kronecker range
# convention: return a sectorrange for this multiplicity
function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::Tuple{Colon,<:Integer}
)
  i = last(indices)
  mult_range = blockedrange(sector_multiplicities(g))
  b = findblock(mult_range, i)
  return g[b][(:, i - first(mult_range[b]) + 1)]
end

# TODO use Kronecker range
# convention: return a gradedunitrange
function BlockSparseArrays.blockedunitrange_getindices(
  g::AbstractGradedUnitRange, indices::Tuple{Colon,<:AbstractUnitRange{<:Integer}}
)
  r = last(indices)
  mult_range = blockedrange(sector_multiplicities(g))
  bf, bl = map(i -> Int(findblock(mult_range, i)), (first(r), last(r)))
  new_first =
    blockfirsts(g)[bf] + (first(r) - first(mult_range[Block(bf)])) * length(sectors(g)[bf])
  new_axes = sectorrange.(
    sectors(g)[bf:bl] .=> blocklengths(blockedunitrange_getindices(mult_range, r)),
    isdual(g),
  )
  new_range = blockedrange(new_first, length.(new_axes))
  return GradedUnitRange(new_axes, new_range)
end

using BlockArrays: BlockedVector
function BlockSparseArrays.blockedunitrange_getindices(
  a::AbstractGradedUnitRange, indices::AbstractVector{Bool}
)
  blocked_indices = BlockedVector(indices, axes(a))
  bs = map(Base.OneTo(blocklength(blocked_indices))) do b
    binds = blocked_indices[Block(b)]
    bstart = blockfirsts(only(axes(blocked_indices)))[b]
    return findall(binds) .+ (bstart - 1)
  end
  keep = map(!isempty, bs)
  secs = sectors(a)[keep]
  bs = bs[keep]
  r = gradedrange(secs .=> length.(bs); isdual=isdual(a))
  I = mortar(bs, (r,))
  return I
end
