module GradedArraysTensorAlgebraExt

using BlockArrays: blocks
using BlockSparseArrays: BlockSparseArray, blockreshape
using GradedArrays: GradedArray
using GradedArrays.GradedUnitRanges:
  AbstractGradedUnitRange,
  blockmergesortperm,
  blocksortperm,
  flip,
  invblockperm,
  unmerged_tensor_product
using GradedArrays.SymmetrySectors: trivial
using TensorAlgebra:
  TensorAlgebra,
  AbstractBlockPermutation,
  BlockedTuple,
  FusionStyle,
  trivial_axis,
  unmatricize

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:GradedArray}) = SectorFusion()

# TBD consider heterogeneous sectors?
TensorAlgebra.trivial_axis(t::Tuple{Vararg{AbstractGradedUnitRange}}) = trivial(first(t))

function matricize_axes(
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}}
)
  @assert !isempty(blocked_axes)
  default_axis = trivial_axis(Tuple(blocked_axes))
  codomain_axes, domain_axes = blocks(blocked_axes)
  codomain_axis = unmerged_tensor_product(default_axis, codomain_axes...)
  unflipped_domain_axis = unmerged_tensor_product(default_axis, domain_axes...)
  return codomain_axis, flip(unflipped_domain_axis)
end

function TensorAlgebra.matricize(
  ::SectorFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
)
  a_perm = permutedims(a, Tuple(biperm))
  codomain_axis, domain_axis = matricize_axes(axes(a)[biperm])
  a_reshaped = blockreshape(a_perm, (codomain_axis, domain_axis))
  # Sort the blocks by sector and merge the equivalent sectors.
  return sectormergesort(a_reshaped)
end

function TensorAlgebra.unmatricize(
  ::SectorFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  # First, fuse axes to get `blockmergesortperm`.
  # Then unpermute the blocks.
  fused_axes = matricize_axes(blocked_axes)

  blockperms = blocksortperm.(fused_axes)
  sorted_axes = map((r, I) -> only(axes(r[I])), fused_axes, blockperms)

  # TODO: This is doing extra copies of the blocks,
  # use `@view a[axes_prod...]` instead.
  # That will require implementing some reindexing logic
  # for this combination of slicing.
  m_unblocked = m[sorted_axes...]
  m_blockpermed = m_unblocked[invblockperm.(blockperms)...]
  return unmatricize(FusionStyle(BlockSparseArray), m_blockpermed, blocked_axes)
end

# Sort the blocks by sector and then merge the common sectors.
function sectormergesort(a::AbstractArray)
  I = blockmergesortperm.(axes(a))
  return a[I...]
end
end
