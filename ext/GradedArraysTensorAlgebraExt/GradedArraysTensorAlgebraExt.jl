module GradedArraysTensorAlgebraExt

using BlockArrays: Block, BlockIndexRange, blockedrange, blocks
using BlockSparseArrays:
  BlockSparseArrays,
  AbstractBlockSparseArray,
  AbstractBlockSparseArrayInterface,
  BlockSparseArray,
  BlockSparseArrayInterface,
  BlockSparseMatrix,
  BlockSparseVector,
  block_merge,
  blockreshape
using DerivableInterfaces: @interface
using GradedArrays.GradedUnitRanges:
  GradedUnitRanges,
  AbstractGradedUnitRange,
  blockmergesortperm,
  blocksortperm,
  dual,
  flip,
  invblockperm,
  nondual,
  unmerged_tensor_product
using GradedArrays.SymmetrySectors: trivial
using LinearAlgebra: Adjoint, Transpose
using TensorAlgebra:
  TensorAlgebra,
  AbstractBlockPermutation,
  BlockedTuple,
  FusionStyle,
  matricize,
  trivial_axis,
  unmatricize

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::AbstractGradedUnitRange) = SectorFusion()

function TensorAlgebra.FusionStyle(::AbstractBlockSparseArray, ::SectorFusion)
  return SectorFusion()
end

# TODO consider heterogeneous sectors?
TensorAlgebra.trivial_axis(t::Tuple{Vararg{AbstractGradedUnitRange}}) = trivial(first(t))

maybe_trivial_axis(axes::Tuple, ::Tuple) = unmerged_tensor_product(axes...)
maybe_trivial_axis(::Tuple{}, axes::Tuple) = trivial_axis(axes)

function row_and_column_axes(
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}}
)
  codomain_axes, domain_axes = blocks(blocked_axes)
  @assert !(isempty(codomain_axes) && isempty(domain_axes))
  row_axis = maybe_trivial_axis(codomain_axes, domain_axes)
  unflipped_col_axis = maybe_trivial_axis(domain_axes, codomain_axes)
  return row_axis, flip(unflipped_col_axis)
end

function TensorAlgebra.matricize(
  ::SectorFusion, a::AbstractArray, biperm::AbstractBlockPermutation{2}
)
  a_perm = permutedims(a, Tuple(biperm))
  row_axis, col_axis = row_and_column_axes(axes(a)[biperm])
  a_reshaped = blockreshape(a_perm, (row_axis, col_axis))
  # Sort the blocks by sector and merge the equivalent sectors.
  return block_mergesort(a_reshaped)
end

function TensorAlgebra.unmatricize(
  ::SectorFusion,
  m::AbstractMatrix,
  blocked_axes::BlockedTuple{2,<:Any,<:Tuple{Vararg{AbstractUnitRange}}},
)
  # First, fuse axes to get `blockmergesortperm`.
  # Then unpermute the blocks.
  row_col_axes = row_and_column_axes(blocked_axes)

  blockperms = blocksortperm.(row_col_axes)
  sorted_axes = map((r, I) -> only(axes(r[I])), row_col_axes, blockperms)

  # TODO: This is doing extra copies of the blocks,
  # use `@view a[axes_prod...]` instead.
  # That will require implementing some reindexing logic
  # for this combination of slicing.
  m_unblocked = m[sorted_axes...]
  m_blockpermed = m_unblocked[invblockperm.(blockperms)...]
  return unmatricize(FusionStyle(m, ()), m_blockpermed, blocked_axes)
end

# Sort the blocks by sector and then merge the common sectors.
function block_mergesort(a::AbstractArray)
  I = blockmergesortperm.(axes(a))
  return a[I...]
end
end
