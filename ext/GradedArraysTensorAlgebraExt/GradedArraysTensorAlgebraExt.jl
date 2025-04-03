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
  block_merge
using DerivableInterfaces: @interface
using GradedArrays.GradedUnitRanges:
  GradedUnitRanges,
  AbstractGradedUnitRange,
  blockmergesortperm,
  blocksortperm,
  dual,
  invblockperm,
  nondual,
  unmerged_tensor_product
using LinearAlgebra: Adjoint, Transpose
using TensorAlgebra:
  TensorAlgebra, FusionStyle, BlockReshapeFusion, SectorFusion, fusedims, splitdims
using TensorProducts: OneToOne

#=
    reducewhile(f, op, collection, state)

reducewhile(x -> length(x) < 3, vcat, ["a", "b", "c", "d"], 2; init=String[]) ==
  (["b", "c"], 4)
=#
function reducewhile(f, op, collection, state; init)
  prev_result = init
  prev_state = state
  result = prev_result
  while f(result)
    prev_result = result
    prev_state = state
    value_and_state = iterate(collection, state)
    isnothing(value_and_state) && break
    value, state = value_and_state
    result = op(result, value)
  end
  return prev_result, prev_state
end

#=
    groupreducewhile(f, op, collection, ngroups)

groupreducewhile((i, x) -> length(x) ≤ i, vcat, ["a", "b", "c", "d", "e", "f"], 3; init=String[]) ==
  (["a"], ["b", "c"], ["d", "e", "f"])
=#
function groupreducewhile(f, op, collection, ngroups; init)
  state = firstindex(collection)
  return ntuple(ngroups) do group_number
    result, state = reducewhile(x -> f(group_number, x), op, collection, state; init)
    return result
  end
end

TensorAlgebra.FusionStyle(::AbstractGradedUnitRange) = SectorFusion()

# Sort the blocks by sector and then merge the common sectors.
function block_mergesort(a::AbstractArray)
  I = blockmergesortperm.(axes(a))
  return a[I...]
end

function TensorAlgebra.fusedims(
  ::SectorFusion, a::AbstractArray, merged_axes::AbstractUnitRange...
)
  # First perform a fusion using a block reshape.
  # TODO avoid groupreducewhile. Require refactor of fusedims.
  unmerged_axes = groupreducewhile(
    unmerged_tensor_product, axes(a), length(merged_axes); init=OneToOne()
  ) do i, axis
    return length(axis) ≤ length(merged_axes[i])
  end

  a_reshaped = fusedims(BlockReshapeFusion(), a, unmerged_axes...)
  # Sort the blocks by sector and merge the equivalent sectors.
  return block_mergesort(a_reshaped)
end

function TensorAlgebra.splitdims(
  ::SectorFusion, a::AbstractArray, split_axes::AbstractUnitRange...
)
  # First, fuse axes to get `blockmergesortperm`.
  # Then unpermute the blocks.
  axes_prod = groupreducewhile(
    unmerged_tensor_product, split_axes, ndims(a); init=OneToOne()
  ) do i, axis
    return length(axis) ≤ length(axes(a, i))
  end
  blockperms = blocksortperm.(axes_prod)
  sorted_axes = map((r, I) -> only(axes(r[I])), axes_prod, blockperms)

  # TODO: This is doing extra copies of the blocks,
  # use `@view a[axes_prod...]` instead.
  # That will require implementing some reindexing logic
  # for this combination of slicing.
  a_unblocked = a[sorted_axes...]
  a_blockpermed = a_unblocked[invblockperm.(blockperms)...]
  return splitdims(BlockReshapeFusion(), a_blockpermed, split_axes...)
end

end
