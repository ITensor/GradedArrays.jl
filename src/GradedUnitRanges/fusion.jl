using BlockArrays: blocklengths
using ..LabelledNumbers: LabelledInteger, label, labelled
using SplitApplyCombine: groupcount
using TensorProducts: TensorProducts, OneToOne, tensor_product

flip_dual(r::AbstractUnitRange) = isdual(r) ? flip(r) : r

function fuse_labels(x, y)
  return error(
    "`fuse_labels` not implemented for object of type `$(typeof(x))` and `$(typeof(y))`."
  )
end

function fuse_blocklengths(x::LabelledInteger, y::LabelledInteger)
  # return blocked unit range to keep non-abelian interface
  return blockedrange([labelled(x * y, fuse_labels(label(x), label(y)))])
end

unmerged_tensor_product() = OneToOne()
unmerged_tensor_product(a) = a
unmerged_tensor_product(a, ::OneToOne) = a
unmerged_tensor_product(::OneToOne, a) = a
unmerged_tensor_product(::OneToOne, ::OneToOne) = OneToOne()
unmerged_tensor_product(a1, a2) = tensor_product(a1, a2)
function unmerged_tensor_product(a1, a2, as...)
  return unmerged_tensor_product(unmerged_tensor_product(a1, a2), as...)
end

function unmerged_tensor_product(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  nested = map(Iterators.flatten((Iterators.product(blocks(a1), blocks(a2)),))) do it
    return mapreduce(length, fuse_blocklengths, it)
  end
  new_blocklengths = mapreduce(blocklengths, vcat, nested)
  return blockedrange(new_blocklengths)
end

# convention: sort GradedUnitRangeDual according to nondual blocks
function sectorsortperm(a::AbstractUnitRange)
  return Block.(sortperm(blocklabels(nondual(a))))
end

# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function groupsortperm(v; kwargs...)
  perm = sortperm(v; kwargs...)
  v_sorted = @view v[perm]
  group_lengths = collect(groupcount(identity, v_sorted))
  return BlockVector(perm, group_lengths)
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedUnitRangesExt`.
# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
function sectormergesortperm(a::AbstractUnitRange)
  return Block.(groupsortperm(blocklabels(nondual(a))))
end

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedUnitRangesExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function sectormergesort(g::AbstractGradedUnitRange)
  glabels = blocklabels(g)
  gblocklengths = blocklengths(g)
  new_blocklengths = map(sort(unique(glabels))) do la
    return labelled(sum(gblocklengths[findall(==(la), glabels)]; init=0), la)
  end
  return gradedrange(new_blocklengths)
end

sectormergesort(g::AbstractUnitRange) = g

# tensor_product produces a sorted, non-dual GradedUnitRange
TensorProducts.tensor_product(g::AbstractGradedUnitRange) = sectormergesort(flip_dual(g))

function TensorProducts.tensor_product(
  g1::AbstractGradedUnitRange, g2::AbstractGradedUnitRange
)
  return sectormergesort(unmerged_tensor_product(g1, g2))
end
