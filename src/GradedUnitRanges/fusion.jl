using BlockArrays: Block, blocklengths, blocks
using SplitApplyCombine: groupcount
using TensorProducts: TensorProducts, ⊗, OneToOne

using ..GradedUnitRanges:
  SectorUnitRange,
  AbstractGradedUnitRange,
  nondual_sector,
  axis_cat,
  sector_axes,
  sector_multiplicities,
  sector_multiplicity
using ..SymmetrySectors: to_gradedrange

flip_dual(r::AbstractUnitRange) = isdual(r) ? flip(r) : r

# TensorProducts interface
function TensorProducts.tensor_product(sr1::SectorUnitRange, sr2::SectorUnitRange)
  # TBD dispatch on SymmetryStyle and return either SectorUnitRange or GradedUnitRange?
  s = to_gradedrange(nondual_sector(flip_dual(sr1)) ⊗ nondual_sector(flip_dual(sr2)))
  return gradedrange(
    blocklabels(s) .=>
      sector_multiplicity(sr1) * sector_multiplicity(sr2) .* sector_multiplicities(s),
  )
end

unmerged_tensor_product() = OneToOne()
unmerged_tensor_product(a) = a
unmerged_tensor_product(a, ::OneToOne) = a
unmerged_tensor_product(::OneToOne, a) = a
unmerged_tensor_product(::OneToOne, ::OneToOne) = OneToOne()
unmerged_tensor_product(a1, a2) = a1 ⊗ a2
function unmerged_tensor_product(a1, a2, as...)
  return unmerged_tensor_product(unmerged_tensor_product(a1, a2), as...)
end

function unmerged_tensor_product(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
  new_axes = map(
    splat(⊗), Iterators.flatten((Iterators.product(sector_axes(a1), sector_axes(a2)),))
  )
  return axis_cat(reduce(vcat, sector_axes.(new_axes)))
end

# convention: sort dual GradedUnitRange according to nondual blocks
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

# Used by `TensorAlgebra.unmatricize` in `GradedArraysTensorAlgebraExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function sectormergesort(g::AbstractGradedUnitRange)
  glabels = blocklabels(g)
  multiplicities = sector_multiplicities(g)
  new_blocklengths = map(sort(unique(glabels))) do la
    return la => sum(multiplicities[findall(==(la), glabels)]; init=0)
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
