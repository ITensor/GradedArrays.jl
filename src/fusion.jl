using BlockArrays: Block, blocks
using SplitApplyCombine: groupcount
using TensorProducts: TensorProducts, ⊗, OneToOne, tensor_product

flip_dual(r::AbstractUnitRange) = isdual(r) ? flip(r) : r

# TensorProducts interface
function TensorProducts.tensor_product(sr1::SectorUnitRange, sr2::SectorUnitRange)
    return tensor_product(combine_styles(SymmetryStyle(sr1), SymmetryStyle(sr2)), sr1, sr2)
end

function TensorProducts.tensor_product(
        ::AbelianStyle, sr1::SectorUnitRange, sr2::SectorUnitRange
    )
    s = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    return sectorrange(s, sector_multiplicity(sr1) * sector_multiplicity(sr2))
end

function TensorProducts.tensor_product(
        ::NotAbelianStyle, sr1::SectorUnitRange, sr2::SectorUnitRange
    )
    g0 = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    return gradedrange(
        sectors(g0) .=>
            sector_multiplicity(sr1) * sector_multiplicity(sr2) .* sector_multiplicities(g0),
    )
end

# allow to fuse a Sector with a GradedUnitRange
function TensorProducts.tensor_product(
        s::Union{AbstractSector, SectorUnitRange}, g::AbstractGradedUnitRange
    )
    return to_gradedrange(s) ⊗ g
end

function TensorProducts.tensor_product(
        g::AbstractGradedUnitRange, s::Union{AbstractSector, SectorUnitRange}
    )
    return g ⊗ to_gradedrange(s)
end

function TensorProducts.tensor_product(sr::SectorUnitRange, s::AbstractSector)
    return sr ⊗ sectorrange(s, 1)
end

function TensorProducts.tensor_product(s::AbstractSector, sr::SectorUnitRange)
    return sectorrange(s, 1) ⊗ sr
end

# unmerged_tensor_product is a private function needed in GradedArraysTensorAlgebraExt
# to get block permutation
# it is not aimed for generic use and does not support all tensor_product methods (no dispatch on SymmetryStyle)
unmerged_tensor_product() = OneToOne()
unmerged_tensor_product(a) = a
unmerged_tensor_product(a, ::OneToOne) = a
unmerged_tensor_product(::OneToOne, a) = a
unmerged_tensor_product(::OneToOne, ::OneToOne) = OneToOne()
function unmerged_tensor_product(a1, a2, as...)
    return unmerged_tensor_product(unmerged_tensor_product(a1, a2), as...)
end

# default to tensor_product
unmerged_tensor_product(a1, a2) = a1 ⊗ a2

function unmerged_tensor_product(a1::AbstractGradedUnitRange, a2::AbstractGradedUnitRange)
    new_axes = map(splat(⊗), Iterators.flatten((Iterators.product(blocks(a1), blocks(a2)),)))
    return mortar_axis(new_axes)
end

# convention: sort dual GradedUnitRange according to nondual blocks
function sectorsortperm(a::AbstractUnitRange)
    return Block.(sortperm(sectors(a)))
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
    return Block.(groupsortperm(sectors(a)))
end

# Used by `TensorAlgebra.unmatricize` in `GradedArraysTensorAlgebraExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function sectormergesort(g::AbstractGradedUnitRange)
    glabels = sectors(g)
    multiplicities = sector_multiplicities(g)
    new_blocklengths = map(sort(unique(glabels))) do la
        return la => sum(multiplicities[findall(==(la), glabels)]; init = 0)
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
