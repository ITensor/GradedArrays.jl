using BlockArrays: Block, blocks
using SplitApplyCombine: groupcount

flip_dual(r::AbstractUnitRange) = isdual(r) ? flip(r) : r

function tensor_product(sr1::SectorUnitRange, sr2::SectorUnitRange)
    return tensor_product(combine_styles(SymmetryStyle(sr1), SymmetryStyle(sr2)), sr1, sr2)
end

function tensor_product(
        ::AbelianStyle, sr1::SectorUnitRange, sr2::SectorUnitRange
    )
    s = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    return sectorrange(s, sector_multiplicity(sr1) * sector_multiplicity(sr2))
end

function tensor_product(
        ::NotAbelianStyle, sr1::SectorUnitRange, sr2::SectorUnitRange
    )
    g0 = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    return gradedrange(
        sectors(g0) .=>
            sector_multiplicity(sr1) * sector_multiplicity(sr2) .* sector_multiplicities(g0),
    )
end

# allow to fuse a Sector with a GradedUnitRange
function tensor_product(
        s::Union{SectorRange, SectorUnitRange}, g::AbstractGradedUnitRange
    )
    return to_gradedrange(s) ⊗ g
end

function tensor_product(
        g::AbstractGradedUnitRange, s::Union{SectorRange, SectorUnitRange}
    )
    return g ⊗ to_gradedrange(s)
end

function tensor_product(sr::SectorUnitRange, s::SectorRange)
    return sr ⊗ sectorrange(s, 1)
end

function tensor_product(s::SectorRange, sr::SectorUnitRange)
    return sectorrange(s, 1) ⊗ sr
end

function tensor_product(r1::AbstractUnitRange, r2::AbstractUnitRange)
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    return Base.OneTo(length(r1) * length(r2))
end

function tensor_product(
        r1::AbstractUnitRange, r2::AbstractUnitRange, r3::AbstractUnitRange,
        rs::AbstractUnitRange...,
    )
    return tensor_product(tensor_product(r1, r2), r3, rs...)
end

# unmerged_tensor_product is a private function needed in GradedArraysTensorAlgebraExt
# to get block permutation
# it is not aimed for generic use and does not support all tensor_product methods (no dispatch on SymmetryStyle)
unmerged_tensor_product() = Base.OneTo(1)
unmerged_tensor_product(a) = a
function unmerged_tensor_product(a1, a2, a3, as...)
    return unmerged_tensor_product(unmerged_tensor_product(a1, a2), a3, as...)
end

# default to tensor_product
unmerged_tensor_product(a1, a2) = a1 ⊗ a2

using BlockSparseArrays: mortar_axis
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
tensor_product(g::AbstractGradedUnitRange) = sectormergesort(flip_dual(g))

function tensor_product(
        g1::AbstractGradedUnitRange, g2::AbstractGradedUnitRange
    )
    return sectormergesort(unmerged_tensor_product(g1, g2))
end
