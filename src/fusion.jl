using BlockArrays: Block, BlockVector, blocks
using BlockSparseArrays: mortar_axis
using SplitApplyCombine: groupcount

flip_dual(r::AbstractUnitRange) = isdual(r) ? flip(r) : r

function tensor_product(
        r1::AbstractUnitRange, r2::AbstractUnitRange, r3::AbstractUnitRange,
        rs::AbstractUnitRange...
    )
    return tensor_product(tensor_product(r1, r2), r3, rs...)
end

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
    g = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    d₁ = sector_multiplicity(sr1)
    d₂ = sector_multiplicity(sr2)
    return gradedrange(
        [
            c => (d₁ * d₂ * d) for (c, d) in zip(sectors(g), sector_multiplicities(g))
        ]
    )
end

# allow to fuse a Sector with a GradedUnitRange
function tensor_product(
        s::Union{SectorRange, SectorUnitRange}, g::GradedUnitRange
    )
    return to_gradedrange(s) ⊗ g
end

function tensor_product(
        g::GradedUnitRange, s::Union{SectorRange, SectorUnitRange}
    )
    return g ⊗ to_gradedrange(s)
end

function tensor_product(sr::SectorUnitRange, s::SectorRange)
    return sr ⊗ sectorrange(s, 1)
end

function tensor_product(s::SectorRange, sr::SectorUnitRange)
    return sectorrange(s, 1) ⊗ sr
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

function unmerged_tensor_product(a1::GradedUnitRange, a2::GradedUnitRange)
    # TODO: eltype(blocks(a1)) loses information
    T1 = eltype(a1.eachblockaxis)
    T2 = eltype(a2.eachblockaxis)
    new_axes = Base.promote_op(⊗, T1, T2)[]
    for b in blocks(a2), a in blocks(a1)
        push!(new_axes, a ⊗ b)
    end
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

function sectormergesort(g::GradedUnitRange)
    glabels = sectors(g)
    multiplicities = sector_multiplicities(g)
    dict = Dict{eltype(glabels), eltype(multiplicities)}()
    for (l, m) in zip(glabels, multiplicities)
        dict[l] = get(dict, l, 0) + m
    end

    total = sort!(collect(pairs(dict)); by = first)
    return gradedrange([c => m for (c, m) in total])
end

sectormergesort(g::AbstractUnitRange) = g

# tensor_product produces a sorted, non-dual GradedUnitRange
tensor_product(g::GradedUnitRange) = sectormergesort(flip_dual(g))

function tensor_product(g1::GradedUnitRange, g2::GradedUnitRange)
    return sectormergesort(unmerged_tensor_product(g1, g2))
end
