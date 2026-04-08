using SplitApplyCombine: groupcount

function tensor_product(r1, r2, r3, rs...)
    return tensor_product(tensor_product(r1, r2), r3, rs...)
end

# ========================  unmerged_tensor_product  ========================

unmerged_tensor_product() = Base.OneTo(1)
unmerged_tensor_product(a) = a
function unmerged_tensor_product(a1, a2, a3, as...)
    return unmerged_tensor_product(unmerged_tensor_product(a1, a2), a3, as...)
end

# default to tensor_product
unmerged_tensor_product(a1, a2) = a1 ⊗ a2

function unmerged_tensor_product(a1::GradedUnitRange, a2::GradedUnitRange)
    ea1 = eachblockaxis(a1)
    ea2 = eachblockaxis(a2)
    T = Base.promote_op(⊗, eltype(ea1), eltype(ea2))
    new_axes = T[]
    for b in ea2, a in ea1
        push!(new_axes, a ⊗ b)
    end
    return mortar_axis(new_axes)
end

# ========================  sorting utilities  ========================

# convention: sort dual GradedUnitRange according to nondual blocks
# Sort by SectorRange to use the custom isless ordering
function sectorsortperm(g::GradedUnitRange)
    return Block.(sortperm(sectors(g)))
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
# Sort by SectorRange to use the custom isless ordering
function sectormergesortperm(g::GradedUnitRange)
    return Block.(groupsortperm(sectors(g)))
end

# Used by `TensorAlgebra.unmatricize` in `GradedArraysTensorAlgebraExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

# Returns a Vector{BlockIndexRange{1}} mapping each block of fine_ax (in original order)
# to its position (block + subrange) within the merged axis merged_ax, given the block
# permutation blockperm used to sort and merge fine_ax into merged_ax.
# Requires that blocks of fine_ax subdivide blocks of merged_ax.
function invblockmergeperm(fine_ax::GradedUnitRange, blockperm, merged_ax::GradedUnitRange)
    n = blocklength(fine_ax)
    fine_bls = blocklengths(fine_ax)
    merged_bls = blocklengths(merged_ax)
    bir_type = Base.promote_op(getindex, Block{1, Int}, UnitRange{Int})
    J = Vector{bir_type}(undef, n)
    j = 1
    offset = 0
    for k′ in 1:n
        k = Int(blockperm[k′])
        size_k = fine_bls[k]
        merged_block_size = merged_bls[j]
        offset + size_k ≤ merged_block_size ||
            throw(ArgumentError("fine_ax blocks do not subdivide merged_ax blocks"))
        J[k] = Block(j)[(offset + 1):(offset + size_k)]
        offset += size_k
        if offset == merged_block_size
            j += 1
            offset = 0
        end
    end
    return J
end

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

# Sort the blocks of an array by sector and merge common sectors.
function sectormergesort(a::AbelianArray)
    I = sectormergesortperm.(axes(a))
    return a[I...]
end

# tensor_product produces a sorted, non-dual GradedUnitRange
tensor_product(g::GradedUnitRange) = sectormergesort(flip_dual(g))

function tensor_product(g1::GradedUnitRange, g2::GradedUnitRange)
    return sectormergesort(unmerged_tensor_product(g1, g2))
end

# ========================  mixed-type tensor_product  ========================
# Convert to a common type via `to_gradedrange` and dispatch to
# tensor_product(::GradedUnitRange, ::GradedUnitRange).

# SectorUnitRange ↔ GradedUnitRange
function tensor_product(s::SectorUnitRange, g::GradedUnitRange)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedUnitRange, s::SectorUnitRange)
    return tensor_product(g, to_gradedrange(s))
end

# SectorRange ↔ GradedUnitRange
function tensor_product(s::SectorRange, g::GradedUnitRange)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedUnitRange, s::SectorRange)
    return tensor_product(g, to_gradedrange(s))
end

# SectorRange ↔ SectorUnitRange
function tensor_product(s::SectorRange, si::SectorUnitRange)
    return tensor_product(to_gradedrange(s), to_gradedrange(si))
end
function tensor_product(si::SectorUnitRange, s::SectorRange)
    return tensor_product(to_gradedrange(si), to_gradedrange(s))
end

# TKS.Sector ↔ GradedUnitRange
function tensor_product(s::TKS.Sector, g::GradedUnitRange)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedUnitRange, s::TKS.Sector)
    return tensor_product(g, to_gradedrange(s))
end

# TKS.Sector ↔ SectorUnitRange
function tensor_product(s::TKS.Sector, si::SectorUnitRange)
    return tensor_product(to_gradedrange(s), to_gradedrange(si))
end
function tensor_product(si::SectorUnitRange, s::TKS.Sector)
    return tensor_product(to_gradedrange(si), to_gradedrange(s))
end
