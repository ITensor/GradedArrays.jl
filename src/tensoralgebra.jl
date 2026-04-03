using SplitApplyCombine: groupcount

flip_dual(r::AbstractUnitRange) = isdual(r) ? flip(r) : r
flip_dual(g::GradedIndices) = isdual(g) ? flip(g) : g
flip_dual(si::SectorIndices) = isdual(si) ? flip(si) : si

function tensor_product(
        r1::AbstractUnitRange, r2::AbstractUnitRange, r3::AbstractUnitRange,
        rs::AbstractUnitRange...
    )
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

function unmerged_tensor_product(a1::GradedIndices, a2::GradedIndices)
    ls1 = labels(a1)
    ms1 = sector_multiplicities(a1)
    ls2 = labels(a2)
    ms2 = sector_multiplicities(a2)
    P = promote_type(eltype(ls1), eltype(ls2))
    new_labels = P[]
    new_mults = Int[]
    for (l2, m2) in zip(ls2, ms2)
        s2 = isdual(a2) ? dual(l2) : l2
        for (l1, m1) in zip(ls1, ms1)
            s1 = isdual(a1) ? dual(l1) : l1
            fused = s1 ⊗ s2
            if fused isa SectorRange
                # Abelian: single output sector
                push!(new_labels, label(fused))
                push!(new_mults, m1 * m2)
            elseif fused isa GradedIndices
                # Non-abelian: multiple output sectors
                for (fl, fm) in zip(labels(fused), sector_multiplicities(fused))
                    push!(new_labels, fl)
                    push!(new_mults, m1 * m2 * fm)
                end
            end
        end
    end
    return GradedIndices(new_labels, new_mults, false)
end

# ========================  sorting utilities  ========================

# convention: sort dual GradedIndices according to nondual blocks
# Sort by SectorRange to use the custom isless ordering
function sectorsortperm(g::GradedIndices)
    return Block.(sortperm(SectorRange.(labels(g))))
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
function sectormergesortperm(g::GradedIndices)
    return Block.(groupsortperm(SectorRange.(labels(g))))
end

# Used by `TensorAlgebra.unmatricize` in `GradedArraysTensorAlgebraExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

function sectormergesort(g::GradedIndices)
    glabels = sectors(g)
    multiplicities = sector_multiplicities(g)
    dict = Dict{eltype(glabels), eltype(multiplicities)}()
    for (l, m) in zip(glabels, multiplicities)
        dict[l] = get(dict, l, 0) + m
    end

    # Sort by SectorRange to use the custom isless ordering
    total = sort!(collect(pairs(dict)); by = p -> SectorRange(first(p)))
    return gradedrange([c => m for (c, m) in total])
end

sectormergesort(g::AbstractUnitRange) = g

# tensor_product produces a sorted, non-dual GradedIndices
tensor_product(g::GradedIndices) = sectormergesort(flip_dual(g))

function tensor_product(g1::GradedIndices, g2::GradedIndices)
    return sectormergesort(unmerged_tensor_product(g1, g2))
end

# Allow fusing a SectorRange with a GradedIndices
function tensor_product(s::SectorRange, g::GradedIndices)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedIndices, s::SectorRange)
    return tensor_product(g, to_gradedrange(s))
end

# Allow fusing a TKS.Sector with a GradedIndices
function tensor_product(s::TKS.Sector, g::GradedIndices)
    return tensor_product(to_gradedrange(SectorRange(s)), g)
end
function tensor_product(g::GradedIndices, s::TKS.Sector)
    return tensor_product(g, to_gradedrange(SectorRange(s)))
end
