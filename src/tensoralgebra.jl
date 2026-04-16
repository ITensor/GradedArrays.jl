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
unmerged_tensor_product(a1, a2) = tensor_product(a1, a2)

function unmerged_tensor_product(a1::GradedOneTo, a2::GradedOneTo)
    ea1 = eachblockaxis(a1)
    ea2 = eachblockaxis(a2)
    T = Base.promote_op(tensor_product, eltype(ea1), eltype(ea2))
    new_axes = T[]
    for b in ea2, a in ea1
        push!(new_axes, tensor_product(a, b))
    end
    return mortar_axis(new_axes)
end

# ========================  sorting utilities  ========================

# convention: sort dual GradedOneTo according to nondual blocks
# Sort by SectorRange to use the custom isless ordering
function sectorsortperm(g::GradedOneTo)
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

# Used by `TensorAlgebra.splitdims` in `BlockSparseArraysGradedOneTosExt`.
# Get the permutation for sorting, then group by common elements.
# groupsortperm([2, 1, 2, 3]) == [[2], [1, 3], [4]]
# Sort by SectorRange to use the custom isless ordering
function sectormergesortperm(g::GradedOneTo)
    return Block.(groupsortperm(sectors(g)))
end

# Used by `TensorAlgebra.unmatricize` in `GradedArraysTensorAlgebraExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

# Returns a Vector{BlockIndexRange{1}} mapping each block of fine_ax (in original order)
# to its position (block + subrange) within the merged axis merged_ax, given the block
# permutation blockperm used to sort and merge fine_ax into merged_ax.
# Requires that blocks of fine_ax subdivide blocks of merged_ax.
function invblockmergeperm(fine_ax::GradedOneTo, blockperm, merged_ax::GradedOneTo)
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

function sectormergesort(g::GradedOneTo)
    glabels = sectors(g)
    slens = datalengths(g)
    dict = Dict{eltype(glabels), eltype(slens)}()
    for (l, m) in zip(glabels, slens)
        dict[l] = get(dict, l, 0) + m
    end

    total = sort!(collect(pairs(dict)); by = first)
    return gradedrange([c => m for (c, m) in total])
end

# Sort the blocks of an array by sector and merge common sectors.
function sectormergesort(a::AbelianGradedArray)
    I = sectormergesortperm.(axes(a))
    return a[I...]
end

# tensor_product produces a sorted, non-dual GradedOneTo
tensor_product(g::GradedOneTo) = sectormergesort(flip_dual(g))

function tensor_product(g1::GradedOneTo, g2::GradedOneTo)
    return sectormergesort(unmerged_tensor_product(g1, g2))
end

# ========================  mixed-type tensor_product  ========================
# Convert to a common type via `to_gradedrange` and dispatch to
# tensor_product(::GradedOneTo, ::GradedOneTo).

# SectorOneTo ↔ GradedOneTo
function tensor_product(s::SectorOneTo, g::GradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedOneTo, s::SectorOneTo)
    return tensor_product(g, to_gradedrange(s))
end

# SectorRange ↔ GradedOneTo
function tensor_product(s::SectorRange, g::GradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedOneTo, s::SectorRange)
    return tensor_product(g, to_gradedrange(s))
end

# SectorRange ↔ SectorOneTo
function tensor_product(s::SectorRange, r::SectorOneTo)
    return tensor_product(to_gradedrange(s), to_gradedrange(r))
end
function tensor_product(r::SectorOneTo, s::SectorRange)
    return tensor_product(to_gradedrange(r), to_gradedrange(s))
end

# TKS.Sector ↔ GradedOneTo
function tensor_product(s::TKS.Sector, g::GradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::GradedOneTo, s::TKS.Sector)
    return tensor_product(g, to_gradedrange(s))
end

# TKS.Sector ↔ SectorOneTo
function tensor_product(s::TKS.Sector, r::SectorOneTo)
    return tensor_product(to_gradedrange(s), to_gradedrange(r))
end
function tensor_product(r::SectorOneTo, s::TKS.Sector)
    return tensor_product(to_gradedrange(r), to_gradedrange(s))
end

# ========================  bipermutedimsopadd!  ========================
# Primary overloads. The flat-perm permutedimsopadd! overloads forward here.

function TensorAlgebra.bipermutedimsopadd!(
        y::AbstractSectorArray, op, x::AbstractSectorArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    perm = (perm_codomain..., perm_domain...)
    phase = fermion_permutation_phase(sector(x), perm)
    TensorAlgebra.bipermutedimsopadd!(
        data(y), op, data(x), perm_codomain, perm_domain, phase * α, β
    )
    return y
end

function TensorAlgebra.bipermutedimsopadd!(
        y::AbstractGradedArray{<:Any, N}, op, x::AbstractGradedArray{<:Any, N},
        perm_codomain, perm_domain,
        α::Number, β::Number
    ) where {N}
    perm = (perm_codomain..., perm_domain...)
    # `scale!(y, 0)` doesn't reliably zero `y`: if any block of `y` holds
    # `NaN`/`Inf` (uninitialized memory from `undef` allocation or a stale
    # garbage value), `NaN * 0 == NaN` keeps it poisoned, and subsequent
    # `bipermutedimsopadd!(..., α, one(β))` calls on a block of `y` that
    # doesn't get visited by the loop below would leak that garbage into the
    # result. Allocating broadcasts like `3 * a` go through this path (they
    # call with β == 0 on a fresh `similar`-allocated array); before this
    # fix they occasionally produced `NaN`s in unstored-block slots. Call
    # `zero!` explicitly for β == 0 to avoid the NaN-propagation trap.
    iszero(β) ? FI.zero!(y) : scale!(y, β)
    for bI in eachblockstoredindex(x)
        b = Tuple(bI)
        b_dest = Block(ntuple(i -> b[perm[i]], N))
        y_b = view(y, Tuple(b_dest)...)
        x_b = x[bI]
        TensorAlgebra.bipermutedimsopadd!(
            y_b, op, x_b, perm_codomain, perm_domain, α, one(β)
        )
    end
    return y
end
