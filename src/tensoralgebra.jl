using SplitApplyCombine: groupcount
using StridedViews: StridedViews, StridedView, isstrided

# ========================  bipartite-axes interface  ========================
# The shared codomain/domain axis interface for `GradedArray` and the fused graded arrays (a candidate
# to move to TensorAlgebra alongside `BiTuple`/`bispace`). A type implements the two primitives
# `axes_codomain`/`axes_domain` — its codomain and domain axis groups in un-dualized (codomain-facing)
# form — and these derived helpers follow. `biaxes` wraps the halves into the `bispace`/`BiTuple` form
# (dualizing the domain), so an implementer never constructs a `BiTuple`; `axis_codomain`/`axis_domain`
# are the single-axis form for a matrix-like array (one axis per side), and `axis` the single axis of a
# one-dimensional array. Overloaded on `AbstractArray` because `GradedArray` shares the interface but is
# not an `AbstractFusedGradedArray`.

biaxes(a::AbstractArray) = bispace(axes_codomain(a), axes_domain(a))
axis_codomain(a::AbstractArray) = only(axes_codomain(a))
axis_domain(a::AbstractArray) = only(axes_domain(a))
axis(a::AbstractArray) = only(axes(a))

# Expression nodes share the same interface: `axes_codomain`/`axes_domain` are the primitives and
# `biaxes` derives from them (as for arrays above). Scaling keeps the operand's split, conjugation
# dualizes both halves, and a sum (always ≥2 addends) or a `PermutedDims` view goes all-codomain.
biaxes(a::TA.LinearBroadcasted) = bispace(axes_codomain(a), axes_domain(a))
biaxes(a::TA.PermutedDims) = bispace(axes_codomain(a), axes_domain(a))
axes_codomain(a::TA.ScaledBroadcasted) = axes_codomain(TA.unscaled(a))
axes_domain(a::TA.ScaledBroadcasted) = axes_domain(TA.unscaled(a))
axes_codomain(a::TA.ConjBroadcasted) = map(conj, axes_codomain(parent(a)))
axes_domain(a::TA.ConjBroadcasted) = map(conj, axes_domain(parent(a)))
axes_codomain(a::TA.AddBroadcasted) = axes(a)
axes_domain(a::TA.AddBroadcasted) = ()
axes_codomain(a::TA.PermutedDims) = axes(a)
axes_domain(a::TA.PermutedDims) = ()

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

function unmerged_tensor_product(a1::AbstractGradedOneTo, a2::AbstractGradedOneTo)
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
function sectorsortperm(g::AbstractGradedOneTo)
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
function sectormergesortperm(g::AbstractGradedOneTo)
    return Block.(groupsortperm(sectors(g)))
end

# Used by `TensorAlgebra.unmatricize` in `GradedArraysTensorAlgebraExt`.
invblockperm(a::Vector{<:Block{1}}) = Block.(invperm(Int.(a)))

# Returns a Vector{BlockIndexRange{1}} mapping each block of fine_ax (in original order)
# to its position (block + subrange) within the merged axis merged_ax, given the block
# permutation blockperm used to sort and merge fine_ax into merged_ax.
# Requires that blocks of fine_ax subdivide blocks of merged_ax.
function invblockmergeperm(
        fine_ax::AbstractGradedOneTo,
        blockperm,
        merged_ax::AbstractGradedOneTo
    )
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

# The result is fused-sorted (each sector once, in order) by construction, so return the type that
# encodes that invariant rather than a plain `GradedOneTo`.
function sectormergesort(g::AbstractGradedOneTo)
    # Merge repeated sectors (summing their data lengths) and sort. The stored sectors are non-dual
    # and the arrow is axis-level, so merge and sort them directly and carry `isdual` through.
    dict = Dict{sectortype(g), Int}()
    for (s, m) in zip(sectors(g), datalengths(g))
        dict[s] = get(dict, s, 0) + m
    end
    merged = sort!(collect(pairs(dict)); by = first)
    return FusedGradedOneTo(first.(merged), last.(merged), isdual(g))
end

# tensor_product produces a fused-sorted, non-dual FusedGradedOneTo
tensor_product(g::AbstractGradedOneTo) = sectormergesort(flip_dual(g))

function tensor_product(g1::AbstractGradedOneTo, g2::AbstractGradedOneTo)
    return sectormergesort(unmerged_tensor_product(g1, g2))
end

# ========================  mixed-type tensor_product  ========================
# Convert to a common type via `to_gradedrange` and dispatch to
# tensor_product(::GradedOneTo, ::GradedOneTo).

# SectorOneTo ↔ GradedOneTo
function tensor_product(s::SectorOneTo, g::AbstractGradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::AbstractGradedOneTo, s::SectorOneTo)
    return tensor_product(g, to_gradedrange(s))
end

# SectorRange ↔ GradedOneTo
function tensor_product(s::SectorRange, g::AbstractGradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::AbstractGradedOneTo, s::SectorRange)
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
function tensor_product(s::TKS.Sector, g::AbstractGradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::AbstractGradedOneTo, s::TKS.Sector)
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
    check_input(bipermutedimsopadd!, y, op, x, perm_codomain, perm_domain)
    perm = (perm_codomain..., perm_domain...)
    sx = sector(x)
    # Fermion signs go on the reduced data (the delta is `one(T)`). `fermion_permutation_phase`
    # (op-aware) gives the braiding sign, plus the ket->bra leg reversal for `op === conj`. The two
    # `fermion_bend_phase` factors reconcile the splits (unbend the source's domain legs, rebend the
    # destination's) and are `1` for all-codomain blocks.
    ndims_domain_src = ndims_domain(sx)
    ndims_domain_dest = ndims_domain(sector(y))
    src_domain_legs = ntuple(i -> ndims_codomain(sx) + i, ndims_domain_src)
    dest_domain_legs =
        ntuple(i -> perm[ndims(x) - ndims_domain_dest + i], ndims_domain_dest)
    phase =
        fermion_permutation_phase(op, sx, invperm(perm)) *
        fermion_bend_phase(sx, src_domain_legs) *
        fermion_bend_phase(sx, dest_domain_legs)
    bipermutedimsopadd!(
        data(y), op, data(x), perm_codomain, perm_domain, phase * α, β
    )
    return y
end

# An abelian sector array's dense form is its data block (each abelian sector is
# one-dimensional, so the structural sector-delta factor is an identity selection).
# Wrapping it as a `StridedView` of its data lets it flow through the generic strided
# permute-add path when the other operand is a plain dense array.
StridedViews.StridedView(a::UniqueSectorArray) = StridedViews.StridedView(data(a))

# Permute-add a sector source into a plain (non-sector) destination. Under unique fusion the reduced
# data is the full dense array, so forward to the dense primitive on `data(x)` rather than letting the
# generic fall back to scalar reads of the sector source. The `y::AbstractSectorArray` method above is
# strictly more specific, so a sector→sector call still takes the block-wise path. This is what lets
# `add!(dest::AbstractArray, ::AbstractSectorArray, α, β)` (via the generic `permutedimsopadd!`)
# work without a bespoke `add!` overload.
function TensorAlgebra.bipermutedimsopadd!(
        y::AbstractArray, op, x::AbstractSectorArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    require_unique_fusion(x)
    bipermutedimsopadd!(y, op, data(x), perm_codomain, perm_domain, α, β)
    return y
end

function TensorAlgebra.bipermutedimsopadd!(
        y::AbstractFusedGradedArray{<:Any, <:Any, N}, op,
        x::AbstractFusedGradedArray{<:Any, <:Any, N},
        perm_codomain, perm_domain,
        α::Number, β::Number
    ) where {N}
    check_input(bipermutedimsopadd!, y, op, x, perm_codomain, perm_domain)
    if Base.mightalias(y, x)
        # A self-aliased permute-add with the identity permutation and no conjugation is really a
        # scale, `y = α*y + β*y = (α+β)*y`, so route it to the block-wise `scale!` (which handles
        # aliasing). This keeps `a .*= 2` working. Any other aliased permute-add can't run in
        # place: the `zero!`/`scale!` step below overwrites `y` before the block loop reads `x`, so
        # refuse it rather than silently corrupt the result, matching `TensorOperations`.
        if y === x && op === identity &&
                (perm_codomain..., perm_domain...) == ntuple(identity, Val(N))
            return scale!(y, α + β)
        end
        throw(ArgumentError("output array must not be aliased with the input array"))
    end
    # `scale!(y, 0)` doesn't reliably zero `y`: if any block of `y` holds
    # `NaN`/`Inf` (uninitialized memory from `undef` allocation or a stale
    # garbage value), `NaN * 0 == NaN` keeps it poisoned, and subsequent
    # `bipermutedimsopadd!(..., α, one(β))` calls on a block of `y` that
    # doesn't get visited by the loop below would leak that garbage into the
    # result. Allocating broadcasts like `3 * a` go through this path (they
    # call with β == 0 on a fresh `similar`-allocated array); before this
    # fix they occasionally produced `NaN`s in unstored-block slots. Call
    # `zero!` explicitly for β == 0 to avoid the NaN-propagation trap.
    iszero(β) ? zero!(y) : scale!(y, β)
    for bI in eachblockstoredindex(x)
        b = Tuple(bI)
        b_dest = Block(ntuple(i -> b[(perm_codomain..., perm_domain...)[i]], N))
        y_b = view(y, Tuple(b_dest)...)
        x_b = x[bI]
        bipermutedimsopadd!(y_b, op, x_b, perm_codomain, perm_domain, α, one(β))
    end
    return y
end

# ========================  fermionic contraction twist  ========================
# Fermionic contractions need the second (right) factor's contracted legs twisted before
# matricization, so the result does not depend on contraction order. This rides on
# TensorAlgebra v0.10's per-position fusion styles: `default_contract_algorithm` puts
# `TwistedGradedMatricize` on the right factor only, and its `matricizeopperm` inserts the twist
# between the permute and the matricize. The twist is a no-op for bosonic sectors.

"""
    contraction_twist!(a::UniqueSectorArray, ndims_codomain::Int) -> a

Apply the twist convention for the supertrace formalism of fermionic contractions.
This means that ``⟨i| ⋅ |j⟩ = δᵢⱼ``, and ``|i⟩ ⋅ ⟨j| = θᵢⱼ δᵢⱼ``.
Here, ``θᵢⱼ = ±1`` is defined as the phase from applying a self-crossing,
which is always ``1`` for bosonic symmetries, but can be ``-1`` for odd fermion charges.

Equivalent to `twist!(a, (i for i in 1:ndims_codomain if isdual(axes(a, i))))`.
A no-op unless `BraidingStyle(sectortype(a))` is `Fermionic`.

See also `twist!`.
"""
function contraction_twist!(a::AbstractArray, ndims_codomain::Int)
    return twist!(a, (i for i in 1:ndims_codomain if isdual(axes(a, i))))
end

function TensorAlgebra.matricizeopperm(
        ::TwistedGradedMatricize, op, a::AbstractArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    a_perm = TensorAlgebra.permutedimsop(op, a, perm_codomain, perm_domain)
    contraction_twist!(a_perm, length(perm_codomain))
    return matricize(GradedMatricize(), a_perm, Val(length(perm_codomain)))
end
