using StridedViews: StridedViews, StridedView, isstrided

# ========================  bipartite-axes interface  ========================
# A type defines `axes_codomain`/`axes_domain`, returning its codomain and domain axes un-dualized,
# and these follow.

biaxes(a::AbstractArray) = bispace(axes_codomain(a), axes_domain(a))
axis_codomain(a::AbstractArray) = only(axes_codomain(a))
axis_domain(a::AbstractArray) = only(axes_domain(a))
axis(a::AbstractArray) = only(axes(a))

# Scaling keeps the codomain/domain split of the operand, while a sum or a `PermutedDims` puts all
# axes in the codomain.
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

# A dual axis is sorted by its non-dual sectors.
function sectorsortperm(g::AbstractGradedOneTo)
    return Block.(sortperm(sectors(g)))
end

# Map each block of `fine_ax` to its `Block[subrange]` in `merged_ax`, given the block permutation
# `blockperm` that sorts `fine_ax` before merging.
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

function fusesectors(g::AbstractGradedOneTo)
    return sortmergesectors(sectors(g), datalengths(g), isdual(g))
end

# Always non-dual. Flipping commutes with merging, so the merged form is flipped.
function tensor_product(g::AbstractGradedOneTo)
    f = fusesectors(g)
    return isdual(f) ? flip(f) : f
end

# `reduce` over one element would otherwise skip the one-argument `tensor_product`, which
# normalizes the arrow.
function Base.reduce_first(
        ::typeof(tensor_product),
        x::Union{AbstractOrientedSector, SectorOneTo, AbstractGradedOneTo}
    )
    return tensor_product(x)
end

function tensor_product(g1::AbstractGradedOneTo, g2::AbstractGradedOneTo)
    return fusesectors(unmerged_tensor_product(g1, g2))
end

# ========================  mixed-type tensor_product  ========================

# SectorOneTo ↔ GradedOneTo
function tensor_product(s::SectorOneTo, g::AbstractGradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::AbstractGradedOneTo, s::SectorOneTo)
    return tensor_product(g, to_gradedrange(s))
end

# Sector ↔ GradedOneTo
function tensor_product(s::Sector, g::AbstractGradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::AbstractGradedOneTo, s::Sector)
    return tensor_product(g, to_gradedrange(s))
end

# OrientedSector ↔ GradedOneTo. Reached mid-`reduce` over oriented sectors once a non-abelian
# fusion step has turned the accumulator into a graded range.
function tensor_product(s::OrientedSector, g::AbstractGradedOneTo)
    return tensor_product(to_gradedrange(s), g)
end
function tensor_product(g::AbstractGradedOneTo, s::OrientedSector)
    return tensor_product(g, to_gradedrange(s))
end

# OrientedSector ↔ SectorOneTo
function tensor_product(s::OrientedSector, r::SectorOneTo)
    return tensor_product(to_gradedrange(s), to_gradedrange(r))
end
function tensor_product(r::SectorOneTo, s::OrientedSector)
    return tensor_product(to_gradedrange(r), to_gradedrange(s))
end

# Sector ↔ SectorOneTo
function tensor_product(s::Sector, r::SectorOneTo)
    return tensor_product(to_gradedrange(s), to_gradedrange(r))
end
function tensor_product(r::SectorOneTo, s::Sector)
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

# ========================  bipermutedimsconjadd!  ========================

function TensorAlgebra.bipermutedimsconjadd!(
        y::AbstractSectorArray, isconj::Bool, x::AbstractSectorArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    check_input(bipermutedimsconjadd!, y, isconj, x, perm_codomain, perm_domain)
    perm = (perm_codomain..., perm_domain...)
    sx = structure(x)
    # The fermion sign goes on the reduced data. `fermion_permutation_phase` includes the leg
    # reversal under `isconj`, and the bend phases unbend the domain legs of the source and rebend
    # those of the destination.
    ndims_domain_src = ndims_domain(sx)
    ndims_domain_dest = ndims_domain(structure(y))
    src_domain_legs = ntuple(i -> ndims_codomain(sx) + i, ndims_domain_src)
    dest_domain_legs =
        ntuple(i -> perm[ndims(x) - ndims_domain_dest + i], ndims_domain_dest)
    phase =
        fermion_permutation_phase(isconj, sx, invperm(perm)) *
        fermion_bend_phase(sx, src_domain_legs) *
        fermion_bend_phase(sx, dest_domain_legs)
    bipermutedimsconjadd!(
        data(y), isconj, data(x), perm_codomain, perm_domain, phase * α, β
    )
    return y
end

# For abelian sectors the data is the dense array.
StridedViews.StridedView(a::UniqueSectorArray) = StridedViews.StridedView(data(a))

# Under unique fusion the reduced data is the dense array.
function TensorAlgebra.bipermutedimsconjadd!(
        y::AbstractArray, isconj::Bool, x::AbstractSectorArray,
        perm_codomain, perm_domain,
        α::Number, β::Number
    )
    require_unique_fusion(x)
    bipermutedimsconjadd!(y, isconj, data(x), perm_codomain, perm_domain, α, β)
    return y
end

function TensorAlgebra.bipermutedimsconjadd!(
        y::AbstractFusedGradedArray{<:Any, <:Any, N}, isconj::Bool,
        x::AbstractFusedGradedArray{<:Any, <:Any, N},
        perm_codomain, perm_domain,
        α::Number, β::Number
    ) where {N}
    check_input(bipermutedimsconjadd!, y, isconj, x, perm_codomain, perm_domain)
    if Base.mightalias(y, x)
        # With the identity permutation this is a scale, which keeps `a .*= 2` working. Any other
        # aliased permute-add would overwrite `y` before reading `x`.
        if y === x && !isconj &&
                (perm_codomain..., perm_domain...) == ntuple(identity, Val(N))
            return scale!(y, α + β)
        end
        throw(ArgumentError("output array must not be aliased with the input array"))
    end
    # `scale!(y, 0)` leaves `NaN` from an `undef` allocation, since `NaN * 0 == NaN`.
    iszero(β) ? zero!(y) : scale!(y, β)
    for bI in eachblockstoredindex(x)
        b = Tuple(bI)
        b_dest = Block(ntuple(i -> b[(perm_codomain..., perm_domain...)[i]], N))
        y_b = view(y, Tuple(b_dest)...)
        x_b = x[bI]
        bipermutedimsconjadd!(y_b, isconj, x_b, perm_codomain, perm_domain, α, one(β))
    end
    return y
end

# ========================  contraction twist  ========================
# The supertrace convention for fermionic contractions: `⟨i| ⋅ |j⟩ = δᵢⱼ` and
# `|i⟩ ⋅ ⟨j| = θᵢⱼ δᵢⱼ`, where `θᵢⱼ = ±1` is the self-crossing phase, `-1` only for an odd
# fermion charge. The twist goes on one leg of each contracted pair: the dual legs of the right
# factor's codomain, or equivalently the non-dual legs of the left factor's domain, since the two
# legs of a pair have equal twists.
#
# The legs are a `Tuple` rather than a generator since `twist!` reads them once per fusion tree.
function twist_codomain!(a::AbstractArray, ndims_codomain::Int)
    return TA.twist!(a, Tuple(i for i in 1:ndims_codomain if isdual(axes(a, i))))
end
function twist_domain!(a::AbstractArray, ndims_codomain::Int)
    return TA.twist!(
        a,
        Tuple(i for i in (ndims_codomain + 1):ndims(a) if !isdual(axes(a, i)))
    )
end

function needs_contraction_twist(isconj::Bool, a, perm_codomain)
    return TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic &&
        any(i -> isdual(axes(a, i)) ⊻ isconj, perm_codomain)
end

# `matricizeconj` with the fermionic contraction twist applied to the contracted legs of the right
# factor (`side = :codomain`) or the left one (`side = :domain`), on a copy so `a` is untouched.
function twisted_matricizeconj(
        isconj::Bool, a::AbstractArray, perm_codomain, perm_domain, side::Symbol
    )
    side in (:codomain, :domain) ||
        throw(ArgumentError("`side` must be `:codomain` or `:domain`, got `$(repr(side))`"))
    a_twisted =
    if TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain) &&
            TensorAlgebra.is_output_view(
            TensorAlgebra.matricizeconj, isconj, a, perm_codomain, perm_domain
        )
        copy(a)
    else
        TensorAlgebra.permutedimsconj(isconj, a, perm_codomain, perm_domain)
    end
    n = length(perm_codomain)
    side === :codomain ? twist_codomain!(a_twisted, n) : twist_domain!(a_twisted, n)
    return matricize(a_twisted)
end
