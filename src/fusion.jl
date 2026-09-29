# ========================  trivial_gradedrange  ========================

function trivial_gradedrange(t::Tuple{Vararg{AbstractGradedOneTo}})
    return tensor_product(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: Sector}
    return fusedgradedrange([trivial(S) => 1])
end

# ========================  fuseaxes  ========================

# Fuse a group of leg axes into its coupled fused-sorted axis. A single leg goes straight to
# `tensor_product` (the axis's cached fused form for a non-dual axis, its `flip` for a dual one);
# a multi-leg group reduces pairwise.
fuseaxes(::Type{S}, axs::Tuple{}) where {S <: Sector} = trivial_gradedrange(S)
fuseaxes(::Type{<:Sector}, axs::Tuple{Any}) = tensor_product(only(axs))
fuseaxes(::Type{<:Sector}, axs::Tuple) = reduce(tensor_product, axs)

# ========================  unmerged_matricize_axes  ========================

# Fuse a bipartitioned tuple of graded axes into the unmerged 2D row/column axes: one
# block per source-block combination, before `fusesectors` merges same-sector blocks
# into the final matricized axes. The codomain group fuses as-is; the domain group is
# `flip`ed (same sectors and sizes, opposite arrow) so the matrix reads as a
# `codomain ← domain` map and the matmul pairs contracted legs correctly.
function unmerged_matricize_axes(
        S::Type{<:Sector},
        axes_codomain::Tuple{Vararg{AbstractGradedOneTo}},
        axes_domain::Tuple{Vararg{AbstractGradedOneTo}}
    )
    # The trivial-sector init seeds each `reduce`, so a group with no axes (a rank-0
    # codomain or domain, as in a full contraction to a scalar) fuses to the trivial
    # sector. `S` supplies that sector when no axis is present to carry it.
    init = trivial_gradedrange(S)
    ax_codomain = reduce(unmerged_tensor_product, axes_codomain; init)
    ax_domain = flip(reduce(unmerged_tensor_product, axes_domain; init))
    return ax_codomain, ax_domain
end

# ========================  UniqueSectorDelta matricize  ========================

# A delta is structural (no data storage), so nothing is shared and the rebuilt identity is
# the copy leaf. `op` applies to the axes, dualizing them for `conj`, the same convention
# `allocate_output(permutedimsop, ...)` follows.
function TensorAlgebra.matricizeopcopy(
        op, a::UniqueSectorDelta, perm_codomain, perm_domain
    )
    ax_codomain = map(i -> op(axes(a, i)), perm_codomain)
    ax_codomain =
        isempty(ax_codomain) ? trivial(sectortype(a)) : tensor_product(ax_codomain...)
    # A one-leg group fuses to its own `flip_dual`, which still carries an arrow, so take the
    # bare sector the identity factor stores.
    return SectorIdentity{Base.promote_op(op, eltype(a))}(sector(ax_codomain))
end

# ========================  UniqueSectorArray matricize  ========================

# The reduced data matricizes to a reshaped view, so the matricization shares `a`'s memory at
# every split that leaves the legs in order; the structural factor stores no data and is rebuilt
# as a `SectorIdentity`.
function TensorAlgebra.is_output_view(
        ::typeof(TensorAlgebra.matricizeop), op, ::UniqueSectorArray, perm_codomain, perm_domain
    )
    return op === identity && TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain)
end
function TensorAlgebra.matricizeopview(
        op, a::UniqueSectorArray, perm_codomain, perm_domain
    )
    ndims_codomain = Val(length(perm_codomain))
    asectors_reshaped = matricize(structure(a), ndims_codomain)
    adata_reshaped = matricize(data(a), ndims_codomain)
    return sector_kron(asectors_reshaped, adata_reshaped)
end
# Permute into fresh storage, then read off that copy's view. At the identity bipermutation
# `permutedimsop` is itself the copy, so this costs one pass either way.
function TensorAlgebra.matricizeopcopy(
        op, a::UniqueSectorArray, perm_codomain, perm_domain
    )
    a_perm = TensorAlgebra.permutedimsop(op, a, perm_codomain, perm_domain)
    return matricize(a_perm, Val(length(perm_codomain)))
end

# ========================  sector array unmatricize  ========================

# `unmatricize` receives the domain axes codomain-facing (un-dualized); a graded array stores
# them dualized, so `conj` re-dualizes them before they are placed.
function TensorAlgebra.unmatricize(
        m::AbstractSectorDelta{<:Any, <:Any, 2},
        codomain_axes::Tuple{Vararg{OrientedSector}},
        domain_axes::Tuple{Vararg{OrientedSector}}
    )
    return UniqueSectorDelta{eltype(m)}((codomain_axes..., conj.(domain_axes)...))
end

# Unmatricize a 2D sector array back to an N-D UniqueSectorArray. The
# codomain/domain axes must be SectorOneTo (carrying multiplicity info).
# Works for both UniqueSectorMatrix and FusedSectorMatrix.
function TensorAlgebra.unmatricize(
        m::AbstractSectorArray{<:Any, <:Any, 2},
        codomain_axes::Tuple{Vararg{SectorOneTo}},
        domain_axes::Tuple{Vararg{SectorOneTo}}
    )
    msectors = unmatricize(
        structure(m),
        structure.(codomain_axes),
        structure.(domain_axes)
    )
    mdata = unmatricize(
        data(m),
        data.(codomain_axes),
        data.(domain_axes)
    )
    return UniqueSectorArray(mdata, msectors)
end

# ========================  adjoint fused graded unmatricize  ========================

# A lazy adjoint has no owned contiguous buffer to reshape; materialize it into a `FusedGradedMatrix`
# first, then unmatricize that.
function TensorAlgebra.unmatricize(
        m::AdjointFusedGradedMatrix,
        codomain_axes::Tuple{Vararg{AbstractGradedOneTo}},
        domain_axes::Tuple{Vararg{AbstractGradedOneTo}}
    )
    return TensorAlgebra.unmatricize(copy(m), codomain_axes, domain_axes)
end

# ========================  Allowed block keys  ========================

function allowedblocks(axs::NTuple{N, AbstractGradedOneTo}) where {N}
    N == 0 && return Block{0, Int}[Block()]
    @assert TKS.FusionStyle(sectortype(eltype(axs))) === TKS.UniqueFusion()
    unfused = reduce(axs; init = trivial_gradedrange(axs)) do ax1, ax2
        return unmerged_tensor_product(ax1, ax2)
    end
    cart = CartesianIndices(Tuple(blocklength.(axs)))
    return Block.(Tuple.(cart[findall(istrivial, sectors(unfused))]))
end
