# ========================  trivial_gradedrange  ========================

function trivial_gradedrange(t::Tuple{Vararg{AbstractGradedOneTo}})
    return tensor_product(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: Sector}
    return fusedgradedrange([trivial(S) => 1])
end

# ========================  fuseaxes  ========================

# Several legs go through TensorKit's `fuse`, which is much faster than a pairwise
# `tensor_product` that builds and merges an unmerged axis at every step.
fuseaxes(::Type{S}, axs::Tuple{}) where {S <: Sector} = trivial_gradedrange(S)
fuseaxes(::Type{<:Sector}, axs::Tuple{Any}) = tensor_product(only(axs))
function fuseaxes(::Type{<:Sector}, axs::Tuple{Any, Any, Vararg{Any}})
    return FusedGradedOneTo(TK.fuse(reduce(TK.:⊗, map(ElementarySpace ∘ fusesectors, axs))))
end

# ========================  unmerged_matricize_axes  ========================

# The row and column axes of the matricization before same-sector blocks are merged. The domain
# group is `flip`ped so the matrix reads as a `codomain ← domain` map.
function unmerged_matricize_axes(
        S::Type{<:Sector},
        axes_codomain::Tuple{Vararg{AbstractGradedOneTo}},
        axes_domain::Tuple{Vararg{AbstractGradedOneTo}}
    )
    # An empty group fuses to the trivial sector, which `S` supplies.
    init = trivial_gradedrange(S)
    ax_codomain = reduce(unmerged_tensor_product, axes_codomain; init)
    ax_domain = flip(reduce(unmerged_tensor_product, axes_domain; init))
    return ax_codomain, ax_domain
end

# ========================  UniqueSectorDelta matricize  ========================

function TensorAlgebra.matricizeconjcopy(
        isconj::Bool, a::UniqueSectorDelta, perm_codomain, perm_domain
    )
    ax_codomain = map(i -> isconj ? conj(axes(a, i)) : axes(a, i), perm_codomain)
    ax_codomain =
        isempty(ax_codomain) ? trivial(sectortype(a)) : tensor_product(ax_codomain...)
    # A one-leg group fuses to its own `flip_dual`, which still carries an arrow.
    return SectorIdentity{eltype(a)}(sector(ax_codomain))
end

# ========================  UniqueSectorArray matricize  ========================

function TensorAlgebra.isoutputview(
        ::typeof(TensorAlgebra.matricizeconj), isconj::Bool, ::UniqueSectorArray,
        perm_codomain, perm_domain
    )
    return !isconj && TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain)
end
function TensorAlgebra.matricizeconjview(
        isconj::Bool, a::UniqueSectorArray, perm_codomain, perm_domain
    )
    ndims_codomain = Val(length(perm_codomain))
    asectors_reshaped = matricize(structure(a), ndims_codomain)
    adata_reshaped = matricize(data(a), ndims_codomain)
    return sector_kron(asectors_reshaped, adata_reshaped)
end
function TensorAlgebra.matricizeconjcopy(
        isconj::Bool, a::UniqueSectorArray, perm_codomain, perm_domain
    )
    a_perm = TensorAlgebra.permutedimsconj(isconj, a, perm_codomain, perm_domain)
    return matricize(a_perm, Val(length(perm_codomain)))
end

# ========================  sector array unmatricize  ========================

# `unmatricize` receives the domain axes un-dualized.
function TensorAlgebra.unmatricize(
        m::AbstractSectorDelta{<:Any, <:Any, 2},
        codomain_axes::Tuple{Vararg{OrientedSector}},
        domain_axes::Tuple{Vararg{OrientedSector}}
    )
    return UniqueSectorDelta{eltype(m)}((codomain_axes..., conj.(domain_axes)...))
end

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

# A lazy adjoint has no contiguous buffer to reshape.
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
