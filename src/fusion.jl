struct SectorMatricize <: MatricizeStyle end

# Matricize style for the right factor of a fermionic contraction: matricize as `SectorMatricize`
# after twisting the contracted legs (see `contraction_twist!`). A no-op twist for bosonic
# sectors, so it matricizes identically to `SectorMatricize` there.
struct TwistedSectorMatricize <: MatricizeStyle end

TensorAlgebra.MatricizeStyle(::Type{<:AbstractSectorDelta}) = SectorMatricize()
TensorAlgebra.MatricizeStyle(::Type{<:AbstractSectorArray}) = SectorMatricize()
TensorAlgebra.MatricizeStyle(::Type{<:AbstractFusedGradedArray}) = SectorMatricize()
TensorAlgebra.MatricizeStyle(::Type{<:SectorOneTo}) = SectorMatricize()

# ========================  trivial_gradedrange  ========================

function trivial_gradedrange(t::Tuple{Vararg{GradedOneTo}})
    return tensor_product(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: SectorRange}
    return fusedgradedrange([trivial(S) => 1])
end

# ========================  unmerged_matricize_axes  ========================

# Fuse a bipartitioned tuple of graded axes into the unmerged 2D row/column axes: one
# block per source-block combination, before `sectormergesort` merges same-sector blocks
# into the final matricized axes. The codomain group fuses as-is; the domain group is
# `flip`ed (same sectors and sizes, opposite arrow) so the matrix reads as a
# `codomain ← domain` map and the matmul pairs contracted legs correctly.
function unmerged_matricize_axes(
        S::Type{<:SectorRange},
        axes_codomain::Tuple{Vararg{GradedOneTo}}, axes_domain::Tuple{Vararg{GradedOneTo}}
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

function TensorAlgebra.matricize(
        ::SectorMatricize, a::UniqueSectorDelta, ndims_codomain::Val{Ncodomain}
    ) where {Ncodomain}
    ax_codomain = first(bipartition(axes(a), ndims_codomain))
    ax_codomain =
        isempty(ax_codomain) ? trivial(sectortype(a)) : tensor_product(ax_codomain...)
    return SectorIdentity{eltype(a)}(ax_codomain)
end

# ========================  UniqueSectorArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorMatricize, a::UniqueSectorArray, ndims_codomain::Val{K}
    ) where {K}
    asectors_reshaped = matricize(sector(a), Val(K))
    adata_reshaped = matricize(data(a), Val(K))
    return sector_kron(asectors_reshaped, adata_reshaped)
end

# ========================  SectorMatricize unmatricize  ========================

# `unmatricize` receives the domain axes codomain-facing (un-dualized); a graded array stores
# them dualized, so `conj` re-dualizes them before they are placed.
function TensorAlgebra.unmatricize(
        ::SectorMatricize, m::AbstractSectorDelta,
        codomain_axes::Tuple{Vararg{SectorRange}},
        domain_axes::Tuple{Vararg{SectorRange}}
    )
    return UniqueSectorDelta{eltype(m)}((codomain_axes..., conj.(domain_axes)...))
end

# Unmatricize a 2D sector array back to an N-D UniqueSectorArray. The
# codomain/domain axes must be SectorOneTo (carrying multiplicity info).
# Works for both UniqueSectorMatrix and FusedSectorMatrix.
function TensorAlgebra.unmatricize(
        ::SectorMatricize, m::AbstractSectorArray{<:Any, <:Any, 2},
        codomain_axes::Tuple{Vararg{SectorOneTo}},
        domain_axes::Tuple{Vararg{SectorOneTo}}
    )
    msectors = unmatricize(
        sector(m),
        sector.(codomain_axes),
        sector.(domain_axes)
    )
    mdata = unmatricize(
        data(m),
        data.(codomain_axes),
        data.(domain_axes)
    )
    return UniqueSectorArray(mdata, msectors)
end

# ========================  SectorMatricize FusedGradedMatrix unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::SectorMatricize, m::FusedGradedMatrix,
        codomain_axes::Tuple{Vararg{AbstractGradedOneTo}},
        domain_axes::Tuple{Vararg{AbstractGradedOneTo}}
    )
    K = length(codomain_axes)
    N = K + length(domain_axes)
    a = TA.similar_map(m, codomain_axes, domain_axes)
    return TensorAlgebra.unmatricizeperm!(
        SectorMatricize(), a, m, ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K))
    )
end

# A lazy adjoint has no owned contiguous buffer to reshape; materialize it into a `FusedGradedMatrix`
# first, then unmatricize that.
function TensorAlgebra.unmatricize(
        style::SectorMatricize, m::AdjointFusedGradedArray,
        codomain_axes::Tuple{Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    return TensorAlgebra.unmatricize(style, copy(m), codomain_axes, domain_axes)
end

# ========================  Allowed block keys  ========================

function allowedblocks(axs::NTuple{N, GradedOneTo}) where {N}
    N == 0 && return Block{0, Int}[Block()]
    @assert TKS.FusionStyle(sectortype(eltype(axs))) === TKS.UniqueFusion()
    unfused = reduce(axs; init = trivial_gradedrange(axs)) do ax1, ax2
        return unmerged_tensor_product(ax1, ax2)
    end
    cart = CartesianIndices(Tuple(blocklength.(axs)))
    return Block.(Tuple.(cart[findall(istrivial, sectors(unfused))]))
end
