using TensorAlgebra: tensor_product_axis, trivial_axis

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:AbstractSectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbstractSectorArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbstractGradedArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorOneTo}) = SectorFusion()

const BlockReshapeFusion = typeof(FusionStyle(AbstractBlockArray))

# ========================  trivial_axis  ========================

function trivial_gradedrange(t::Tuple{Vararg{GradedOneTo}})
    return tensor_product(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: SectorRange}
    return gradedrange([trivial(S) => 1])
end

function TensorAlgebra.trivial_axis(
        ::BlockReshapeFusion,
        ::Val{:codomain},
        a::AbstractGradedArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_gradedrange((axes_codomain..., axes_domain...))
end
function TensorAlgebra.trivial_axis(
        ::BlockReshapeFusion,
        ::Val{:domain},
        a::AbstractGradedArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return flip(trivial_gradedrange((axes_codomain..., axes_domain...)))
end

# ========================  tensor_product_axis  ========================

# SectorOneTo level: fuse two block axes
function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:codomain}, r1::SectorOneTo, r2::SectorOneTo
    )
    return tensor_product(r1, r2)
end
function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:domain}, r1::SectorOneTo, r2::SectorOneTo
    )
    return flip(tensor_product(r1, r2))
end

# GradedOneTo level: iterate block axes, fuse each pair, reassemble
function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:codomain},
        r1::GradedOneTo, r2::GradedOneTo
    )
    return tensor_product_gradedrange(style, side, r1, r2)
end
function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:domain},
        r1::GradedOneTo, r2::GradedOneTo
    )
    return tensor_product_gradedrange(style, side, r1, r2)
end
function tensor_product_gradedrange(
        ::BlockReshapeFusion, side::Val,
        r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    blockaxpairs = Iterators.product(eachblockaxes1(r1), eachblockaxes1(r2))
    blockaxs = map(blockaxpairs) do (b1, b2)
        return tensor_product_axis(side, b1, b2)
    end
    return mortar_axis(vec(blockaxs))
end

# ========================  AbelianSectorDelta matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianSectorDelta, ndims_codomain::Val{Ncodomain}
    ) where {Ncodomain}
    biperm = trivialbiperm(ndims_codomain, Val(ndims(a)))
    ax_codomain, ax_domain = blocks(axes(a)[biperm])
    ax_codomain =
        isempty(ax_codomain) ? trivial(sector_type(a)) : tensor_product(ax_codomain...)
    return SectorIdentity{eltype(a)}(ax_codomain)
end

# ========================  AbelianSectorArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianSectorArray, ndims_codomain::Val{K}
    ) where {K}
    asectors_reshaped = matricize(sector(a), Val(K))

    T = TKS.sectorscalartype(sector_type(a))
    phase = prod(
        ntuple(K) do i
            return ifelse(isdual(a, i), twist(sectoraxes(a, i)), one(T))
        end
    )

    adata_reshaped = matricize(data(a), Val(K))
    isone(phase) || (adata_reshaped = phase .* adata_reshaped)

    return asectors_reshaped ⊗ adata_reshaped
end

# ========================  BlockReshapeFusion AbelianGradedArray matricize  ========================

function TensorAlgebra.matricize(
        style::BlockReshapeFusion, a::AbelianGradedArray{T, N}, ndims_codomain::Val{K}
    ) where {T, N, K}
    ax_2d = matricize_axes(style, a, ndims_codomain)
    a_2d = FI.zero!(similar(a, ax_2d))

    codomain_nblocks = Tuple(blocklength.(axes(a)[1:K]))
    domain_nblocks = Tuple(blocklength.(axes(a)[(K + 1):N]))
    cod_cart = CartesianIndices(codomain_nblocks)
    dom_cart = CartesianIndices(domain_nblocks)

    for bI_src in eachblockstoredindex(a)
        src = Tuple(bI_src)
        ci_cod = ntuple(i -> Int(src[i]), Val(K))
        ci_dom = ntuple(i -> Int(src[K + i]), Val(N - K))

        row_block = LinearIndices(cod_cart)[ci_cod...]
        col_block = LinearIndices(dom_cart)[ci_dom...]

        src_block = a[bI_src]
        dest_block = matricize(src_block, ndims_codomain)

        a_2d[Block(row_block, col_block)] = dest_block
    end

    return a_2d
end

# ========================  SectorFusion AbelianGradedArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianGradedArray, ndims_codomain::Val{K}
    ) where {K}
    a_reshaped = matricize(BlockReshapeFusion(), a, ndims_codomain)
    a_merged = sectormergesort(a_reshaped)
    return FusedGradedMatrix(a_merged)
end

# ========================  AbelianGradedArray unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::AbstractSectorDelta,
        codomain_axes::Tuple{Vararg{SectorRange}},
        domain_axes::Tuple{Vararg{SectorRange}}
    )
    return AbelianSectorDelta{eltype(m)}((codomain_axes..., domain_axes...))
end

# Unmatricize a 2D sector array back to an N-D AbelianSectorArray. Decomposes into
# sector (delta) and data (plain matrix) components, unmatricizes each
# independently, applies the fermionic contraction phase, and recombines.
# The codomain/domain axes must be SectorOneTo (carrying multiplicity info).
# Works for both AbelianSectorMatrix and SectorMatrix.
function TensorAlgebra.unmatricize(
        ::SectorFusion, m::AbstractSectorArray{<:Any, 2},
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

    phase = fermion_contraction_phase(msectors, length(codomain_axes))
    isone(phase) || (mdata = phase .* mdata)

    return AbelianSectorArray(msectors, mdata)
end

# ========================  BlockReshapeFusion AbelianGradedArray unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::BlockReshapeFusion, m::AbelianGradedMatrix{T},
        codomain_axes::Tuple{Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    K = length(codomain_axes)
    N = K + length(domain_axes)
    dest_axes = (codomain_axes..., domain_axes...)
    a = FI.zero!(similar(m, dest_axes))

    cod_cart = CartesianIndices(Tuple(map(blocklength, codomain_axes)))
    dom_cart = CartesianIndices(Tuple(map(blocklength, domain_axes)))

    for bI_src in eachblockstoredindex(m)
        row_block = Int(Tuple(bI_src)[1])
        col_block = Int(Tuple(bI_src)[2])
        dest_bk = (Tuple(cod_cart[row_block])..., Tuple(dom_cart[col_block])...)

        src_block = m[bI_src]
        dest_sects = ntuple(d -> sectors(dest_axes[d])[dest_bk[d]], Val(N))
        dest_dims = ntuple(d -> blocklengths(dest_axes[d])[dest_bk[d]], Val(N))
        dest_block = AbelianSectorArray(dest_sects, reshape(data(src_block), dest_dims))
        a[Block(dest_bk...)] = dest_block
    end

    return a
end

# ========================  SectorFusion FusedGradedMatrix unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::FusedGradedMatrix,
        codomain_axes::Tuple{Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    blocked_axes = (codomain_axes..., domain_axes...)
    if isempty(blocked_axes)
        error("Scalar unmatricize not yet supported for FusedGradedMatrix")
    end

    unfused_axes = matricize_axes(BlockReshapeFusion(), m, codomain_axes, domain_axes)
    m_abelian = AbelianGradedArray(m)
    blockperms = sectorsortperm.(unfused_axes)
    J = map(invblockmergeperm, unfused_axes, blockperms, axes(m_abelian))
    m_split = m_abelian[J...]
    return unmatricize(BlockReshapeFusion(), m_split, codomain_axes, domain_axes)
end

# ========================  Allowed block keys  ========================

function allowedblocks(axs::NTuple{N, GradedOneTo{S}}) where {N, S}
    N == 0 && return Block{0, Int}[Block()]
    @assert SymmetryStyle(S) === AbelianStyle()
    unfused = reduce(axs; init = trivial_gradedrange(axs)) do ax1, ax2
        return unmerged_tensor_product(ax1, ax2)
    end
    cart = CartesianIndices(Tuple(blocklength.(axs)))
    return Block.(Tuple.(cart[findall(istrivial, sectors(unfused))]))
end
