using TensorAlgebra: tensor_product_axis, trivial_axis

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:SectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbelianArray}) = SectorFusion()

# ========================  trivial_axis  ========================

function trivial_gradedrange(t::Tuple{Vararg{GradedOneTo}})
    return ⊗(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: SectorRange}
    return gradedrange([trivial(S) => 1])
end

function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:codomain},
        a::AbelianArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_gradedrange(axes(a))
end
function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:domain},
        a::AbelianArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return flip(trivial_gradedrange(axes(a)))
end

function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:codomain},
        a::FusedSectorMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_gradedrange((axes_codomain..., axes_domain...))
end
function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:domain},
        a::FusedSectorMatrix,
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
    return r1 ⊗ r2
end
function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:domain}, r1::SectorOneTo, r2::SectorOneTo
    )
    return flip(r1 ⊗ r2)
end

# GradedOneTo level: iterate block axes, fuse each pair, reassemble
function TensorAlgebra.tensor_product_axis(
        style::SectorFusion, side::Val{:codomain},
        r1::GradedOneTo, r2::GradedOneTo
    )
    blockaxpairs = Iterators.product(eachblockaxis(r1), eachblockaxis(r2))
    blockaxs = map(blockaxpairs) do (b1, b2)
        return tensor_product_axis(style, side, b1, b2)
    end
    return mortar_axis(vec(blockaxs))
end
function TensorAlgebra.tensor_product_axis(
        style::SectorFusion, side::Val{:domain},
        r1::GradedOneTo, r2::GradedOneTo
    )
    blockaxpairs = Iterators.product(eachblockaxis(r1), eachblockaxis(r2))
    blockaxs = map(blockaxpairs) do (b1, b2)
        return tensor_product_axis(style, side, b1, b2)
    end
    return mortar_axis(vec(blockaxs))
end

# ========================  SectorDelta matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::SectorDelta, ndims_codomain::Val{Ncodomain}
    ) where {Ncodomain}
    biperm = trivialbiperm(ndims_codomain, Val(ndims(a)))
    ax_codomain, ax_domain = blocks(axes(a)[biperm])
    ax_codomain =
        isempty(ax_codomain) ? trivial(sector_type(a)) : tensor_product(ax_codomain...)
    ax_domain =
        isempty(ax_domain) ? trivial(sector_type(a)) : flip(tensor_product(ax_domain...))
    return SectorDelta{eltype(a)}((ax_codomain, ax_domain))
end

# ========================  SectorArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::SectorArray, ndims_codomain::Val{K}
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

    return SectorArray(asectors_reshaped, adata_reshaped)
end

# ========================  AbelianArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianArray, ndims_codomain::Val{K}
    ) where {K}
    a_reshaped = block_reshape(a, Val(K))
    a_merged = sectormergesort(a_reshaped)
    return FusedSectorMatrix(a_merged)
end

"""
    block_reshape(a::AbelianArray, ndims_codomain::Val{K}) -> AbelianArray{T,2}

Reshape an N-d AbelianArray into a 2D AbelianArray by grouping the first K
dimensions as codomain and the remaining as domain. Computes unfused 2D graded
axes via `tensor_product_axis`, then permutes and reshapes each block's data.
"""
function block_reshape(
        a::AbelianArray{T, N}, ndims_codomain::Val{K}
    ) where {T, N, K}
    ax_2d = matricize_axes(SectorFusion(), a, ndims_codomain)
    a_2d = similar(a, ax_2d)

    # CartesianIndices for mapping 2D block index → per-axis block indices
    codomain_nblocks = Tuple(blocklength.(axes(a)[1:K]))
    domain_nblocks = Tuple(blocklength.(axes(a)[(K + 1):N]))
    cod_cart = CartesianIndices(codomain_nblocks)
    dom_cart = CartesianIndices(domain_nblocks)

    for bI_src in eachblockstoredindex(a)
        src = Tuple(bI_src)
        # Split into codomain and domain block indices
        ci_cod = ntuple(i -> Int(src[i]), Val(K))
        ci_dom = ntuple(i -> Int(src[K + i]), Val(N - K))

        # 2D block index
        row_block = LinearIndices(cod_cart)[ci_cod...]
        col_block = LinearIndices(dom_cart)[ci_dom...]

        # Matricize the individual block (SectorArray handles fermionic phase)
        src_block = a[bI_src]
        dest_block = matricize(src_block, ndims_codomain)

        # Store in the 2D array
        a_2d[Block(row_block, col_block)] = dest_block
    end

    return a_2d
end

# ========================  AbelianArray unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::SectorDelta,
        codomain_axes::Tuple{Vararg{SectorRange}},
        domain_axes::Tuple{Vararg{SectorRange}}
    )
    return SectorDelta{eltype(m)}((codomain_axes..., domain_axes...))
end

# TODO: Port unmatricize(::SectorFusion, ::SectorMatrix, ...) with fermionic
# contraction phase. The old code used kroneckerfactors to split sector and data
# components; the new SectorArray stores them directly but the multiplicity
# information needed for reshaping the data component is not carried by SectorRange
# axes alone. Needed when contracting bare SectorArray tensors (not AbelianArray).

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::FusedSectorMatrix,
        codomain_axes::Tuple{Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    blocked_axes = (codomain_axes..., domain_axes...)
    if isempty(blocked_axes)
        error("Scalar unmatricize not yet supported for FusedSectorMatrix")
    end

    fused_axes = matricize_axes(SectorFusion(), m, codomain_axes, domain_axes)
    m_abelian = AbelianArray(m)
    blockperms = sectorsortperm.(fused_axes)
    J = map(invblockmergeperm, fused_axes, blockperms, axes(m_abelian))
    m_split = m_abelian[J...]
    return block_unreshape(m_split, codomain_axes, domain_axes)
end

"""
    block_unreshape(m::AbelianArray{T,2}, codomain_axes, domain_axes) -> AbelianArray{T,N}

Inverse of `block_reshape`. Reshapes a 2D AbelianArray with unfused graded axes
back to an N-d AbelianArray by splitting each 2D block into its constituent
N-d blocks.
"""
function block_unreshape(
        m::AbelianMatrix{T}, codomain_axes::Tuple, domain_axes::Tuple
    ) where {T}
    N = length(codomain_axes) + length(domain_axes)
    dest_axes = (codomain_axes..., domain_axes...)
    a = similar(m, dest_axes)

    cod_cart = CartesianIndices(Tuple(map(blocklength, codomain_axes)))
    dom_cart = CartesianIndices(Tuple(map(blocklength, domain_axes)))

    for bI_src in eachblockstoredindex(m)
        row_block = Int(Tuple(bI_src)[1])
        col_block = Int(Tuple(bI_src)[2])
        dest_bk = (Tuple(cod_cart[row_block])..., Tuple(dom_cart[col_block])...)

        src_block = m[bI_src]
        dest_sects = ntuple(d -> sectors(dest_axes[d])[dest_bk[d]], Val(N))
        dest_dims = ntuple(d -> blocklengths(dest_axes[d])[dest_bk[d]], Val(N))
        dest_block = SectorArray(dest_sects, reshape(src_block.data, dest_dims))
        a[Block(dest_bk...)] = dest_block
    end

    return a
end

# ========================  permutedimsopadd!  ========================

function TensorAlgebra.permutedimsopadd!(
        y::SectorArray, op, x::SectorArray, perm,
        α::Number, β::Number
    )
    sector(y) == permutedims(sector(x), perm) || throw(DimensionMismatch())
    phase = fermion_permutation_phase(sector(x), perm)
    TensorAlgebra.permutedimsopadd!(y.data, op, x.data, perm, phase * α, β)
    return y
end

function TensorAlgebra.permutedimsopadd!(
        y::AbelianArray{<:Any, N}, op, x::AbelianArray{<:Any, N}, perm,
        α::Number, β::Number
    ) where {N}
    if !iszero(β)
        for bI in eachblockstoredindex(y)
            y_b = @view!(y[bI])
            idperm = ntuple(identity, ndims(y_b))
            TensorAlgebra.permutedimsopadd!(y_b, identity, y_b, idperm, β, false)
        end
    end
    for bI in eachblockstoredindex(x)
        b = Tuple(bI)
        b_dest = Block(ntuple(i -> b[perm[i]], N))
        y_b = @view!(y[b_dest])
        x_b = x[bI]
        TensorAlgebra.permutedimsopadd!(y_b, op, x_b, perm, α, true)
    end
    return y
end
