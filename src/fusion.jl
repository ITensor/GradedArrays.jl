using TensorAlgebra: tensor_product_axis, trivial_axis

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:AbstractSectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbstractSectorArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbstractGradedArray}) = SectorFusion()

# ========================  trivial_axis  ========================

function trivial_gradedrange(t::Tuple{Vararg{GradedOneTo}})
    return tensor_product(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: SectorRange}
    return gradedrange([trivial(S) => 1])
end

function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:codomain},
        a::AbelianGradedArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_gradedrange(axes(a))
end
function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:domain},
        a::AbelianGradedArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return flip(trivial_gradedrange(axes(a)))
end

function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:codomain},
        a::FusedGradedMatrix,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_gradedrange((axes_codomain..., axes_domain...))
end
function TensorAlgebra.trivial_axis(
        ::SectorFusion,
        ::Val{:domain},
        a::FusedGradedMatrix,
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

# ========================  AbelianGradedArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianGradedArray, ndims_codomain::Val{K}
    ) where {K}
    a_reshaped = block_reshape(a, Val(K))
    a_merged = sectormergesort(a_reshaped)
    return FusedGradedMatrix(a_merged)
end

"""
    block_reshape(a::AbelianGradedArray, ndims_codomain::Val{K}) -> AbelianGradedArray{T,2}

Reshape an N-d AbelianGradedArray into a 2D AbelianGradedArray by grouping the first K
dimensions as codomain and the remaining as domain. Computes unfused 2D graded
axes via `tensor_product_axis`, then permutes and reshapes each block's data.
"""
function block_reshape(
        a::AbelianGradedArray{T, N}, ndims_codomain::Val{K}
    ) where {T, N, K}
    ax_2d = matricize_axes(SectorFusion(), a, ndims_codomain)
    a_2d = FI.zero!(similar(a, ax_2d))

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

        # Matricize the individual block (AbelianSectorArray handles fermionic phase)
        src_block = a[bI_src]
        dest_block = matricize(src_block, ndims_codomain)

        # Store in the 2D array
        a_2d[Block(row_block, col_block)] = dest_block
    end

    return a_2d
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

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::FusedGradedMatrix,
        codomain_axes::Tuple{Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    blocked_axes = (codomain_axes..., domain_axes...)
    if isempty(blocked_axes)
        error("Scalar unmatricize not yet supported for FusedGradedMatrix")
    end

    fused_axes = matricize_axes(SectorFusion(), m, codomain_axes, domain_axes)
    m_abelian = AbelianGradedArray(m)
    blockperms = sectorsortperm.(fused_axes)
    J = map(invblockmergeperm, fused_axes, blockperms, axes(m_abelian))
    m_split = m_abelian[J...]
    return block_unreshape(m_split, codomain_axes, domain_axes)
end

"""
    block_unreshape(m::AbelianGradedArray{T,2}, codomain_axes, domain_axes) -> AbelianGradedArray{T,N}

Inverse of `block_reshape`. Reshapes a 2D AbelianGradedArray with unfused graded axes
back to an N-d AbelianGradedArray by splitting each 2D block into its constituent
N-d blocks.
"""
function block_unreshape(
        m::AbelianGradedMatrix{T}, codomain_axes::Tuple, domain_axes::Tuple
    ) where {T}
    N = length(codomain_axes) + length(domain_axes)
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

# ========================  Allowed block keys  ========================

"""
    allowedblocks(axs::NTuple{N, GradedOneTo}) -> Vector{Block{N, Int}}

Return the `Block` indices of all allowed (zero-flux) blocks for a graded
array with axes `axs`.

Uses a codomain/domain split (K=1) and fusion to enumerate allowed blocks
efficiently. The domain axes are fused into a single unfused `GradedOneTo`
via `matricize_axes`, then domain blocks are grouped by sector in a hash table.
For each codomain sector, matching domain blocks are found via lookup, and the
2D (codomain, domain) indices are mapped back to N-d via `CartesianIndices`.

Cost: O(B_cod + B_dom + #allowed) instead of the O(B_cod × B_dom) naïve
Cartesian filter.
"""
function allowedblocks(axs::NTuple{N, GradedOneTo{I}}) where {N, I}
    N == 0 && return Block{0, Int}[Block()]
    codomain_axs = (axs[1],)
    domain_axs = Base.tail(axs)

    # TODO: The dummy array is only needed because TensorAlgebra.trivial_axis dispatches
    # on an array argument. Adding a trivial_axis overload that takes just the axes would
    # eliminate this.
    dummy = AbelianGradedArray{Float64, N, Array{Float64, N}, I}(
        Dict{NTuple{N, Int}, Array{Float64, N}}(), axs
    )
    unfused_cod, unfused_dom = matricize_axes(
        SectorFusion(), dummy, codomain_axs, domain_axs
    )

    # Group unfused domain blocks by dual(sector) for fast lookup
    dom_secs = sectors(unfused_dom)
    cod_secs = sectors(unfused_cod)
    dom_by_sector = Dict{eltype(cod_secs), Vector{Int}}()
    for (j, s) in enumerate(dom_secs)
        push!(get!(dom_by_sector, dual(s), Int[]), j)
    end

    # Map 2D (codomain_block, domain_block) to N-d Block indices
    codomain_nblocks = Tuple(blocklength.(codomain_axs))
    domain_nblocks = Tuple(blocklength.(domain_axs))
    cod_cart = CartesianIndices(codomain_nblocks)
    dom_cart = CartesianIndices(domain_nblocks)
    bks = Block{N, Int}[]
    for (i, s) in enumerate(cod_secs)
        for j in get(dom_by_sector, s, Int[])
            push!(bks, Block((Tuple(cod_cart[i])..., Tuple(dom_cart[j])...)))
        end
    end
    return bks
end
