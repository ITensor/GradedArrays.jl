using TensorAlgebra: tensor_product_axis, trivial_axis

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:SectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbelianArray}) = SectorFusion()

# ========================  trivial_axis  ========================

function trivial_gradedrange(t::Tuple{Vararg{GradedIndices}})
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

# ========================  tensor_product_axis  ========================

# SectorIndices level: fuse two block axes
function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:codomain}, r1::SectorIndices, r2::SectorIndices
    )
    return r1 ⊗ r2
end
function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:domain}, r1::SectorIndices, r2::SectorIndices
    )
    return flip(r1 ⊗ r2)
end

# GradedIndices level: iterate block axes, fuse each pair, reassemble
function TensorAlgebra.tensor_product_axis(
        style::SectorFusion, side::Val,
        r1::GradedIndices, r2::GradedIndices
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
    asectors = SectorDelta{eltype(a)}(a.labels, a.isdual)
    asectors_reshaped = matricize(asectors, Val(K))

    T = TKS.sectorscalartype(sector_type(a))
    phase = prod(
        ntuple(K) do i
            return ifelse(isdual(a, i), twist(sector(a, i)), one(T))
        end
    )

    adata_reshaped = matricize(a.data, Val(K))
    isone(phase) || (adata_reshaped = phase .* adata_reshaped)

    return SectorArray(asectors_reshaped.labels, asectors_reshaped.isdual, adata_reshaped)
end

# ========================  AbelianArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianArray, ndims_codomain::Val{K}
    ) where {K}
    a_reshaped = block_reshape(a, Val(K))
    return sectormergesort(a_reshaped)
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
    # Compute the unfused 2D axes by fusing codomain and domain block axes
    codomain_axes = axes(a)[1:K]
    domain_axes = axes(a)[(K + 1):N]
    row_axis = if isempty(codomain_axes)
        trivial_gradedrange(axes(a))
    else
        reduce(codomain_axes) do r1, r2
            return tensor_product_axis(SectorFusion(), Val(:codomain), r1, r2)
        end
    end
    col_axis = if isempty(domain_axes)
        flip(trivial_gradedrange(axes(a)))
    else
        reduce(domain_axes) do r1, r2
            return tensor_product_axis(SectorFusion(), Val(:domain), r1, r2)
        end
    end
    a_2d = AbelianArray{T}(undef, row_axis, col_axis)

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
        a_2d.blockdata[(row_block, col_block)] = dest_block.data
    end

    return a_2d
end

# ========================  permutedimsopadd!  ========================

function TensorAlgebra.permutedimsopadd!(
        y::SectorArray, op, x::SectorArray, perm,
        α::Number, β::Number
    )
    xdelta = SectorDelta{eltype(x)}(x.labels, x.isdual)
    ydelta = SectorDelta{eltype(y)}(y.labels, y.isdual)
    ydelta == permutedims(xdelta, perm) || throw(DimensionMismatch())
    phase = fermion_permutation_phase(xdelta, perm)
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
