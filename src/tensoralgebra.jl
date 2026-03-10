using BlockArrays: blocks, eachblockaxes1
using BlockSparseArrays: BlockSparseArray, blockrange, blockreshape
using GradedArrays: GradedArray, GradedUnitRange, SectorRange, flip, invblockperm,
    sectormergesortperm, sectorsortperm, trivial, unmerged_tensor_product, ×
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockedTuple, FusionStyle,
    ReshapeFusion, matricize, matricize_axes, tensor_product_axis, trivialbiperm,
    tuplemortar, unmatricize

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:SectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:GradedArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorUnitRange}) = SectorFusion()

using BlockArrays: AbstractBlockArray
const BlockReshapeFusion = typeof(FusionStyle(AbstractBlockArray))

function TensorAlgebra.trivial_axis(
        ::BlockReshapeFusion,
        ::Val{:codomain},
        a::GradedArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return trivial_gradedrange(axes(a))
end
function TensorAlgebra.trivial_axis(
        ::BlockReshapeFusion,
        ::Val{:domain},
        a::GradedArray,
        axes_codomain::Tuple{Vararg{AbstractUnitRange}},
        axes_domain::Tuple{Vararg{AbstractUnitRange}}
    )
    return flip(trivial_gradedrange(axes(a)))
end
function trivial_gradedrange(t::Tuple{Vararg{G}}) where {G <: GradedUnitRange}
    return trivial(first(t))
end
trivial_gradedrange(t::Tuple{Vararg{GradedUnitRange}}) = ⊗(trivial.(t)...)
function trivial_gradedrange(::Type{S}) where {S <: SectorRange}
    return gradedrange([trivial(S) => 1])
end

function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:codomain}, r1::SectorUnitRange, r2::SectorUnitRange
    )
    return r1 ⊗ r2
end
function TensorAlgebra.tensor_product_axis(
        ::SectorFusion, ::Val{:domain}, r1::SectorUnitRange, r2::SectorUnitRange
    )
    return flip(r1 ⊗ r2)
end
function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:codomain},
        r1::GradedUnitRange, r2::GradedUnitRange
    )
    return tensor_product_gradedrange(style, side, r1, r2)
end
function TensorAlgebra.tensor_product_axis(
        style::BlockReshapeFusion, side::Val{:domain},
        r1::GradedUnitRange, r2::GradedUnitRange
    )
    return tensor_product_gradedrange(style, side, r1, r2)
end
# TODO: Could this call out to a generic tensor_product_axis for AbstractBlockedUnitRange?
function tensor_product_gradedrange(
        ::BlockReshapeFusion, side::Val,
        r1::AbstractUnitRange, r2::AbstractUnitRange
    )
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    blockaxpairs = Iterators.product(eachblockaxes1(r1), eachblockaxes1(r2))
    blockaxs = map(blockaxpairs) do (b1, b2)
        # TODO: Store a FusionStyle for the blocks in `BlockReshapeFusion`
        # and use that here.
        return tensor_product_axis(side, b1, b2)
    end
    return mortar_axis(vec(blockaxs))
end

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbstractArray, length_codomain::Val
    )
    a_reshaped = matricize(BlockReshapeFusion(), a, length_codomain)
    return sectormergesort(a_reshaped)
end
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

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::SectorDelta,
        codomain_axes::Tuple{Vararg{SectorRange}},
        domain_axes::Tuple{Vararg{SectorRange}}
    )
    return SectorDelta{eltype(m)}((codomain_axes..., domain_axes...))
end
function TensorAlgebra.unmatricize(
        ::SectorFusion, m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}}
    )
    blocked_axes = tuplemortar((codomain_axes, domain_axes))
    if isempty(blocked_axes)
        # Handle edge case of empty blocked_axes, which can occur
        # when matricizing a 0-dimensional array (a scalar).
        a = similar(m, ())
        a[] = only(m)
        return a
    end

    # First, fuse axes to get `sectormergesortperm`.
    # Then unpermute the blocks.
    fused_axes = matricize_axes(BlockReshapeFusion(), m, codomain_axes, domain_axes)

    blockperms = sectorsortperm.(fused_axes)
    sorted_axes = map((r, I) -> only(axes(r[I])), fused_axes, blockperms)

    # TODO: This is doing extra copies of the blocks,
    # use `@view a[axes_prod...]` instead.
    # That will require implementing some reindexing logic
    # for this combination of slicing.
    m_unblocked = m[sorted_axes...]
    m_blockpermed = m_unblocked[invblockperm.(blockperms)...]
    return unmatricize(FusionStyle(BlockSparseArray), m_blockpermed, blocked_axes)
end

# Sort the blocks by sector and then merge the common sectors.
function sectormergesort(a::AbstractArray)
    I = sectormergesortperm.(axes(a))
    return a[I...]
end

using BlockArrays: AbstractBlockVector, Block
function Base.getindex(
        a::GradedArray{<:Any, N},
        I::Vararg{AbstractBlockVector{<:Block{1}}, N}
    ) where {N}
    axes_dest = ntuple(d -> only(axes(axes(a, d)[I[d]])), N)
    a_dest = BlockSparseArray{eltype(a)}(undef, axes_dest)

    grouped_blocks = ntuple(N) do d
        return map(blocks(I[d])) do bI
            return map(k -> I[d][k], only(bI.indices))
        end
    end
    grouped_ranges = ntuple(N) do d
        return map(grouped_blocks[d]) do bs
            lengths = map(b -> length(axes(a, d)[b]), bs)
            starts = cumsum(vcat(1, lengths[1:(end - 1)]))
            return map((s, n) -> s:(s + n - 1), starts, lengths)
        end
    end

    for I_dest in CartesianIndices(map(gs -> Base.OneTo(length(gs)), grouped_blocks))
        src_blocks = ntuple(d -> grouped_blocks[d][I_dest[d]], N)
        src_ranges = ntuple(d -> grouped_ranges[d][I_dest[d]], N)
        block_dims = ntuple(d -> sum(length.(src_ranges[d])), N)
        block_data = zeros(eltype(a), block_dims)
        block_stored = false

        src_group_indices = Iterators.product(map(Base.OneTo ∘ length, src_blocks)...)
        for I_src in src_group_indices
            bI_src = Block(ntuple(d -> src_blocks[d][I_src[d]], N))
            !isstored(a, bI_src) && continue
            block_stored = true
            rI_dest = ntuple(d -> src_ranges[d][I_src[d]], N)
            block_data[rI_dest...] = a[bI_src]
        end

        block_stored || continue
        a_dest[Block(Tuple(I_dest))] = block_data
    end
    return a_dest
end
