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
    axes_dest = ntuple(d -> only(axes(axes(a, d)[I[d]])), Val(N))
    a_dest = similar(a, axes_dest)

    # Group selected source blocks per destination block index along each dimension.
    grouped_blocks = ntuple(N) do d
        map(blocks(I[d])) do bI
            return collect(bI)
        end
    end

    # Map source block index -> (destination block index, destination subrange).
    source_to_dest = ntuple(N) do d
        block_map = Dict{Int, Tuple{Int, UnitRange{Int}}}()
        for (j, src_blocks) in pairs(grouped_blocks[d])
            offset = 1
            for b in src_blocks
                b_int = Int(b)
                haskey(block_map, b_int) &&
                    throw(
                    ArgumentError(
                        "Source block appears in multiple destination groups."
                    )
                )
                len_b = length(axes(a, d)[b])
                block_map[b_int] = (j, offset:(offset + len_b - 1))
                offset += len_b
            end
        end
        return block_map
    end

    # Populate destination blocks by placing each stored source block into the
    # corresponding destination subblock.
    for bI_src in eachblockstoredindex(a)
        bI_dest = Vector{Int}(undef, N)
        rI_dest = Vector{UnitRange{Int}}(undef, N)
        valid_dest = true
        for d in 1:N
            dst = get(source_to_dest[d], Int(Tuple(bI_src)[d]), nothing)
            if isnothing(dst)
                valid_dest = false
                break
            end
            j, r = dst
            bI_dest[d] = j
            rI_dest[d] = r
        end
        valid_dest || continue
        b_dest = Block(Tuple(bI_dest))
        a_dest_b = @view!(a_dest[b_dest])
        copyto!(@view(a_dest_b[Tuple(rI_dest)...]), a[bI_src])
    end
    return a_dest
end
