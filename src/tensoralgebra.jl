using BlockArrays: Block, BlockVector, blocks, eachblockaxes1
using BlockSparseArrays: BlockSparseArray, blockrange, blockreshape, eachblockstoredindex
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

    # Compute the unmerged fused axes (block structure before sectormergesort).
    fused_axes = matricize_axes(BlockReshapeFusion(), m, codomain_axes, domain_axes)

    # Split merged blocks back to unmerged block structure, then unmatricize.
    m_unmerged = unsectormergesort(m, fused_axes)
    return unmatricize(FusionStyle(BlockSparseArray), m_unmerged, blocked_axes)
end

# Split merged blocks back to unmerged block structure (inverse of sectormergesort for arrays).
_asblock(i::Block{1}) = i
_asblock(i::Integer) = Block(i)

function unsectormergesort(m::AbstractArray, original_axes)
    N = ndims(m)
    perms = sectormergesortperm.(original_axes)

    result = zeros(eltype(m), original_axes)

    for I in eachblockstoredindex(m)
        dest_idx = map(Int, Tuple(I))
        merged_block_data = m[I]

        # For each dimension, find which source blocks map to this merged block
        # and their offsets within it.
        src_groups = ntuple(N) do dim
            grp = blocks(perms[dim])[dest_idx[dim]]
            cumoff = 0
            map(eachindex(grp)) do k
                off = cumoff
                src_block = _asblock(grp[k])
                cumoff += length(original_axes[dim][src_block])
                return (src_block, off)
            end
        end

        # Iterate over all combinations of source blocks.
        for src_combo in Iterators.product(src_groups...)
            src_blocks = ntuple(dim -> src_combo[dim][1], N)  # Block{1, Int64}
            src_offsets = ntuple(dim -> src_combo[dim][2], N)
            src_sizes = ntuple(dim -> length(original_axes[dim][src_blocks[dim]]), N)
            src_region =
                ntuple(dim -> (src_offsets[dim] + 1):(src_offsets[dim] + src_sizes[dim]), N)
            src_data = merged_block_data[src_region...]
            if !all(iszero, src_data)
                result[src_blocks...] = src_data
            end
        end
    end

    return result
end

# Materialize grouped block indexing used by `sectormergesort`.
function Base.getindex(
        a::BlockSparseArray,
        perms::Vararg{<:BlockVector{<:Union{Int, Block{1}}}, N}
    ) where {N}
    axs = axes(a)
    merged_axes = sectormergesort.(axs)
    result = zeros(eltype(a), merged_axes)

    function find_dest_and_pos(perm::BlockVector, src_block::Block{1})
        for (igroup, grp) in enumerate(blocks(perm))
            pos = findfirst(i -> _asblock(i) == src_block, grp)
            isnothing(pos) || return (igroup, pos)
        end
        return error("Block $src_block not found in permutation")
    end

    # Group stored source blocks by destination grouped block and local offsets.
    dest_groups = Dict{NTuple{N, Int}, Vector{Tuple{Block{N, Int}, NTuple{N, Int}}}}()
    for I in eachblockstoredindex(a)
        src_block_indices = Tuple(I)
        dest_and_pos = ntuple(N) do dim
            return find_dest_and_pos(perms[dim], src_block_indices[dim])
        end
        dest_idx = ntuple(dim -> dest_and_pos[dim][1], N)

        offsets = ntuple(N) do dim
            igroup, pos = dest_and_pos[dim]
            grp = blocks(perms[dim])[igroup]
            return sum(length(axs[dim][_asblock(grp[k])]) for k in 1:(pos - 1); init = 0)
        end

        push!(get!(dest_groups, dest_idx, []), (I, offsets))
    end

    for (dest_idx, src_list) in dest_groups
        dest_size = ntuple(dim -> length(merged_axes[dim][Block(dest_idx[dim])]), N)
        dest_data = zeros(eltype(a), dest_size)

        for (I, offsets) in src_list
            src_data = a[I]
            src_sizes = size(src_data)
            dest_region =
                ntuple(dim -> (offsets[dim] + 1):(offsets[dim] + src_sizes[dim]), N)
            dest_data[dest_region...] = src_data
        end

        result[Block.(dest_idx)...] = dest_data
    end

    return result
end

function sectormergesort(a::AbstractArray)
    I = sectormergesortperm.(axes(a))
    return a[I...]
end
