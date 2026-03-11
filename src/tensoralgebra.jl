using BlockArrays: BlockIndexRange, blocks, eachblockaxes1
using BlockSparseArrays: BlockSparseArray, blockrange, blockreshape
using GradedArrays: GradedArray, GradedUnitRange, SectorRange, flip, gradedrange,
    invblockperm, sectormergesortperm, sectors, sectorsortperm, trivial,
    unmerged_tensor_product, ×
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

using BlockArrays: AbstractBlockVector, Block, BlockVector

# Splitting: each I[d][k] = Block(b)[r] means dest block k comes from source block b
# at subrange r. This is the inverse of the merging getindex below.
function Base.getindex(
        a::GradedArray{<:Any, N},
        I::Vararg{AbstractVector{<:BlockIndexRange{1}}, N}
    ) where {N}
    ax = axes(a)
    ax_dest = ntuple(Val(N)) do d
        return gradedrange(
            [
                sectors(ax[d])[only(Tuple(I[d][k].block))] => length(only(I[d][k].indices))
                    for k in eachindex(I[d])
            ]
        )
    end
    a_dest = similar(a, ax_dest)
    # Map source block b → list of (dest block k, src subrange r, dest subrange 1:length(r))
    src_to_dests = ntuple(Val(N)) do d
        dict = Dict{Block{1}, Vector{Tuple{Int, UnitRange{Int}, Base.OneTo{Int}}}}()
        for k in eachindex(I[d])
            bir = I[d][k]
            b = Block(only(Tuple(bir.block)))
            r = only(bir.indices)
            push!(
                get!(dict, b, Tuple{Int, UnitRange{Int}, Base.OneTo{Int}}[]),
                (k, r, Base.OneTo(length(r)))
            )
        end
        return dict
    end
    for bI_src in eachblockstoredindex(a)
        src_tuple = Tuple(bI_src)
        all(d -> haskey(src_to_dests[d], src_tuple[d]), 1:N) || continue
        dest_refs = ntuple(d -> src_to_dests[d][src_tuple[d]], Val(N))
        for combo in Iterators.product(dest_refs...)
            dest_b = Block(ntuple(d -> combo[d][1], Val(N)))
            a_dest_b = @view!(a_dest[dest_b])
            src_r = ntuple(d -> combo[d][2], Val(N))
            dest_r = ntuple(d -> combo[d][3], Val(N))
            copyto!(@view(a_dest_b[dest_r...]), @view(a[bI_src][src_r...]))
        end
    end
    return a_dest
end

# GradedUnitRange index: compute BlockIndexRange per block by linear scan,
# then delegate to the splitting getindex above.
# Assumes blocks of each I[d] subdivide blocks of the corresponding axis of a
# (as is the case in unmatricize, where I[d] is derived from the unmerged fused axis).
function Base.getindex(
        a::GradedArray{<:Any, N},
        I::Vararg{GradedUnitRange, N}
    ) where {N}
    J = map(axes(a), I) do src_ax, tgt_ax
        n = length(sectors(tgt_ax))
        J_d = Vector{BlockIndexRange{1}}(undef, n)
        j = 1
        offset = 0
        for k in 1:n
            size_k = length(tgt_ax[Block(k)])
            J_d[k] = Block(j)[(offset + 1):(offset + size_k)]
            offset += size_k
            if offset == length(src_ax[Block(j)])
                j += 1
                offset = 0
            end
        end
        return J_d
    end
    return a[J...]
end

# Vector{Block{1}} index: block permutation with no merging.
# Wraps as a singleton-group BlockVector and delegates to the merging getindex below.
function Base.getindex(
        a::GradedArray{<:Any, N},
        I::Vararg{Vector{<:Block{1}}, N}
    ) where {N}
    return a[map(v -> BlockVector(v, fill(1, length(v))), I)...]
end

# Merging: each I[d] groups source blocks into destination blocks.
function Base.getindex(
        a::GradedArray{<:Any, N},
        I::Vararg{AbstractBlockVector{<:Block{1}}, N}
    ) where {N}
    ax_dest = ntuple(d -> only(axes(axes(a, d)[I[d]])), Val(N))
    a_dest = similar(a, ax_dest)
    ax = axes(a)
    # Map source Block -> BlockIndexRange encoding dest block + subrange within it
    src_to_dest = ntuple(Val(N)) do d
        key_type = eltype(I[d])
        range_type = UnitRange{Int}
        val_type = Base.promote_op(getindex, key_type, range_type)
        dict = Dict{key_type, val_type}()
        for j in eachindex(blocks(I[d]))
            sub_blocks = I[d][Block(j)]
            start = 1
            for b in sub_blocks
                r = Base.OneTo(length(ax[d][b])) .+ (start - 1)
                dict[b] = Block(j)[r]
                start += length(r)
            end
        end
        return dict
    end
    for bI_src in eachblockstoredindex(a)
        src_tuple = Tuple(bI_src)
        dest_info = ntuple(d -> src_to_dest[d][src_tuple[d]], Val(N))
        dest_b = Block(map(di -> only(Tuple(di.block)), dest_info))
        a_dest_b = @view!(a_dest[dest_b])
        dest_r = map(di -> only(di.indices), dest_info)
        copyto!(@view(a_dest_b[dest_r...]), a[bI_src])
    end
    return a_dest
end
