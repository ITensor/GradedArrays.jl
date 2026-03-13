struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:SectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:GradedArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorUnitRange}) = SectorFusion()

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
function TensorAlgebra.matricize(
        ::SectorFusion, a::SectorArray, length_codomain::Val
    )
    asectors, adata = kroneckerfactors(a)
    asectors_reshaped = matricize(asectors, length_codomain)
    adata_reshaped = matricize(adata, length_codomain)

    T = TKS.sectorscalartype(sector_type(a))
    phase = prod(
        ntuple(length_codomain) do i
            return ifelse(isdual(axes(a, i)), twist(sectors(a, i)), one(T))
        end
    )
    isone(phase) || (adata_reshaped .*= phase)

    return SectorArray(asectors_reshaped.sectors, adata_reshaped)
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

    fused_axes = matricize_axes(BlockReshapeFusion(), m, codomain_axes, domain_axes)
    blockperms = sectorsortperm.(fused_axes)
    J = map(invblockmergeperm, fused_axes, blockperms, axes(m))
    return unmatricize(FusionStyle(BlockSparseArray), m[J...], blocked_axes)
end
function TensorAlgebra.unmatricize(
        ::SectorFusion, m::SectorMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}}
    )
    msectors, mdata = kroneckerfactors(m)
    msectors = unmatricize(
        kroneckerfactors(m, 1),
        kroneckerfactors.(codomain_axes, 1),
        kroneckerfactors.(domain_axes, 1)
    )
    mdata = unmatricize(
        kroneckerfactors(m, 2),
        kroneckerfactors.(codomain_axes, 2),
        kroneckerfactors.(domain_axes, 2)
    )

    T = TKS.sectorscalartype(sector_type(m))
    phase = prod(
        ntuple(length(codomain_axes)) do i
            ax = kroneckerfactors(codomain_axes[i], 1)
            return ifelse(isdual(ax), twist(ax), one(T))
        end
    )
    isone(phase) || (mdata .*= phase)

    return SectorArray(msectors.sectors, mdata)
end

function TensorAlgebra.permutedimsadd!(
        y::SectorArray, x::SectorArray, perm,
        α::Number, β::Number
    )
    ysectors, ydata = kroneckerfactors(y)
    xsectors, xdata = kroneckerfactors(x)
    ysectors == permutedims(xsectors, perm) || throw(DimensionMismatch())
    phase = permutation_phase(xsectors, perm)
    TensorAlgebra.permutedimsadd!(ydata, xdata, perm, phase * α, β)
    return y
end

# Sort the blocks by sector and then merge the common sectors.
function sectormergesort(a::AbstractArray)
    I = sectormergesortperm.(axes(a))
    return a[I...]
end

# Returns a Vector{BlockIndexRange{1}} mapping each block of fine_ax (in original order)
# to its position (block + subrange) within the merged axis merged_ax, given the block
# permutation blockperm used to sort and merge fine_ax into merged_ax.
# Requires that blocks of fine_ax subdivide blocks of merged_ax.
function invblockmergeperm(fine_ax, blockperm, merged_ax)
    n = length(blockperm)
    bir_type = Base.promote_op(getindex, Block{1, Int}, UnitRange{Int})
    J = Vector{bir_type}(undef, n)
    j = 1
    offset = 0
    for k′ in 1:n
        k = Int(blockperm[k′])
        size_k = length(fine_ax[Block(k)])
        merged_block_size = length(merged_ax[Block(j)])
        offset + size_k ≤ merged_block_size ||
            throw(ArgumentError("fine_ax blocks do not subdivide merged_ax blocks"))
        J[k] = Block(j)[(offset + 1):(offset + size_k)]
        offset += size_k
        if offset == merged_block_size
            j += 1
            offset = 0
        end
    end
    return J
end

function checkindices(
        a::GradedArray{<:Any, N}, I::NTuple{N, AbstractVector{<:BlockIndexRange{1}}}
    ) where {N}
    for d in 1:N
        nblocks_d = length(axes(a, d))
        for bir in I[d]
            Int(bir.block) ≤ nblocks_d ||
                throw(BlockBoundsError(a, ntuple(i -> i == d ? bir : I[i][1], Val(N))))
        end
    end
    return nothing
end

# Splitting: each I[d][k] = Block(b)[r] means dest block k comes from source block b
# at subrange r. This is the inverse of the merging getindex below.
function Base.getindex(
        a::GradedArray{<:Any, N}, I::Vararg{AbstractVector{<:BlockIndexRange{1}}, N}
    ) where {N}
    checkindices(a, I)
    ax_dest = ntuple(d -> only(axes(axes(a, d)[I[d]])), Val(N))
    a_dest = similar(a, ax_dest)
    # Map source block b → list of (dest BlockIndexRange, src subrange).
    # Stored blocks of a not referenced by I are skipped (partial block selection).
    src_to_dests = ntuple(Val(N)) do d
        key_type = Block{1, Int}
        dest_bir_type = Base.promote_op(getindex, key_type, Base.OneTo{Int})
        val_type = Tuple{dest_bir_type, UnitRange{Int}}
        dict = Dict{key_type, Vector{val_type}}()
        for k in eachindex(I[d])
            bir = I[d][k]
            b = Block(Int(bir.block))
            r = only(bir.indices)
            push!(get!(dict, b, val_type[]), (Block(k)[Base.axes1(r)], r))
        end
        return dict
    end
    for bI_src in eachblockstoredindex(a)
        src_tuple = Tuple(bI_src)
        all(d -> haskey(src_to_dests[d], src_tuple[d]), 1:N) || continue
        dest_refs = ntuple(d -> src_to_dests[d][src_tuple[d]], Val(N))
        for combo in Iterators.product(dest_refs...)
            src_r = ntuple(d -> combo[d][2], Val(N))
            src_data = @view(a[bI_src][src_r...])
            iszero(src_data) && continue
            dest_b = Block(ntuple(d -> only(Tuple(combo[d][1].block)), Val(N)))
            a_dest_b = @view!(a_dest[dest_b])
            dest_r = ntuple(d -> only(combo[d][1].indices), Val(N))
            copyto!(@view(a_dest_b[dest_r...]), src_data)
        end
    end
    return a_dest
end

# Merging: each I[d] groups source blocks into destination blocks.
function Base.getindex(
        a::GradedArray{<:Any, N}, I::Vararg{AbstractBlockVector{<:Block{1}}, N}
    ) where {N}
    ax_dest = ntuple(d -> Base.axes1(axes(a, d)[I[d]]), Val(N))
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
