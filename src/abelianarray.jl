# ===========================================================================
#  AbelianArray — dict-of-keys graded array with GradedIndices axes
# ===========================================================================

"""
    AbelianArray{T,N,I<:TKS.Sector,D<:AbstractArray{T,N}} <: AbstractGradedArray{T,N}

A graded array that stores non-zero blocks in a dictionary keyed by block indices.
Each axis is a [`GradedIndices`](@ref) carrying sector labels, multiplicities, and a dual flag.

Blocks are stored as plain dense arrays of type `D` (default `Array{T,N}`).
Accessing a block via `a[Block(i,j)]` returns a [`SectorArray`](@ref) wrapping the data
with the appropriate sector labels and dual flags.
"""
struct AbelianArray{T, N, I <: TKS.Sector, D <: AbstractArray{T, N}} <:
    AbstractGradedArray{T, N}
    axes::NTuple{N, GradedIndices{I}}
    blockdata::Dict{NTuple{N, Int}, D}
end

# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

function AbelianArray{T}(
        ::UndefInitializer, axs::NTuple{N, GradedIndices{I}}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T, N, I, Array{T, N}}(axs, Dict{NTuple{N, Int}, Array{T, N}}())
end

function AbelianArray{T}(
        init::UndefInitializer, axs::Vararg{GradedIndices{I}, N}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T}(init, axs)
end

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

"""
    _block_length(g::GradedIndices, k::Int)

Total length of block `k` in axis `g`: quantum dimension times multiplicity.
"""
function _block_length(g::GradedIndices, k::Int)
    return TKS.dim(labels(g)[k]) * sector_multiplicities(g)[k]
end

# ---------------------------------------------------------------------------
#  AbstractArray interface
# ---------------------------------------------------------------------------

Base.size(a::AbelianArray) = map(length, a.axes)
Base.axes(a::AbelianArray) = a.axes

function Base.getindex(a::AbelianArray, I::Vararg{Int})
    return error(
        "Scalar indexing is not supported for AbelianArray. Use block indexing: a[Block(i,j)]"
    )
end
function Base.setindex!(a::AbelianArray, v, I::Vararg{Int})
    return error(
        "Scalar indexing is not supported for AbelianArray. Use block indexing: a[Block(i,j)] = v"
    )
end

# ---------------------------------------------------------------------------
#  Block indexing
# ---------------------------------------------------------------------------

# Get or create a block, returning a SectorArray wrapping the data in-place.
# Mutations to the returned SectorArray propagate to blockdata.
function BlockSparseArrays.view!(
        a::AbelianArray{T, N}, I::Vararg{Block{1}, N}
    ) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    if !haskey(a.blockdata, bk)
        block_dims = ntuple(d -> _block_length(a.axes[d], bk[d]), Val(N))
        a.blockdata[bk] = zeros(T, block_dims)
    end
    block_labels = ntuple(d -> labels(a.axes[d])[bk[d]], Val(N))
    block_isdual = ntuple(d -> isdual(a.axes[d]), Val(N))
    return SectorArray(block_labels, block_isdual, a.blockdata[bk])
end
function BlockSparseArrays.view!(a::AbelianArray{<:Any, N}, I::Block{N}) where {N}
    return BlockSparseArrays.view!(a, Tuple(I)...)
end

function Base.getindex(a::AbelianArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    block_labels = ntuple(d -> labels(a.axes[d])[bk[d]], Val(N))
    block_isdual = ntuple(d -> isdual(a.axes[d]), Val(N))
    if haskey(a.blockdata, bk)
        return SectorArray(block_labels, block_isdual, a.blockdata[bk])
    else
        block_dims = ntuple(d -> _block_length(a.axes[d], bk[d]), Val(N))
        return SectorArray(block_labels, block_isdual, zeros(T, block_dims))
    end
end

# Single Block{N} argument: splat to N Block{1} arguments
function Base.getindex(a::AbelianArray{T, N}, I::Block{N}) where {T, N}
    return a[Block.(Tuple(I))...]
end

function Base.setindex!(a::AbelianArray{T, N}, value, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    raw = value isa SectorArray ? value.data : value
    a.blockdata[bk] = convert(Array{T, N}, raw)
    return a
end

function Base.setindex!(a::AbelianArray{T, N}, value, I::Block{N}) where {T, N}
    return setindex!(a, value, Block.(Tuple(I))...)
end

# ---------------------------------------------------------------------------
#  Merging getindex: reindex by block permutation/merge
# ---------------------------------------------------------------------------

# Merging: each I[d] groups source blocks into destination blocks.
# Follows the same pattern as the old GradedArray getindex(::AbstractBlockVector...).
function Base.getindex(
        a::AbelianArray{T, N}, I::Vararg{AbstractBlockVector{<:Block{1}}, N}
    ) where {T, N}
    ax_dest = ntuple(d -> a.axes[d][I[d]], Val(N))
    a_dest = AbelianArray{T}(undef, ax_dest)
    ax = a.axes
    # Map source Block → BlockIndexRange encoding dest block + subrange within it
    src_to_dest = ntuple(Val(N)) do d
        key_type = eltype(I[d])
        range_type = UnitRange{Int}
        val_type = Base.promote_op(getindex, key_type, range_type)
        dict = Dict{key_type, val_type}()
        for j in eachindex(blocks(I[d]))
            sub_blocks = I[d][Block(j)]
            start = 1
            for b in sub_blocks
                blen = blocklengths(ax[d])[Int(b)]
                r = Base.OneTo(blen) .+ (start - 1)
                dict[b] = Block(j)[r]
                start += blen
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
        copyto!(view(a_dest_b, dest_r...), a[bI_src])
    end
    return a_dest
end

# ---------------------------------------------------------------------------
#  eachblockstoredindex
# ---------------------------------------------------------------------------

function BlockSparseArrays.eachblockstoredindex(a::AbelianArray{T, N}) where {T, N}
    return (Block(k) for k in keys(a.blockdata))
end

# ---------------------------------------------------------------------------
#  similar
# ---------------------------------------------------------------------------

function Base.similar(a::AbelianArray{T, N, I}) where {T, N, I}
    return AbelianArray{T}(undef, a.axes)
end

function Base.similar(a::AbelianArray{<:Any, N, I}, ::Type{S}) where {S, N, I}
    return AbelianArray{S}(undef, a.axes)
end

function Base.similar(
        ::AbelianArray{<:Any, <:Any, I}, ::Type{S}, axs::NTuple{M, GradedIndices{I}}
    ) where {S, M, I}
    return AbelianArray{S}(undef, axs)
end

# ---------------------------------------------------------------------------
#  sector_type
# ---------------------------------------------------------------------------

sector_type(::Type{<:AbelianArray{T, N, I}}) where {T, N, I} = SectorRange{I}

# ---------------------------------------------------------------------------
#  permutedims
# ---------------------------------------------------------------------------

function Base.permutedims(a::AbelianArray{<:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> a.axes[perm[i]], Val(N))
    a_dest = AbelianArray{eltype(a)}(undef, dest_axes)
    return permutedims!(a_dest, a, perm)
end

function Base.permutedims!(
        y::AbelianArray{<:Any, N}, x::AbelianArray{<:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ---------------------------------------------------------------------------
#  fill! / zero!
# ---------------------------------------------------------------------------

function Base.fill!(a::AbelianArray, v)
    if iszero(v)
        empty!(a.blockdata)
    else
        for bk in keys(a.blockdata)
            fill!(a.blockdata[bk], v)
        end
    end
    return a
end

# ---------------------------------------------------------------------------
#  Matrix multiplication (block-diagonal)
# ---------------------------------------------------------------------------

const AbelianMatrix{T, I, D} = AbelianArray{T, 2, I, D}

function _check_mul_axes(A::AbelianMatrix, B::AbelianMatrix)
    return A.axes[2] == dual(B.axes[1]) || throw(
        DimensionMismatch(
            "second axis of A, $(A.axes[2]), and first axis of B, $(B.axes[1]), must match"
        )
    )
end

function LinearAlgebra.mul!(
        C::AbelianMatrix, A::AbelianMatrix, B::AbelianMatrix, α::Number, β::Number
    )
    _check_mul_axes(A, B)

    # Scale existing blocks of C by β
    if !iszero(β)
        for bk in keys(C.blockdata)
            C.blockdata[bk] .*= β
        end
    else
        empty!(C.blockdata)
    end

    # Build lookup: col axis of A sector label → block index
    # Build lookup: row axis of B sector label → block index
    row_ea_B = eachblockaxis(B.axes[1])
    row_B_lookup = Dict(label(si) => i for (i, si) in enumerate(row_ea_B))

    for (bk_A, data_A) in A.blockdata
        row_A, col_A = bk_A
        # The contracted sector: col axis of A
        contracted_label = labels(A.axes[2])[col_A]
        # Find matching row in B: same label (dual convention handled by _check_mul_axes)
        contracted_label_B = isdual(A.axes[2]) ? contracted_label : dual(contracted_label)
        j_B = get(row_B_lookup, contracted_label_B, nothing)
        isnothing(j_B) && continue

        for (bk_B, data_B) in B.blockdata
            row_B, col_B = bk_B
            row_B == j_B || continue

            dest_bk = (row_A, col_B)
            C_block = view!(C, Block(dest_bk...))
            mul!(C_block.data, data_A, data_B, α, true)
        end
    end
    return C
end

function Base.:(*)(A::AbelianMatrix, B::AbelianMatrix)
    _check_mul_axes(A, B)
    T = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    C = AbelianArray{T}(undef, A.axes[1], B.axes[2])
    mul!(C, A, B, true, false)
    return C
end

# ---------------------------------------------------------------------------
#  show
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", a::AbelianArray{T, N}) where {T, N}
    block_str = join(map(g -> string(BlockArrays.blocklength(g)), a.axes), "×")
    size_str = join(map(string, size(a)), "×")
    nstored = length(a.blockdata)
    print(io, block_str, "-blocked ", size_str, " AbelianArray{", T, "}")
    print(io, " with ", nstored, " stored block", nstored == 1 ? "" : "s")
    return nothing
end

function Base.show(io::IO, a::AbelianArray{T, N}) where {T, N}
    block_str = join(map(g -> string(BlockArrays.blocklength(g)), a.axes), "×")
    size_str = join(map(string, size(a)), "×")
    print(io, block_str, "-blocked ", size_str, " AbelianArray{", T, "}")
    return nothing
end
