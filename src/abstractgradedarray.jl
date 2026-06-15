"""
    AbstractGradedArray{T,N} <: AbstractArray{T,N}

Abstract supertype for graded (symmetry-structured) arrays whose axes carry sector labels.
Concrete subtypes include [`AbelianGradedArray`](@ref) and [`FusedGradedMatrix`](@ref).
"""
abstract type AbstractGradedArray{T, N} <: AbstractArray{T, N} end
const AbstractGradedMatrix{T} = AbstractGradedArray{T, 2}

# Used by NamedDimsArrays broadcast alignment. Eager for simplicity for now,
# pending the follow-ups on lazy permutations and the `FI.permuteddims`
# interface itself.
FI.permuteddims(a::AbstractGradedArray, perm) = permutedims(a, perm)

function BlockSparseArrays.isblockdiagonal(A::AbstractGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

# Block-aware `LinearAlgebra.isdiag` for graded matrices. The generic
# `LinearAlgebra.isdiag` would fall through to `_isbanded_impl`'s scalar-indexed
# `iszero(view)` iteration, which throws on block storage. A graded matrix is
# diagonal iff it is block-diagonal (no off-diagonal blocks stored) and each
# stored block is itself diagonal — the latter checked block-by-block to avoid
# materializing the whole matrix.

# Scalar indexing is not supported for graded arrays.
function Base.getindex(::AbstractGradedArray, ::Vararg{Int})
    return error(
        "Scalar indexing is not supported for AbstractGradedArray. Use block indexing."
    )
end
function Base.setindex!(::AbstractGradedArray, _, ::Vararg{Int})
    return error(
        "Scalar indexing is not supported for AbstractGradedArray. Use block indexing."
    )
end

# ---------------------------------------------------------------------------
#  Block indexing interface
#
#  Concrete subtypes must implement:
#    view(a::ConcreteType, ::Block{N})  → sector-wrapped view (e.g. SectorMatrix)
#
#  Everything else is derived here.
# ---------------------------------------------------------------------------

function Base.view(a::AbstractGradedArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    return view(a, Block(Int.(I)))
end
# Disambiguate against subtype-specific `view(::ConcreteGradedVector, ::Block{1})` methods.
Base.view(a::AbstractGradedArray{T, 1}, I::Block{1}) where {T} = view(a, Block((Int(I),)))

function Base.getindex(a::AbstractGradedArray{T, N}, I::Block{N}) where {T, N}
    return copy(view(a, I))
end
function Base.getindex(a::AbstractGradedArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    return a[Block(Int.(I))]
end
# Disambiguate the N=1 case: route through the `Block{N}` method to avoid recursion.
Base.getindex(a::AbstractGradedArray{T, 1}, I::Block{1}) where {T} = copy(view(a, I))

function Base.setindex!(a::AbstractGradedArray{<:Any, N}, value, I::Block{N}) where {N}
    return setindex!(a, value, Tuple(I)...)
end
function Base.setindex!(
        a::AbstractGradedArray{<:Any, N}, value, I::Vararg{Block{1}, N}
    ) where {N}
    view(a, I...) .= value
    return a
end
function Base.setindex!(a::AbstractGradedArray{<:Any, 1}, value, I::Block{1})
    view(a, I) .= value
    return a
end

# ---------------------------------------------------------------------------
#  Data indexing — raw block data without sector wrappers
#
#  Built on top of Block view: view(a, Data(I)) = data(view(a, Block(I)))
# ---------------------------------------------------------------------------

function Base.view(a::AbstractGradedArray{T, N}, I::Data{N}) where {T, N}
    return data(view(a, Block(I)))
end

function Base.getindex(a::AbstractGradedArray{T, N}, I::Data{N}) where {T, N}
    return copy(view(a, I))
end

function Base.setindex!(
        a::AbstractGradedArray{<:Any, N}, value::AbstractArray{<:Any, N}, I::Data{N}
    ) where {N}
    view(a, I) .= value
    return a
end

# ---------------------------------------------------------------------------
#  Accessors
# ---------------------------------------------------------------------------

datatype(a::AbstractGradedArray) = datatype(typeof(a))
sectortype(a::AbstractGradedArray) = sectortype(typeof(a))

# ---------------------------------------------------------------------------
#  fill! / zero! / scale! — block-wise over the stored blocks
#
#  Defined once via the `eachblockstoredindex`/`view` interface every
#  `AbstractGradedArray` implements, so every concrete subtype is covered.
#  These only touch stored (symmetry-allowed) blocks, so a nonzero `fill!`
#  value leaves the forbidden positions at zero.
# ---------------------------------------------------------------------------

scale!(a::AbstractArray, β::Number) = (a .*= β; a)

function scale!(a::AbstractGradedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

function FI.zero!(a::AbstractGradedArray)
    for bI in eachblockstoredindex(a)
        FI.zero!(view(a, bI))
    end
    return a
end

function Base.fill!(a::AbstractGradedArray, v)
    for bI in eachblockstoredindex(a)
        fill!(view(a, bI), v)
    end
    return a
end

# ---------------------------------------------------------------------------
#  Display — convert to BlockSparseArray for printing
# ---------------------------------------------------------------------------

using BlockSparseArrays: BlockSparseArray

function _to_blocksparsearray(a::AbstractGradedArray{T, N}) where {T, N}
    blocked_axes = map(g -> blockedrange(blocklengths(g)), axes(a))
    bsa = BlockSparseArray{T}(undef, blocked_axes)
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        bsa[bI] = collect(Array(sector(blk)) ⊗ data(blk))
    end
    return bsa
end

function Base.print_array(io::IO, a::AbstractGradedArray)
    return Base.print_array(io, _to_blocksparsearray(a))
end
