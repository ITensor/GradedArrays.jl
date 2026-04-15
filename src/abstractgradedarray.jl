"""
    AbstractGradedArray{T,N} <: AbstractArray{T,N}

Abstract supertype for graded (symmetry-structured) arrays whose axes carry sector labels.
Concrete subtypes include [`AbelianGradedArray`](@ref) and [`FusedGradedMatrix`](@ref).
"""
abstract type AbstractGradedArray{T, N} <: AbstractArray{T, N} end
const AbstractGradedMatrix{T} = AbstractGradedArray{T, 2}

function BlockSparseArrays.isblockdiagonal(A::AbstractGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

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

function Base.getindex(a::AbstractGradedArray{T, N}, I::Block{N}) where {T, N}
    return copy(view(a, I))
end
function Base.getindex(a::AbstractGradedArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    return a[Block(Int.(I))]
end

function Base.setindex!(a::AbstractGradedArray{<:Any, N}, value, I::Block{N}) where {N}
    return setindex!(a, value, Tuple(I)...)
end
function Base.setindex!(
        a::AbstractGradedArray{<:Any, N}, value, I::Vararg{Block{1}, N}
    ) where {N}
    view(a, I...) .= value
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
