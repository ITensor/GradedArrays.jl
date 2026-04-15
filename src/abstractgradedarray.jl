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
#  Display — convert to BlockedOneTo axes for block-structured printing
# ---------------------------------------------------------------------------

function _block_is_stored(a::AbstractGradedArray, bI::Block)
    return any(==(bI), eachblockstoredindex(a))
end

# Lightweight wrapper that provides BlockedOneTo axes and scalar getindex
# for Base/BlockArrays printing infrastructure.
struct _GradedPrintView{T, A <: AbstractGradedMatrix{T}} <: AbstractMatrix{T}
    parent::A
    rowaxis::BlockedOneTo{Int, Vector{Int}}
    colaxis::BlockedOneTo{Int, Vector{Int}}
end

function _GradedPrintView(a::AbstractGradedMatrix)
    rax = blockedrange(blocklengths(axes(a, 1)))
    cax = blockedrange(blocklengths(axes(a, 2)))
    return _GradedPrintView{eltype(a), typeof(a)}(a, rax, cax)
end

Base.size(p::_GradedPrintView) = size(p.parent)
Base.axes(p::_GradedPrintView) = (p.rowaxis, p.colaxis)

function Base.getindex(p::_GradedPrintView{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(p, i, j)
    bi = findblockindex(p.rowaxis, i)
    bj = findblockindex(p.colaxis, j)
    bI = Block(Int(block(bi)), Int(block(bj)))
    _block_is_stored(p.parent, bI) || return zero(T)
    blk = view(p.parent, bI)
    return @inbounds blk[blockindex(bi), blockindex(bj)]
end

function Base.replace_in_print_matrix(
        p::_GradedPrintView, i::Integer, j::Integer, s::AbstractString
    )
    bi = findblockindex(p.rowaxis, i)
    bj = findblockindex(p.colaxis, j)
    bI = Block(Int(block(bi)), Int(block(bj)))
    return _block_is_stored(p.parent, bI) ? s : Base.replace_with_centered_mark(s)
end

# Delegate to BlockArrays' block-structured row printing for separators.
function Base.print_matrix_row(
        io::IO, X::_GradedPrintView, A::Vector,
        i::Integer, cols::AbstractVector, sep::AbstractString,
        idxlast::Integer = last(axes(X, 2))
    )
    return BlockArrays._blockarray_print_matrix_row(io, X, A, i, cols, sep)
end

function Base.print_array(io::IO, a::AbstractGradedMatrix)
    return Base.print_array(io, _GradedPrintView(a))
end
