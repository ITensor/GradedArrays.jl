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
