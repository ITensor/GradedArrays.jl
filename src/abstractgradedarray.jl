"""
    AbstractGradedArray{T,N} <: AbstractArray{T,N}

Abstract supertype for graded (symmetry-structured) arrays whose axes carry sector labels.
Concrete subtypes include [`AbelianArray`](@ref) and [`FusedSectorMatrix`](@ref).
"""
abstract type AbstractGradedArray{T, N} <: AbstractArray{T, N} end

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
