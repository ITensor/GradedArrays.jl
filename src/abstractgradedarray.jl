"""
    AbstractGradedArray{T,N} <: AbstractArray{T,N}

Abstract supertype for graded (symmetry-structured) arrays whose axes carry sector labels.
Concrete subtypes include [`AbelianArray`](@ref).
"""
abstract type AbstractGradedArray{T, N} <: AbstractArray{T, N} end
