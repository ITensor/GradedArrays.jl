"""
    AbstractSectorDelta{T,N} <: AbstractArray{T,N}

Abstract supertype for structural (Kronecker/identity) tensors associated to sector labels.
Concrete subtypes:

  - [`AbelianSectorDelta`](@ref): unfused N-D abelian structural tensor (product of Kronecker deltas)
  - [`SectorIdentity`](@ref): fused 2D structural factor (identity matrix per coupled sector)
"""
abstract type AbstractSectorDelta{T, N} <: AbstractArray{T, N} end

# Used by NamedDimsArrays broadcast alignment. Eager for simplicity for now,
# pending the follow-ups on lazy permutations and the `FI.permuteddims`
# interface itself.
FI.permuteddims(a::AbstractSectorDelta, perm) = permutedims(a, perm)

Base.copy(A::AbstractSectorDelta) = A
Base.size(A::AbstractSectorDelta) = length.(axes(A))
