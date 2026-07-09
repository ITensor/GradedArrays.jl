"""
    AbstractSectorDelta{T,S,N} <: AbstractArray{T,N}

Abstract supertype for structural (Kronecker/identity) tensors associated to sector labels.
Concrete subtypes:

  - [`AbelianSectorDelta`](@ref): unfused N-D abelian structural tensor (product of Kronecker deltas)
  - [`SectorIdentity`](@ref): fused 2D structural factor (identity matrix per coupled sector)
"""
abstract type AbstractSectorDelta{T, S, N} <: AbstractArray{T, N} end

sectortype(::Type{<:AbstractSectorDelta{T, S}}) where {T, S} = S

Base.copy(A::AbstractSectorDelta) = A
Base.size(A::AbstractSectorDelta) = length.(axes(A))

# A 2D structural delta on a diagonal (coupled) block is the identity over the sector, so its
# trace is the sector's quantum dimension: the length of the diagonal.
LinearAlgebra.tr(A::AbstractSectorDelta{<:Any, <:Any, 2}) = size(A, 1)
