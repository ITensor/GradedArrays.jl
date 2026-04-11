"""
    AbstractSectorArray{T,N} <: AbstractArray{T,N}

Abstract supertype for data tensors labeled by sector information.
Concrete subtypes:

  - [`AbelianSectorArray`](@ref): unfused N-D abelian data tensor (one sector per axis)
  - [`SectorMatrix`](@ref): fused 2D data matrix (one coupled sector label)
"""
abstract type AbstractSectorArray{T, N} <: AbstractArray{T, N} end

"""
    data(sa::AbstractSectorArray)

Return the raw data array underlying the sector array.
"""
data(sa::AbstractSectorArray) = sa.data

Base.size(sa::AbstractSectorArray) = map(length, axes(sa))

Base.@propagate_inbounds function Base.getindex(
        A::AbstractSectorArray{T, N},
        I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    return @inbounds data(A)[I...]
end
Base.@propagate_inbounds function Base.setindex!(
        A::AbstractSectorArray{T, N},
        v,
        I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds data(A)[I...] = v
    return A
end

# ========================  Shared utilities  ========================

function require_unique_fusion(A)
    return TKS.FusionStyle(sectortype(A)) === TKS.UniqueFusion() ||
        error("not implemented for non-abelian tensors")
end

# ========================  scale! / zero!  ========================

function scale!(a::AbstractSectorArray, β::Number)
    scale!(data(a), β)
    return a
end

function FI.zero!(a::AbstractSectorArray)
    FI.zero!(data(a))
    return a
end
