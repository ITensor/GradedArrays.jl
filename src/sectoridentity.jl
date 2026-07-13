"""
    SectorIdentity{T,S<:SectorRange} <: AbstractSectorDelta{T, S, 2}

Fused 2D structural factor for a single coupled sector. By Schur's lemma, the
structural part of each block in the fused (matricized) basis is the identity
matrix for the irrep. Carries no free data — completely determined by the sector.
The codomain axis is non-dual, the domain axis is dual.
"""
struct SectorIdentity{T, S <: SectorRange} <: AbstractSectorDelta{T, S, 2}
    sector::S
end
function SectorIdentity{T}(s::S) where {T, S <: SectorRange}
    return SectorIdentity{T, S}(s)
end

Base.@propagate_inbounds function Base.getindex(
        A::SectorIdentity{T}, i::Int, j::Int
    ) where {T}
    @boundscheck checkbounds(A, i, j)
    return ifelse(i == j, one(T), zero(T))
end

function Base.axes(A::SectorIdentity)
    return (A.sector, dual(A.sector))
end

# Structural inner product: the identity contracts to its dimension, the quantum dimension.
function LinearAlgebra.dot(a::SectorIdentity, b::SectorIdentity)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return length(a.sector)
end

# `p`-norm: the identity has `length(sector)` unit entries (its diagonal), so `norm^p` counts them.
# The single formula also covers `p == Inf` (`count^0 == 1`, the max entry).
function LinearAlgebra.norm(a::SectorIdentity{T}, p::Real = 2) where {T}
    return convert(real(float(T)), length(a.sector)^(1 / p))
end

function Base.permutedims(a::SectorIdentity, perm)
    perm == ntuple(identity, ndims(a)) && return a
    return SectorIdentity{eltype(a)}(dual(a.sector))
end
