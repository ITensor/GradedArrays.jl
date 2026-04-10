"""
    SectorIdentity{T,S<:SectorRange} <: AbstractSectorDelta{T, 2}

Fused 2D structural factor for a single coupled sector. By Schur's lemma, the
structural part of each block in the fused (matricized) basis is the identity
matrix for the irrep. Carries no free data — completely determined by the sector.
The codomain axis is non-dual, the domain axis is dual.
"""
struct SectorIdentity{T, S <: SectorRange} <: AbstractSectorDelta{T, 2}
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

sector_type(::Type{<:SectorIdentity{T, S}}) where {T, S} = S

function Base.permutedims(a::SectorIdentity, perm)
    perm == ntuple(identity, ndims(a)) && return a
    return SectorIdentity{eltype(a)}(dual(a.sector))
end
