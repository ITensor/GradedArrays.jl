"""
    SectorOnesVector{T,S<:SectorRange} <: AbstractSectorDelta{T, S, 1}

Fused 1-D structural factor for a single coupled sector: the all-ones vector whose length is the
sector's quantum dimension. It is the diagonal of the [`SectorIdentity`](@ref) that a
`FusedGradedVector` picks out as the diagonal of a `FusedGradedMatrix`, so each reduced value is
repeated once per state of the irrep. Carries no free data — completely determined by the sector.
The axis is non-dual.
"""
struct SectorOnesVector{T, S <: SectorRange} <: AbstractSectorDelta{T, S, 1}
    sector::S
end
function SectorOnesVector{T}(s::S) where {T, S <: SectorRange}
    return SectorOnesVector{T, S}(s)
end

Base.@propagate_inbounds function Base.getindex(A::SectorOnesVector{T}, i::Int) where {T}
    @boundscheck checkbounds(A, i)
    return one(T)
end

Base.axes(A::SectorOnesVector) = (A.sector,)

# A single index has only the identity permutation.
Base.permutedims(a::SectorOnesVector, perm) = a
