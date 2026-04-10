"""
    SectorIdentity{T,I<:TKS.Sector} <: AbstractSectorDelta{T, 2}

Fused 2D structural factor for a single coupled sector. By Schur's lemma, the
structural part of each block in the fused (matricized) basis is the identity
matrix for the irrep. Carries no free data — completely determined by the sector
label. The codomain axis is non-dual, the domain axis is dual.
"""
struct SectorIdentity{T, I <: TKS.Sector} <: AbstractSectorDelta{T, 2}
    label::I
end
function SectorIdentity{T}(l::I) where {T, I <: TKS.Sector}
    return SectorIdentity{T, I}(l)
end

# Convenience: construct from SectorRange (extracts label, ignores dual flag)
function SectorIdentity{T}(sr::SectorRange{I}) where {T, I}
    return SectorIdentity{T}(label(sr))
end

Base.@propagate_inbounds function Base.getindex(
        A::SectorIdentity{T}, i::Int, j::Int
    ) where {T}
    @boundscheck checkbounds(A, i, j)
    return ifelse(i == j, one(T), zero(T))
end

function Base.axes(A::SectorIdentity)
    return (SectorRange(A.label, false), dual(SectorRange(A.label, false)))
end

labels(x::SectorIdentity) = (x.label, x.label)
label(x::SectorIdentity) = x.label
sector_type(::Type{<:SectorIdentity{T, I}}) where {T, I} = SectorRange{I}

function sectoraxes(x::SectorIdentity)
    return (SectorRange(x.label, false), SectorRange(x.label, true))
end

# Permuting axes may change dual flags, so delegate to AbelianSectorDelta.
function Base.permutedims(x::SectorIdentity, perm)
    return permutedims(AbelianSectorDelta{eltype(x)}(labels(x), map(isdual, axes(x))), perm)
end
