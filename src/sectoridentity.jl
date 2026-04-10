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
    return (SectorRange(A.label), dual(SectorRange(A.label)))
end

labels(x::SectorIdentity) = (x.label, x.label)
label(x::SectorIdentity) = x.label
sector_type(::Type{<:SectorIdentity{T, I}}) where {T, I} = SectorRange{I}

function Base.permutedims(a::SectorIdentity, perm)
    perm == ntuple(identity, ndims(a)) && return a
    return SectorIdentity{eltype(a)}(dual(a.label))
end
