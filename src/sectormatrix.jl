# ========================  SectorIdentity  ========================

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

# ========================  SectorMatrix  ========================

"""
    SectorMatrix{T,D<:AbstractMatrix{T},I<:TKS.Sector} <: AbstractSectorArray{T, 2}

Fused 2D data matrix for a single coupled sector. One block of a
[`FusedGradedMatrix`](@ref). In the representation-theoretic sense, this is an
element of Hom_G(V_c, W_c) for coupled sector c — the reduced matrix element
(degeneracy/multiplicity tensor) after Schur's lemma has factored out the
structural part ([`SectorIdentity`](@ref)).

The codomain (row) axis is non-dual; the domain (column) axis is dual.
"""
struct SectorMatrix{T, D <: AbstractMatrix{T}, I <: TKS.Sector} <: AbstractSectorArray{T, 2}
    label::I
    data::D
end

# Convenience: construct from SectorRange
function SectorMatrix(sr::SectorRange, data::AbstractMatrix)
    return SectorMatrix(label(sr), data)
end

# ---- accessors ----

labels(sm::SectorMatrix) = (sm.label, sm.label)
label(sm::SectorMatrix) = sm.label

function sectoraxes(sm::SectorMatrix)
    return (SectorRange(sm.label, false), SectorRange(sm.label, true))
end

sector(sm::SectorMatrix) = SectorIdentity{eltype(sm)}(sm.label)
dataaxes(sm::SectorMatrix) = axes(data(sm))

sector_type(::Type{<:SectorMatrix{T, D, I}}) where {T, D, I} = SectorRange{I}
datatype(::Type{SectorMatrix{T, D, I}}) where {T, D, I} = D

function Base.axes(sm::SectorMatrix)
    d = TKS.dim(sm.label)
    m1 = div(size(data(sm), 1), d)
    m2 = div(size(data(sm), 2), d)
    return (
        sectorrange(SectorRange(sm.label, false), m1),
        sectorrange(SectorRange(sm.label, true), m2),
    )
end

Base.copy(sm::SectorMatrix) = SectorMatrix(sm.label, copy(data(sm)))

function Base.fill!(sm::SectorMatrix, v)
    fill!(data(sm), v)
    return sm
end

function Base.convert(
        ::Type{SectorMatrix{T₁, D, I}},
        x::SectorMatrix{T₂, E, I}
    )::SectorMatrix{T₁, D, I} where {T₁, T₂, D, E, I}
    D === E && return x
    return SectorMatrix(x.label, convert(D, data(x)))
end
