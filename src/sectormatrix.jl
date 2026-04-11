"""
    SectorMatrix{T,D<:AbstractMatrix{T},S<:SectorRange} <: AbstractSectorArray{T, 2}

Fused 2D data matrix for a single coupled sector. One block of a
[`FusedGradedMatrix`](@ref). In the representation-theoretic sense, this is an
element of Hom_G(V_c, W_c) for coupled sector c — the reduced matrix element
(degeneracy/multiplicity tensor) after Schur's lemma has factored out the
structural part ([`SectorIdentity`](@ref)).

The codomain (row) axis is non-dual; the domain (column) axis is dual.
The stored `SectorRange` is always non-dual (codomain convention).
"""
struct SectorMatrix{T, D <: AbstractMatrix{T}, S <: SectorRange} <:
    AbstractSectorArray{T, 2}
    sector::S
    data::D
end

# ---- accessors ----

function sectoraxes(sm::SectorMatrix)
    return (sm.sector, dual(sm.sector))
end

# Kronecker factor decomposition: SectorMatrix = sector ⊗ data
# sector() returns the structural delta factor (SectorIdentity), not the stored SectorRange.
# Access the stored SectorRange via sm.sector or sectoraxes(sm)[1].
sector(sm::SectorMatrix) = SectorIdentity{eltype(sm)}(sm.sector)
dataaxes(sm::SectorMatrix) = axes(data(sm))

sectortype(::Type{<:SectorMatrix{T, D, S}}) where {T, D, S} = S
datatype(::Type{SectorMatrix{T, D, S}}) where {T, D, S} = D

function Base.axes(sm::SectorMatrix)
    return (
        SectorOneTo(sm.sector, size(data(sm), 1)),
        SectorOneTo(dual(sm.sector), size(data(sm), 2)),
    )
end

Base.copy(sm::SectorMatrix) = SectorMatrix(sm.sector, copy(data(sm)))

function Base.fill!(sm::SectorMatrix, v)
    fill!(data(sm), v)
    return sm
end

function Base.convert(
        ::Type{SectorMatrix{T₁, D, S}},
        x::SectorMatrix{T₂, E, S}
    )::SectorMatrix{T₁, D, S} where {T₁, T₂, D, E, S}
    D === E && return x
    return SectorMatrix(x.sector, convert(D, data(x)))
end

function Base.similar(sm::SectorMatrix, ::Type{T}) where {T}
    return SectorMatrix(sm.sector, similar(data(sm), T))
end

function KroneckerArrays.:(⊗)(s::SectorIdentity, data::AbstractMatrix)
    return SectorMatrix(s.sector, data)
end
