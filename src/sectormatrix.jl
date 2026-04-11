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

# ---- undef constructors ----

# Innermost: fully parameterized, takes AbstractUnitRange axes.
function SectorMatrix{T, D, S}(
        ::UndefInitializer, sector::S, r1::AbstractUnitRange, r2::AbstractUnitRange
    ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
    return SectorMatrix{T, D, S}(sector, similar(D, (r1, r2)))
end

# Convenience: default D = Matrix{T}.
function SectorMatrix{T}(
        ::UndefInitializer, sector::S, r1::AbstractUnitRange, r2::AbstractUnitRange
    ) where {T, S <: SectorRange}
    return SectorMatrix{T, Matrix{T}, S}(undef, sector, r1, r2)
end

# Int convenience: maps to Base.OneTo.
function SectorMatrix{T}(
        ::UndefInitializer, sector::SectorRange, m::Int, n::Int
    ) where {T}
    return SectorMatrix{T}(undef, sector, Base.OneTo(m), Base.OneTo(n))
end

# ---- accessors ----

# Primitive accessors: sector(sm) and data(sm).
# sector() returns the structural delta factor (SectorIdentity), not the stored SectorRange.
# Access the stored SectorRange via sm.sector or sectoraxes(sm)[1].
sector(sm::SectorMatrix) = SectorIdentity{eltype(sm)}(sm.sector)
dataaxes(sm::SectorMatrix) = axes(data(sm))

# Derived accessors: sectoraxes and axes are written in terms of sector and data.
sectoraxes(sm::SectorMatrix) = axes(sector(sm))

sectortype(::Type{<:SectorMatrix{T, D, S}}) where {T, D, S} = S
datatype(::Type{SectorMatrix{T, D, S}}) where {T, D, S} = D

function Base.axes(sm::SectorMatrix)
    return map(SectorOneTo, sectoraxes(sm), dataaxes(sm))
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
    return SectorMatrix{T₁, D, S}(x.sector, convert(D, data(x)))
end

function Base.similar(sm::SectorMatrix{<:Any, <:Any, S}, ::Type{T}) where {T, S}
    new_data = similar(data(sm), T)
    D = typeof(new_data)
    return SectorMatrix{T, D, S}(sm.sector, new_data)
end

function KroneckerArrays.:(⊗)(s::SectorIdentity, data::AbstractMatrix)
    return SectorMatrix(s.sector, data)
end
