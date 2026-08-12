"""
    FusedSectorMatrix{T,S<:SectorRange,D<:AbstractMatrix{T}} <: AbstractSectorArray{T, S, 2}

Fused 2D data matrix for a single coupled sector. One block of a
[`FusedGradedMatrix`](@ref). In the representation-theoretic sense, this is an
element of Hom_G(V_c, W_c) for coupled sector c — the reduced matrix element
(degeneracy/multiplicity tensor) after Schur's lemma has factored out the
structural part ([`SectorIdentity`](@ref)).

The codomain (row) axis is non-dual; the domain (column) axis is dual.
The stored `SectorRange` is always non-dual (codomain convention).
"""
struct FusedSectorMatrix{T, S <: SectorRange, D <: AbstractMatrix{T}} <:
    AbstractSectorArray{T, S, 2}
    data::D
    sector::S
end

# ---- undef constructors ----

# Innermost: fully parameterized, takes AbstractUnitRange axes.
function FusedSectorMatrix{T, S, D}(
        ::UndefInitializer, sector::S, r1::AbstractUnitRange, r2::AbstractUnitRange
    ) where {T, S <: SectorRange, D <: AbstractMatrix{T}}
    return FusedSectorMatrix{T, S, D}(similar(D, (r1, r2)), sector)
end

# Convenience: default D = Matrix{T}.
function FusedSectorMatrix{T}(
        ::UndefInitializer, sector::S, r1::AbstractUnitRange, r2::AbstractUnitRange
    ) where {T, S <: SectorRange}
    return FusedSectorMatrix{T, S, Matrix{T}}(undef, sector, r1, r2)
end

# Int convenience: maps to Base.OneTo.
function FusedSectorMatrix{T}(
        ::UndefInitializer, sector::SectorRange, m::Int, n::Int
    ) where {T}
    return FusedSectorMatrix{T}(undef, sector, Base.OneTo(m), Base.OneTo(n))
end

# ---- accessors ----

# Primitive accessor: sector(sm) returns the structural delta factor (SectorIdentity), not the
# stored SectorRange. Access the stored SectorRange via sm.sector or sectoraxes(sm)[1]. sectoraxes,
# dataaxes, and axes are derived generically on AbstractSectorArray from sector and data.
sector(sm::FusedSectorMatrix) = SectorIdentity{eltype(sm)}(sm.sector)

datatype(::Type{FusedSectorMatrix{T, S, D}}) where {T, S, D} = D

Base.copy(sm::FusedSectorMatrix) = FusedSectorMatrix(copy(data(sm)), sm.sector)

function Base.convert(
        ::Type{FusedSectorMatrix{T₁, S, D}},
        x::FusedSectorMatrix{T₂, S, E}
    )::FusedSectorMatrix{T₁, S, D} where {T₁, T₂, S, D, E}
    D === E && return x
    return FusedSectorMatrix{T₁, S, D}(convert(D, data(x)), x.sector)
end

function Base.similar(sm::FusedSectorMatrix{<:Any, S, <:Any}, ::Type{T}) where {T, S}
    new_data = similar(data(sm), T)
    D = typeof(new_data)
    return FusedSectorMatrix{T, S, D}(new_data, sm.sector)
end

function sector_kron(s::SectorIdentity, data::AbstractMatrix)
    return FusedSectorMatrix(data, s.sector)
end

# ---- matrix operations ----

# A block is the tensor product of its structural factor `sector(a)` (a `SectorIdentity`) and its
# reduced data `data(a)`, so the trace factorizes: the sector's quantum dimension (the structural
# trace) times the trace of the reduced data.
function LinearAlgebra.tr(a::FusedSectorMatrix)
    return LinearAlgebra.tr(sector(a)) * LinearAlgebra.tr(data(a))
end

# The stored element count factorizes the same way: the structural factor contributes its quantum
# dimension (the length of its diagonal), the reduced data its full size. Abelian sectors have quantum
# dimension 1, so this is just `length(data(a))`.
function SparseArraysBase.storedlength(a::FusedSectorMatrix)
    return storedlength(sector(a)) * length(data(a))
end
