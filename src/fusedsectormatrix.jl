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
    function FusedSectorMatrix{T, S, D}(
            data::D, sector::S
        ) where {T, S <: SectorRange, D <: AbstractMatrix{T}}
        !isdual(sector) ||
            throw(
            ArgumentError(
                "`FusedSectorMatrix` requires a non-dual sector, got `$sector`"
            )
        )
        return new{T, S, D}(data, sector)
    end
end

# Default the parameters from the data and sector types.
function FusedSectorMatrix(data::D, sector::S) where {S <: SectorRange, D <: AbstractMatrix}
    return FusedSectorMatrix{eltype(D), S, D}(data, sector)
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

Base.conj(a::FusedSectorMatrix) = throw_flips_first_axis(conj, a)

# The stored element count factorizes the same way: the structural factor contributes its quantum
# dimension (the length of its diagonal), the reduced data its full size. Abelian sectors have quantum
# dimension 1, so this is just `length(data(a))`.
function storedlength(a::FusedSectorMatrix)
    return storedlength(sector(a)) * length(data(a))
end

# ---- reductions ----

# `sum` over a graded matrix is restricted to zero-preserving `f` (`f(0) == 0`) for now, so the
# structural zeros contribute nothing and need not be counted. Shared by the `FusedSectorMatrix` and
# `FusedGradedMatrix` `sum` methods.
@noinline function throw_not_zero_preserving_sum(z)
    return error(
        "`sum` over a graded matrix supports only zero-preserving `f` (`f(0) == 0`) for now; \
        got `f(0) = $z`. Materialize with `Array` first."
    )
end

# The dense block is `data(a) ⊗ sector(a)`: the quantum dimension `d` copies of the reduced data on the
# diagonal, with `length - storedlength` structural zeros off it. `sum` (zero-preserving `f` only for
# now) weights the reduced sum by `d` and drops the structural zeros; `maximum`/`minimum` are unchanged
# by the duplication and fold in a single `f(0)` when the block has structural zeros (`d > 1`).
Base.sum(a::FusedSectorMatrix) = sum(identity, a)
function Base.sum(f, a::FusedSectorMatrix)
    z = f(zero(eltype(a)))
    iszero(z) || throw_not_zero_preserving_sum(z)
    return storedlength(sector(a)) * sum(f, data(a))
end
Base.maximum(a::FusedSectorMatrix) = maximum(identity, a)
function Base.maximum(f, a::FusedSectorMatrix)
    m = maximum(f, data(a))
    return length(a) > storedlength(a) ? max(m, f(zero(eltype(a)))) : m
end
Base.minimum(a::FusedSectorMatrix) = minimum(identity, a)
function Base.minimum(f, a::FusedSectorMatrix)
    m = minimum(f, data(a))
    return length(a) > storedlength(a) ? min(m, f(zero(eltype(a)))) : m
end
Base.extrema(a::FusedSectorMatrix) = extrema(identity, a)
Base.extrema(f, a::FusedSectorMatrix) = (minimum(f, a), maximum(f, a))
