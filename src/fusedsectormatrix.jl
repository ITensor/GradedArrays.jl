# One block of a `FusedGradedMatrix`, for a single coupled sector. By Schur's lemma its structural
# factor is the identity (`SectorIdentity`), and `data` holds the reduced matrix elements.
#
# The codomain (row) axis is non-dual and the domain (column) axis is dual.
struct FusedSectorMatrix{T, S <: Sector, D <: AbstractMatrix{T}} <:
    AbstractSectorArray{T, S, 2}
    data::D
    sector::S
end

# ---- undef constructors ----

function FusedSectorMatrix{T, S, D}(
        ::UndefInitializer, sector::S, r1::AbstractUnitRange, r2::AbstractUnitRange
    ) where {T, S <: Sector, D <: AbstractMatrix{T}}
    return FusedSectorMatrix{T, S, D}(similar(D, (r1, r2)), sector)
end

function FusedSectorMatrix{T}(
        ::UndefInitializer, sector::S, r1::AbstractUnitRange, r2::AbstractUnitRange
    ) where {T, S <: Sector}
    return FusedSectorMatrix{T, S, Matrix{T}}(undef, sector, r1, r2)
end

function FusedSectorMatrix{T}(
        ::UndefInitializer, sector::Sector, m::Int, n::Int
    ) where {T}
    return FusedSectorMatrix{T}(undef, sector, Base.OneTo(m), Base.OneTo(n))
end

# ---- accessors ----

structure(sm::FusedSectorMatrix) = SectorIdentity{eltype(sm)}(sector(sm))
sector(sm::FusedSectorMatrix) = sm.sector

datatype(::Type{FusedSectorMatrix{T, S, D}}) where {T, S, D} = D

Base.copy(sm::FusedSectorMatrix) = FusedSectorMatrix(copy(data(sm)), sector(sm))

function Base.convert(
        ::Type{FusedSectorMatrix{T₁, S, D}},
        x::FusedSectorMatrix{T₂, S, E}
    )::FusedSectorMatrix{T₁, S, D} where {T₁, T₂, S, D, E}
    D === E && return x
    return FusedSectorMatrix{T₁, S, D}(convert(D, data(x)), sector(x))
end

function Base.similar(sm::FusedSectorMatrix{<:Any, S, <:Any}, ::Type{T}) where {T, S}
    new_data = similar(data(sm), T)
    D = typeof(new_data)
    return FusedSectorMatrix{T, S, D}(new_data, sector(sm))
end

function sector_kron(s::SectorIdentity, data::AbstractMatrix)
    return FusedSectorMatrix(data, sector(s))
end

# ---- matrix operations ----

# `tr(I ⊗ A) = tr(I) * tr(A)`, where `tr(I)` is the quantum dimension.
function LinearAlgebra.tr(a::FusedSectorMatrix)
    return LinearAlgebra.tr(structure(a)) * LinearAlgebra.tr(data(a))
end

Base.conj(a::FusedSectorMatrix) = throw_flips_first_axis(conj, a)

function storedlength(a::FusedSectorMatrix)
    return storedlength(structure(a)) * length(data(a))
end

# ---- reductions ----

# Restricting `sum` to zero-preserving `f` means the structural zeros need not be counted.
@noinline function throw_not_zero_preserving_sum(z)
    return error(
        "`sum` over a graded matrix supports only zero-preserving `f` (`f(0) == 0`) for now; \
        got `f(0) = $z`. Materialize with `Array` first."
    )
end

# The dense block holds `d` copies of the reduced data, for quantum dimension `d`, plus structural
# zeros when `d > 1`. So `sum` weights by `d`, and `maximum`/`minimum` fold in `f(0)` when there are
# structural zeros.
Base.sum(a::FusedSectorMatrix) = sum(identity, a)
function Base.sum(f, a::FusedSectorMatrix)
    z = f(zero(eltype(a)))
    iszero(z) || throw_not_zero_preserving_sum(z)
    return storedlength(structure(a)) * sum(f, data(a))
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
