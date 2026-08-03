"""
    AbelianSectorArray{T,S,N,A,NC,ND} <: AbstractSectorArray{T, S, N}

Unfused N-D data tensor for abelian symmetries. Stores a dense data array plus an
[`AbelianSectorDelta`](@ref) carrying one `SectorRange` per axis with a codomain/domain split
(`NC` codomain legs, `ND` domain legs, `NC + ND == N`). Implements the Wigner-Eckart
decomposition: the full tensor is the Kronecker product of the structural delta with the data
array (reduced matrix elements). The all-codomain case (`NC == N`) is the block an
`AbelianGradedArray` yields.
"""
struct AbelianSectorArray{T, S <: SectorRange, N, A <: AbstractArray{T, N}, NC, ND} <:
    AbstractSectorArray{T, S, N}
    data::A
    delta::AbelianSectorDelta{T, S, N, NC, ND}
end

# Constructors

# Fully-parameterized undef constructor: accepts SectorOneTo axes, splitting them into the
# first `NC` codomain legs and remaining `ND` domain legs.
function AbelianSectorArray{T, S, N, A, NC, ND}(
        ::UndefInitializer, axs::NTuple{N, SectorOneTo{S}}
    ) where {T, S <: SectorRange, N, A <: AbstractArray{T, N}, NC, ND}
    sects = sector.(axs)
    delta = AbelianSectorDelta{T, S, N, NC, ND}(
        ntuple(i -> sects[i], Val(NC)), ntuple(i -> sects[NC + i], Val(ND))
    )
    return AbelianSectorArray{T, S, N, A, NC, ND}(similar(A, data.(axs)), delta)
end
# Omitting the split defaults to all-codomain (`NC == N`, `ND == 0`).
function AbelianSectorArray{T, S, N, A}(
        ::UndefInitializer, axs::NTuple{N, SectorOneTo{S}}
    ) where {T, S <: SectorRange, N, A <: AbstractArray{T, N}}
    return AbelianSectorArray{T, S, N, A, N, 0}(undef, axs)
end

# Convenience: infer A = Array{T,N} and S from the axes, all-codomain. Requires at least one
# axis: the sector type of a rank-0 array cannot be inferred from empty axes, so a rank-0 array
# is built through the fully-parameterized constructor with an explicit `S`.
function AbelianSectorArray{T}(
        ::UndefInitializer, axs::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    N = length(axs)
    return AbelianSectorArray{T, sectortype(eltype(axs)), N, Array{T, N}, N, 0}(undef, axs)
end

# Construct from a flat sector tuple (all-codomain), the `AbelianGradedArray` block default.
function AbelianSectorArray(
        data::AbstractArray{T, N}, sectors::NTuple{N, S}
    ) where {T, S <: SectorRange, N}
    return AbelianSectorArray(data, AbelianSectorDelta{T, S, N, N, 0}(sectors, ()))
end
# Typed flat-tuple constructor: split the flat sectors at `NC` (covers rank-0, whose empty
# tuple carries no `S`, through the type parameter).
function AbelianSectorArray{T, S, N, A, NC, ND}(
        data::A, sectors::NTuple{N, S}
    ) where {T, S <: SectorRange, N, A <: AbstractArray{T, N}, NC, ND}
    delta = AbelianSectorDelta{T, S, N, NC, ND}(
        ntuple(i -> sectors[i], Val(NC)), ntuple(i -> sectors[NC + i], Val(ND))
    )
    return AbelianSectorArray{T, S, N, A, NC, ND}(data, delta)
end
# Omitting the split defaults to all-codomain.
function AbelianSectorArray{T, S, N, A}(
        data::A, sectors::NTuple{N, S}
    ) where {T, S <: SectorRange, N, A <: AbstractArray{T, N}}
    return AbelianSectorArray{T, S, N, A, N, 0}(data, sectors)
end

# Construct from an AbelianSectorDelta whose eltype differs from the data (e.g. `real`/`imag`
# or `convert` changing the eltype); the split is preserved. The eltype-matching case hits the
# default constructor, which stores the delta directly so the Kronecker round-trip is `===`.
function AbelianSectorArray(
        data::AbstractArray{T, N}, delta::AbelianSectorDelta{<:Any, S, N, NC, ND}
    ) where {T, S, N, NC, ND}
    return AbelianSectorArray{T, S, N, typeof(data), NC, ND}(
        data,
        AbelianSectorDelta{T, S, N, NC, ND}(delta.sectors_codomain, delta.sectors_domain)
    )
end

const AbelianSectorVector{T, S <: SectorRange, A <: AbstractVector{T}} =
    AbelianSectorArray{T, S, 1, A, NC, ND} where {NC, ND}
const AbelianSectorMatrix{T, S <: SectorRange, A <: AbstractMatrix{T}} =
    AbelianSectorArray{T, S, 2, A, NC, ND} where {NC, ND}

# Accessors

# Kronecker factor decomposition: AbelianSectorArray = sector ⊗ data. `sector` returns the
# stored delta directly, so `sector_kron(sector(a), data(a)) === a`.
sector(sa::AbelianSectorArray) = sa.delta

datatype(::Type{<:AbelianSectorArray{T, S, N, A, NC, ND}}) where {T, S, N, A, NC, ND} = A

Base.copy(A::AbelianSectorArray) = AbelianSectorArray(copy(data(A)), sector(A))

# similar for AbelianSectorArray with SectorOneTo axes.
# Delegates to similar on the data array for the data dimensions.
function Base.similar(
        ::AbelianSectorArray,
        ::Type{T},
        axes::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    return AbelianSectorArray{T}(undef, axes)
end

function Base.convert(
        ::Type{AbelianSectorArray{T₁, S, N, A, NC, ND}},
        x::AbelianSectorArray{T₂, S, N, B, NC, ND}
    )::AbelianSectorArray{T₁, S, N, A, NC, ND} where {T₁, T₂, S, N, A, B, NC, ND}
    A === B && return x
    return AbelianSectorArray(convert(A, data(x)), sector(x))
end
# Omitting the split takes it from the source, so a `convert` never silently changes the split.
function Base.convert(
        ::Type{AbelianSectorArray{T₁, S, N, A}},
        x::AbelianSectorArray{T₂, S, N, B, NC, ND}
    ) where {T₁, T₂, S, N, A, B, NC, ND}
    return convert(AbelianSectorArray{T₁, S, N, A, NC, ND}, x)
end

# ========================  permutedims  ========================

function Base.permutedims(x::AbelianSectorArray, perm)
    new_sector = permutedims(sector(x), perm)
    y = AbelianSectorArray(similar(data(x), size(x)[collect(perm)]), new_sector)
    return permutedims!(y, x, perm)
end
function Base.permutedims!(y::AbelianSectorArray, x::AbelianSectorArray, perm)
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ========================  mul!  ========================

# TODO: Define this as part of:
# `check_input(::typeof(mul!), ::AbelianSectorMatrix, ::AbelianSectorMatrix, ::AbelianSectorMatrix)`
function check_mul_axes(
        c::AbelianSectorMatrix,
        a::AbelianSectorMatrix,
        b::AbelianSectorMatrix
    )
    sectoraxes(a, 2) == dual(sectoraxes(b, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    sectoraxes(c, 1) == sectoraxes(a, 1) || throw(DimensionMismatch())
    sectoraxes(c, 2) == sectoraxes(b, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        c::AbelianSectorMatrix, a::AbelianSectorMatrix, b::AbelianSectorMatrix, α::Number,
        β::Number
    )
    check_mul_axes(c, a, b)
    mul!(data(c), data(a), data(b), α, β)
    return c
end

# ========================  twist!  ========================

function twist!(a::AbelianSectorArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    phase = mapreduce(i -> twist(sectoraxes(a, i)), *, dims; init = 1)
    isone(phase) || (data(a) .*= phase)
    return a
end

# ========================  Other  ========================

function sector_kron(
        s::AbelianSectorDelta{<:Any, <:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return AbelianSectorArray(data, s)
end
