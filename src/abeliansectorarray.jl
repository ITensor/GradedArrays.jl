"""
    AbelianSectorArray{T,S,N,A} <: AbstractSectorArray{T, S, N}

Unfused N-D data tensor for abelian symmetries. Stores one `SectorRange` per axis,
plus a dense data array. Implements the Wigner-Eckart decomposition:
the full tensor is the Kronecker product of an [`AbelianSectorDelta`](@ref) (structural)
with the data array (reduced matrix elements).
"""
struct AbelianSectorArray{T, S <: SectorRange, N, A <: AbstractArray{T, N}} <:
    AbstractSectorArray{T, S, N}
    sectors::NTuple{N, S}
    data::A
end

# Constructors

# Fully-parameterized undef constructor: accepts SectorOneTo axes.
function AbelianSectorArray{T, S, N, A}(
        ::UndefInitializer, axs::NTuple{N, SectorOneTo{S}}
    ) where {T, S <: SectorRange, N, A <: AbstractArray{T, N}}
    sects = sector.(axs)
    return AbelianSectorArray{T, S, N, A}(sects, similar(A, data.(axs)))
end

# Convenience: infer A = Array{T,N} and S from the axes. Requires at least one axis: the
# sector type of a rank-0 array cannot be inferred from empty axes, so a rank-0 array is
# built through the fully-parameterized constructor with an explicit `S`.
function AbelianSectorArray{T}(
        ::UndefInitializer, axs::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    N = length(axs)
    return AbelianSectorArray{T, sectortype(eltype(axs)), N, Array{T, N}}(undef, axs)
end

# Construct from AbelianSectorDelta (inverse of sector/data decomposition). Take `S` from
# the delta's type rather than inferring it from `delta.sectors`, which is empty (and so
# carries no `S`) for a rank-0 array.
function AbelianSectorArray(
        delta::AbelianSectorDelta{<:Any, S, N},
        data::AbstractArray{T, N}
    ) where {T, S, N}
    return AbelianSectorArray{T, S, N, typeof(data)}(delta.sectors, data)
end
function AbelianSectorArray{T, S, N, A}(
        delta::AbelianSectorDelta{<:Any, S, N},
        data::A
    ) where {T, S <: SectorRange, N, A <: AbstractArray{T, N}}
    return AbelianSectorArray{T, S, N, A}(delta.sectors, data)
end

const AbelianSectorVector{T, S <: SectorRange, A <: AbstractVector{T}} =
    AbelianSectorArray{T, S, 1, A}
const AbelianSectorMatrix{T, S <: SectorRange, A <: AbstractMatrix{T}} =
    AbelianSectorArray{T, S, 2, A}

# Accessors

# Kronecker factor decomposition: AbelianSectorArray = sector ⊗ data.
# Pass `N`/`S` explicitly so the structural factor of a rank-0 array (empty `sectors`,
# which carry no `S`) still resolves its sector type.
function sector(sa::AbelianSectorArray{T, S, N, A}) where {T, S, N, A}
    return AbelianSectorDelta{T, S, N}(sa.sectors)
end
sectoraxes(sa::AbelianSectorArray) = axes(sector(sa))
dataaxes(sa::AbelianSectorArray) = axes(data(sa))

datatype(::Type{AbelianSectorArray{T, S, N, A}}) where {T, S, N, A} = A

# AbstractArray interface
# -----------------------
function Base.axes(sa::AbelianSectorArray)
    return map(SectorOneTo, sectoraxes(sa), dataaxes(sa))
end

Base.copy(A::AbelianSectorArray) = AbelianSectorArray(sector(A), copy(data(A)))

# similar for AbelianSectorArray with SectorOneTo axes.
# Delegates to similar on the data array for the data dimensions.
function Base.similar(
        ::AbelianSectorArray,
        ::Type{T},
        axes::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    return AbelianSectorArray{T}(undef, axes)
end

function Base.fill!(A::AbelianSectorArray, v)
    if iszero(v)
        return zero!(A)
    end
    require_unique_fusion(A)
    fill!(data(A), v)
    return A
end

function Base.convert(
        ::Type{AbelianSectorArray{T₁, S, N, A}},
        x::AbelianSectorArray{T₂, S, N, B}
    )::AbelianSectorArray{T₁, S, N, A} where {T₁, T₂, S, N, A, B}
    A === B && return x
    return AbelianSectorArray{T₁, S, N, A}(sector(x), convert(A, data(x)))
end

# ========================  permutedims  ========================

function Base.permutedims(x::AbelianSectorArray, perm)
    new_sector = permutedims(sector(x), perm)
    y = AbelianSectorArray(new_sector, similar(data(x), size(x)[collect(perm)]))
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

# ========================  conj  ========================

# Op A (the ket->bra involution): conjugate the data, flip the duality of every axis, and
# apply the fermionic phase from reversing the leg order. Routed through the lazy conjugating
# broadcast so there is a single implementation: `conj.` lowers to a `ConjArray` (dualizing
# the axes) and materializes via `bipermutedimsopadd!` with `op = conj`, which carries the
# reversal phase that a bare data conjugation drops. This also overrides Base's
# `conj(::AbstractArray{<:Real}) = A` short-circuit, so a real-eltype sector array still
# dualizes its axes.
Base.conj(x::AbelianSectorArray) = conj.(x)

# ========================  Other  ========================

function sector_kron(
        s::AbelianSectorDelta{<:Any, <:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return AbelianSectorArray(s, data)
end
