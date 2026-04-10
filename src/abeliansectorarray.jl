"""
    AbelianSectorArray{T,N,A,S} <: AbstractSectorArray{T, N}

Unfused N-D data tensor for abelian symmetries. Stores one `SectorRange` per axis,
plus a dense data array. Implements the Wigner-Eckart decomposition:
the full tensor is the Kronecker product of an [`AbelianSectorDelta`](@ref) (structural)
with the data array (reduced matrix elements).
"""
struct AbelianSectorArray{T, N, A <: AbstractArray{T, N}, S <: SectorRange} <:
    AbstractSectorArray{T, N}
    sectors::NTuple{N, S}
    data::A
end

# Constructors
function AbelianSectorArray{T}(
        ::UndefInitializer,
        sectors::NTuple{N, S},
        dims::NTuple{N, Int}
    ) where {T, N, S <: SectorRange}
    data = Array{T, N}(undef, dims)
    return AbelianSectorArray{T, N, Array{T, N}, S}(sectors, data)
end

# Construct from AbelianSectorDelta (inverse of sector/data decomposition)
function AbelianSectorArray(
        delta::AbelianSectorDelta{<:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return AbelianSectorArray(delta.sectors, data)
end

const AbelianSectorMatrix{T, A <: AbstractMatrix{T}, S <: SectorRange} =
    AbelianSectorArray{T, 2, A, S}

# Accessors
sectoraxes(sa::AbelianSectorArray) = sa.sectors
function sector_multiplicities(sa::AbelianSectorArray)
    return ntuple(
        d -> div(size(data(sa), d), quantum_dimension(sectoraxes(sa, d))), Val(ndims(sa))
    )
end

# Kronecker factor decomposition: AbelianSectorArray = sector ⊗ data
sector(sa::AbelianSectorArray) = AbelianSectorDelta{eltype(sa)}(sa.sectors)
dataaxes(sa::AbelianSectorArray) = axes(data(sa))

sector_type(::Type{<:AbelianSectorArray{T, N, A, S}}) where {T, N, A, S} = S
datatype(::Type{AbelianSectorArray{T, N, A, S}}) where {T, N, A, S} = A

# AbstractArray interface
# -----------------------
function Base.axes(sa::AbelianSectorArray)
    mults = sector_multiplicities(sa)
    return ntuple(d -> sectorrange(sectoraxes(sa, d), mults[d]), Val(ndims(sa)))
end

Base.copy(A::AbelianSectorArray) = AbelianSectorArray(sector(A), copy(data(A)))

# similar for AbelianSectorArray with SectorOneTo axes.
# Delegates to similar on the data array for the data dimensions.
function Base.similar(
        a::AbelianSectorArray,
        ::Type{T},
        axes::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    sects = sector.(axes)
    data_ax = data.(axes)
    return AbelianSectorArray(sects, similar(data(a), T, data_ax))
end

function Base.fill!(A::AbelianSectorArray, v)
    if iszero(v)
        return FI.zero!(A)
    end
    require_unique_fusion(A)
    fill!(data(A), v)
    return A
end

function Base.convert(
        ::Type{AbelianSectorArray{T₁, N, A, S}},
        x::AbelianSectorArray{T₂, N, B, S}
    )::AbelianSectorArray{T₁, N, A, S} where {T₁, T₂, N, A, B, S}
    A === B && return x
    return AbelianSectorArray(sector(x), convert(A, data(x)))
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
function FI.permuteddims(x::AbelianSectorArray, perm)
    return AbelianSectorArray(permutedims(sector(x), perm), FI.permuteddims(data(x), perm))
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

# ========================  Other  ========================

function KroneckerArrays.:(⊗)(
        A::AbelianSectorDelta{<:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return AbelianSectorArray(A, data)
end
