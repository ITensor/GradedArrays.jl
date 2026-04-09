"""
    AbelianSectorArray{T,N,A,I} <: AbstractSectorArray{T, N}

Unfused N-D data tensor for abelian symmetries. Stores one sector label and dual
flag per axis, plus a dense data array. Implements the Wigner-Eckart decomposition:
the full tensor is the Kronecker product of an [`AbelianSectorDelta`](@ref) (structural)
with the data array (reduced matrix elements).
"""
struct AbelianSectorArray{T, N, A <: AbstractArray{T, N}, I <: TKS.Sector} <:
    AbstractSectorArray{T, N}
    labels::NTuple{N, I}
    isduals::NTuple{N, Bool}
    data::A
end

# Constructors
function AbelianSectorArray{T}(
        ::UndefInitializer,
        labels::NTuple{N, I},
        isduals::NTuple{N, Bool},
        dims::NTuple{N, Int}
    ) where {T, N, I <: TKS.Sector}
    data = Array{T, N}(undef, dims)
    return AbelianSectorArray{T, N, Array{T, N}, I}(labels, isduals, data)
end

# Convenience: construct from SectorRange tuples (backward compat bridge)
function AbelianSectorArray(
        sranges::NTuple{N, SectorRange},
        data::AbstractArray{T, N}
    ) where {T, N}
    ls = map(label, sranges)
    ds = map(GradedArrays.isdual, sranges)
    return AbelianSectorArray(ls, ds, data)
end

# Construct from AbelianSectorDelta (inverse of sector/data decomposition)
function AbelianSectorArray(
        delta::AbelianSectorDelta{<:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return AbelianSectorArray(delta.labels, delta.isduals, data)
end

const AbelianSectorMatrix{T, A <: AbstractMatrix{T}, I <: TKS.Sector} =
    AbelianSectorArray{T, 2, A, I}

# Primitive accessors
# -------------------
labels(sa::AbelianSectorArray) = sa.labels
label(sa::AbelianSectorArray, d::Int) = sa.labels[d]

# Derived accessors
# -----------------
function sectoraxes(sa::AbelianSectorArray)
    return ntuple(d -> SectorRange(sa.labels[d], sa.isduals[d]), Val(ndims(sa)))
end
function sector_multiplicities(sa::AbelianSectorArray)
    return ntuple(
        d -> div(size(data(sa), d), quantum_dimension(sectoraxes(sa, d))), Val(ndims(sa))
    )
end

# Kronecker factor decomposition: AbelianSectorArray = sector ⊗ data
sector(sa::AbelianSectorArray) = AbelianSectorDelta{eltype(sa)}(sa.labels, sa.isduals)
dataaxes(sa::AbelianSectorArray) = axes(data(sa))

sector_type(::Type{<:AbelianSectorArray{T, N, A, I}}) where {T, N, A, I} = SectorRange{I}
datatype(::Type{AbelianSectorArray{T, N, A, I}}) where {T, N, A, I} = A

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
    sects = map(sectoraxes1, axes)
    data_ax = map(dataaxes1, axes)
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
        ::Type{AbelianSectorArray{T₁, N, A, I}},
        x::AbelianSectorArray{T₂, N, B, I}
    )::AbelianSectorArray{T₁, N, A, I} where {T₁, T₂, N, A, B, I}
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

function TensorAlgebra.add!(
        dest::AbstractArray,
        src::AbelianSectorArray,
        α::Number,
        β::Number
    )
    TensorAlgebra.add!(dest, data(src), α, β)
    return dest
end

function TensorAlgebra.add!(
        dest::AbelianSectorArray,
        src::AbelianSectorArray,
        α::Number,
        β::Number
    )
    size(dest) == size(src) || throw(DimensionMismatch())
    TensorAlgebra.add!(data(dest), data(src), α, β)
    return dest
end
