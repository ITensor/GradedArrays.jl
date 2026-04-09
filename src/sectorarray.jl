# Array
# -----
"""
    SectorDelta{T,N,I<:TKS.Sector} <: AbstractArray{T, N}

An immutable representation of the structural tensor associated to the representation space
of a number of sectors. For abelian symmetries, this boils down to a scalar which can
always be normalized to 1.

Stores primitive fields: raw sector labels and dual flags. `axes` returns `SectorRange`
values constructed on the fly.
"""
struct SectorDelta{T, N, I <: TKS.Sector} <: AbstractArray{T, N}
    labels::NTuple{N, I}
    isdual::NTuple{N, Bool}
end
function SectorDelta{T}(
        labels::NTuple{N, I}, isdual::NTuple{N, Bool}
    ) where {T, N, I <: TKS.Sector}
    return SectorDelta{T, N, I}(labels, isdual)
end

# Convenience: construct from SectorRange tuples (backward compat bridge)
function SectorDelta{T}(sranges::NTuple{N, SectorRange}) where {T, N}
    ls = map(label, sranges)
    ds = map(GradedArrays.isdual, sranges)
    return SectorDelta{T}(ls, ds)
end

function require_unique_fusion(A)
    return TKS.FusionStyle(sector_type(A)) === TKS.UniqueFusion() ||
        error("not implemented for non-abelian tensors")
end

# ========================  AbstractArray interface  ========================

Base.@propagate_inbounds function Base.getindex(
        A::SectorDelta{T, N},
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return one(T)
end

function Base.axes(A::SectorDelta)
    return ntuple(d -> SectorRange(A.labels[d], A.isdual[d]), Val(ndims(A)))
end
Base.size(A::SectorDelta) = length.(axes(A))

function Base.similar(
        ::SectorDelta,
        ::Type{T},
        sranges::Tuple{SectorRange, Vararg{SectorRange}}
    ) where {T}
    return SectorDelta{T}(sranges)
end

function Base.similar(
        ::Type{<:AbstractArray{T}},
        sranges::Tuple{SectorRange, Vararg{SectorRange}}
    ) where {T}
    return SectorDelta{T}(sranges)
end

# ========================  Primitive accessors  ========================

labels(x::SectorDelta) = x.labels
label(x::SectorDelta, d::Int) = x.labels[d]
isdual(x::SectorDelta, d::Int) = x.isdual[d]

# ========================  Derived accessors  ========================

sectoraxes(x, d::Int) = sectoraxes(x)[d]
sector_type(::Type{<:SectorDelta{T, N, I}}) where {T, N, I} = SectorRange{I}

# ========================  permutedims  ========================

function Base.permutedims(x::SectorDelta, perm)
    new_labels = ntuple(n -> label(x, perm[n]), Val(ndims(x)))
    new_isdual = ntuple(n -> isdual(x, perm[n]), Val(ndims(x)))
    return SectorDelta{eltype(x)}(new_labels, new_isdual)
end
function FI.permuteddims(x::SectorDelta, perm)
    return permutedims(x, perm)
end

# ========================  copy / broadcasting  ========================

Base.copy(A::SectorDelta) = A
function Base.copy!(C::SectorDelta, A::SectorDelta)
    axes(C) == axes(A) || throw(DimensionMismatch())
    return C
end
function Base.copyto!(C::SectorDelta, A::SectorDelta)
    axes(C) == axes(A) || throw(DimensionMismatch())
    return C
end
function Base.copy(A::Adjoint{T, <:SectorDelta{T, 2}}) where {T}
    return SectorDelta{T}(reverse(dual.(axes(adjoint(A)))))
end
function LinearAlgebra.adjoint!(
        A::SectorDelta{T, 2}, B::SectorDelta{T, 2}
    ) where {T}
    reverse(dual.(axes(B))) == axes(A) || throw(DimensionMismatch())
    return A
end

# ========================  multiplication  ========================

function Base.:(*)(a::SectorDelta{T₁, 2}, b::SectorDelta{T₂, 2}) where {T₁, T₂}
    axes(a, 2) == dual(axes(b, 1)) ||
        throw(DimensionMismatch("$(axes(a, 2)) != dual($(axes(b, 1))))"))
    T = Base.promote_type(T₁, T₂)
    return SectorDelta{T}((axes(a, 1), axes(b, 2)))
end

"""
    SectorArray(labels, isdual, data) <: AbstractArray

A representation of a general symmetric array as the combination of sector labels,
dual flags, and a data array. This can be thought of as a direct implementation of
the Wigner-Eckart theorem.

Each dimension has:

  - a sector label (`labels`): the raw `TKS.Sector` value
  - a dual flag (`isdual`): whether that dimension is in the dual space
  - a data slice from `data`: the reduced matrix element storage
"""
struct SectorArray{T, N, I <: TKS.Sector, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    labels::NTuple{N, I}
    isdual::NTuple{N, Bool}
    data::A
end

# Constructors
function SectorArray{T}(
        ::UndefInitializer,
        labels::NTuple{N, I},
        isdual::NTuple{N, Bool},
        dims::NTuple{N, Int}
    ) where {T, N, I <: TKS.Sector}
    data = Array{T, N}(undef, dims)
    return SectorArray{T, N, I, Array{T, N}}(labels, isdual, data)
end

# Convenience: construct from SectorRange tuples (backward compat bridge)
function SectorArray(
        sranges::NTuple{N, SectorRange},
        data::AbstractArray{T, N}
    ) where {T, N}
    ls = map(label, sranges)
    ds = map(GradedArrays.isdual, sranges)
    return SectorArray(ls, ds, data)
end

const SectorMatrix{T, I, A <: AbstractMatrix{T}} = SectorArray{T, 2, I, A}

# Primitive accessors
# -------------------
labels(sa::SectorArray) = sa.labels
label(sa::SectorArray, d::Int) = sa.labels[d]
isdual(sa::SectorArray, d::Int) = sa.isdual[d]

# Derived accessors
# -----------------
function sectoraxes(sa::SectorArray)
    return ntuple(d -> SectorRange(label(sa, d), isdual(sa, d)), Val(ndims(sa)))
end
function sector_multiplicities(sa::SectorArray)
    return ntuple(
        d -> div(size(sa.data, d), quantum_dimension(sectoraxes(sa, d))), Val(ndims(sa))
    )
end

# Kronecker factor decomposition: SectorArray = sector ⊗ data
data(sa::SectorArray) = sa.data
sector(sa::SectorArray) = SectorDelta{eltype(sa)}(sa.labels, sa.isdual)
dataaxes(sa::SectorArray) = axes(data(sa))

sector_type(::Type{<:SectorArray{T, N, I}}) where {T, N, I} = SectorRange{I}
datatype(::Type{SectorArray{T, N, I, A}}) where {T, N, I, A} = A

# AbstractArray interface
# -----------------------
Base.size(sa::SectorArray) = size(sa.data)

Base.@propagate_inbounds function Base.getindex(
        A::SectorArray{T, N},
        I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    return @inbounds A.data[I...]
end
Base.@propagate_inbounds function Base.setindex!(
        A::SectorArray{T, N},
        v,
        I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds A.data[I...] = v
    return A
end

Base.copy(A::SectorArray) = SectorArray(A.labels, A.isdual, copy(A.data))

# similar for SectorArray with SectorOneTo axes.
# Delegates to similar on the data array for the data dimensions.
function Base.similar(
        a::SectorArray,
        ::Type{T},
        axes::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    sects = map(sectoraxes1, axes)
    data_ax = map(dataaxes1, axes)
    return SectorArray(sects, similar(data(a), T, data_ax))
end

function FI.zero!(A::SectorArray)
    fill!(A.data, zero(eltype(A)))
    return A
end

function Base.fill!(A::SectorArray, v)
    if iszero(v)
        return FI.zero!(A)
    end
    require_unique_fusion(A)
    fill!(A.data, v)
    return A
end

function Base.convert(
        ::Type{SectorArray{T₁, N, I, A}},
        x::SectorArray{T₂, N, I, B}
    )::SectorArray{T₁, N, I, A} where {T₁, T₂, N, I, A, B}
    A === B && return x
    return SectorArray(x.labels, x.isdual, convert(A, x.data))
end

# Fermionic specializations
# -------------------------
"""
Compute the parity of the number of inversions of a masked permutation
"""
function masked_inversion_parity(mask::NTuple{N, Bool}, perm::NTuple{N, Int}) where {N}
    parity = false
    @inbounds for i in 1:N
        mask[i] || continue
        for j in (i + 1):N
            parity ⊻= mask[j] & (perm[i] > perm[j]) # branchless is important here
        end
    end
    return ifelse(parity, -1, 1)
end

function fermion_permutation_phase(x::SectorDelta{<:Any, N}, perm::NTuple{N, Int}) where {N}
    require_unique_fusion(x)
    BS = TKS.BraidingStyle(sector_type(x))
    BS isa TKS.Bosonic && return true
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"

    mask = map(fermionparity, axes(x))
    return masked_inversion_parity(mask, perm)
end

function fermion_contraction_phase(x::SectorDelta{<:Any, N}, length_codomain::Int) where {N}
    require_unique_fusion(x)
    BS = TKS.BraidingStyle(sector_type(x))
    BS isa TKS.Bosonic && return true
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"
    length_codomain <= ndims(x) ||
        throw(ArgumentError(lazy"Cannot contract more than ndim legs ($N > $(ndims(x))"))

    parity = mapreduce(⊻, enumerate(axes(x))) do (n, ax)
        return n <= length_codomain & isdual(ax) & fermionparity(ax)
    end
    return ifelse(parity, -1, 1)
end

function Base.permutedims(x::SectorArray, perm)
    new_labels = ntuple(n -> label(x, perm[n]), Val(ndims(x)))
    new_isdual = ntuple(n -> isdual(x, perm[n]), Val(ndims(x)))
    y = SectorArray(new_labels, new_isdual, similar(x.data, size(x)[collect(perm)]))
    return permutedims!(y, x, perm)
end
function Base.permutedims!(y::SectorArray, x::SectorArray, perm)
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end
function FI.permuteddims(x::SectorArray, perm)
    new_labels = ntuple(n -> label(x, perm[n]), Val(ndims(x)))
    new_isdual = ntuple(n -> isdual(x, perm[n]), Val(ndims(x)))
    return SectorArray(new_labels, new_isdual, FI.permuteddims(x.data, perm))
end

# TODO: Define this as part of:
# `check_input(::typeof(mul!), ::SectorMatrix, ::SectorMatrix, ::SectorMatrix)`
function check_mul_axes(c::SectorMatrix, a::SectorMatrix, b::SectorMatrix)
    sectoraxes(a, 2) == dual(sectoraxes(b, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    sectoraxes(c, 1) == sectoraxes(a, 1) || throw(DimensionMismatch())
    sectoraxes(c, 2) == sectoraxes(b, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        c::SectorMatrix, a::SectorMatrix, b::SectorMatrix, α::Number, β::Number
    )
    check_mul_axes(c, a, b)
    mul!(c.data, a.data, b.data, α, β)
    return c
end

# Other
# -----
function KroneckerArrays.:(⊗)(A::SectorDelta{T, N}, data::AbstractArray{T, N}) where {T, N}
    return SectorArray(A.labels, A.isdual, data)
end
function KroneckerArrays.:(⊗)(
        A::SectorDelta{T₁, N},
        data::AbstractArray{T₂, N}
    ) where {T₁, T₂, N}
    T = Base.promote_type(*, T₁, T₂)
    return SectorArray(A.labels, A.isdual, collect(T, data))
end

function TensorAlgebra.add!(dest::AbstractArray, src::SectorArray, α::Number, β::Number)
    TensorAlgebra.add!(dest, src.data, α, β)
    return dest
end

function TensorAlgebra.add!(dest::SectorArray, src::SectorArray, α::Number, β::Number)
    size(dest) == size(src) || throw(DimensionMismatch())
    TensorAlgebra.add!(dest.data, src.data, α, β)
    return dest
end
