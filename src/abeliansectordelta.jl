"""
    AbelianSectorDelta{T,N,I<:TKS.Sector} <: AbstractSectorDelta{T, N}

Unfused N-D structural tensor for abelian symmetries. Stores one sector label and
dual flag per axis. For abelian symmetries, every element equals `one(T)` (the
Kronecker delta selection rule).
"""
struct AbelianSectorDelta{T, N, I <: TKS.Sector} <: AbstractSectorDelta{T, N}
    labels::NTuple{N, I}
    isduals::NTuple{N, Bool}
end
function AbelianSectorDelta{T}(
        labels::NTuple{N, I}, isduals::NTuple{N, Bool}
    ) where {T, N, I <: TKS.Sector}
    return AbelianSectorDelta{T, N, I}(labels, isduals)
end

# Convenience: construct from SectorRange tuples (backward compat bridge)
function AbelianSectorDelta{T}(sranges::NTuple{N, SectorRange}) where {T, N}
    ls = map(label, sranges)
    ds = map(GradedArrays.isdual, sranges)
    return AbelianSectorDelta{T}(ls, ds)
end

# ========================  AbstractArray interface  ========================

Base.@propagate_inbounds function Base.getindex(
        A::AbelianSectorDelta{T, N},
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return one(T)
end

function Base.axes(A::AbelianSectorDelta)
    return ntuple(d -> SectorRange(A.labels[d], A.isduals[d]), Val(ndims(A)))
end

function Base.similar(
        ::AbelianSectorDelta,
        ::Type{T},
        sranges::Tuple{SectorRange, Vararg{SectorRange}}
    ) where {T}
    return AbelianSectorDelta{T}(sranges)
end

function Base.similar(
        ::Type{<:AbstractArray{T}},
        sranges::Tuple{SectorRange, Vararg{SectorRange}}
    ) where {T}
    return AbelianSectorDelta{T}(sranges)
end

# ========================  Primitive accessors  ========================

labels(x::AbelianSectorDelta) = x.labels
label(x::AbelianSectorDelta, d::Int) = x.labels[d]

# ========================  Derived accessors  ========================

isdual(x, d::Int) = isdual(axes(x, d))
sectoraxes(x, d::Int) = sectoraxes(x)[d]
sector_type(::Type{<:AbelianSectorDelta{T, N, I}}) where {T, N, I} = SectorRange{I}

# ========================  permutedims  ========================

function Base.permutedims(x::AbelianSectorDelta, perm)
    new_labels = ntuple(n -> label(x, perm[n]), Val(ndims(x)))
    new_isdual = ntuple(n -> isdual(x, perm[n]), Val(ndims(x)))
    return AbelianSectorDelta{eltype(x)}(new_labels, new_isdual)
end
function FI.permuteddims(x::AbelianSectorDelta, perm)
    return permutedims(x, perm)
end

# ========================  adjoint / broadcasting  ========================

function Base.copy(A::Adjoint{T, <:AbelianSectorDelta{T, 2}}) where {T}
    return AbelianSectorDelta{T}(reverse(dual.(axes(adjoint(A)))))
end
function LinearAlgebra.adjoint!(
        A::AbelianSectorDelta{T, 2}, B::AbelianSectorDelta{T, 2}
    ) where {T}
    reverse(dual.(axes(B))) == axes(A) || throw(DimensionMismatch())
    return A
end

# ========================  multiplication  ========================

function Base.:(*)(
        a::AbelianSectorDelta{T₁, 2},
        b::AbelianSectorDelta{T₂, 2}
    ) where {T₁, T₂}
    axes(a, 2) == dual(axes(b, 1)) ||
        throw(DimensionMismatch("$(axes(a, 2)) != dual($(axes(b, 1))))"))
    T = Base.promote_type(T₁, T₂)
    return AbelianSectorDelta{T}((axes(a, 1), axes(b, 2)))
end

# ========================  Fermionic specializations  ========================

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

function fermion_permutation_phase(
        x::AbstractSectorDelta{<:Any, N},
        perm::NTuple{N, Int}
    ) where {N}
    require_unique_fusion(x)
    BS = TKS.BraidingStyle(sector_type(x))
    BS isa TKS.Bosonic && return true
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"

    mask = map(fermionparity, axes(x))
    return masked_inversion_parity(mask, perm)
end

function fermion_contraction_phase(
        x::AbstractSectorDelta{<:Any, N},
        length_codomain::Int
    ) where {N}
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
