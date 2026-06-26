"""
    AbelianSectorDelta{T,N,S<:SectorRange} <: AbstractSectorDelta{T, N}

Unfused N-D structural tensor for abelian symmetries. Stores one `SectorRange` per axis.
For abelian symmetries, every element equals `one(T)` (the Kronecker delta selection rule).
"""
struct AbelianSectorDelta{T, N, S <: SectorRange} <: AbstractSectorDelta{T, N}
    sectors::NTuple{N, S}
end
function AbelianSectorDelta{T}(
        sectors::NTuple{N, S}
    ) where {T, N, S <: SectorRange}
    return AbelianSectorDelta{T, N, S}(sectors)
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

Base.axes(A::AbelianSectorDelta) = A.sectors

# ========================  Accessors  ========================

isdual(x, d::Int) = isdual(axes(x, d))
sectoraxes(x, d::Int) = sectoraxes(x)[d]
sectortype(::Type{<:AbelianSectorDelta{T, N, S}}) where {T, N, S} = S

# ========================  conj  ========================

# Structural part of the ket->bra involution: dualize every axis. The data-side
# conjugation and fermionic reversal phase live in `Base.conj(::AbelianSectorArray)`;
# here only the selection rule (which lives in the axes) is conjugated.
function Base.conj(x::AbelianSectorDelta{T}) where {T}
    return AbelianSectorDelta{T}(map(conj, x.sectors))
end

# ========================  permutedims  ========================

function Base.permutedims(x::AbelianSectorDelta, perm)
    new_sectors = ntuple(n -> x.sectors[perm[n]], Val(ndims(x)))
    return AbelianSectorDelta{eltype(x)}(new_sectors)
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
    BS = TKS.BraidingStyle(sectortype(x))
    BS isa TKS.Bosonic && return true
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"

    mask = map(fermionparity, axes(x))
    return masked_inversion_parity(mask, perm)
end

# Fermionic phase for permuting `x` by `perm` under the conjugation flag `op`. `op === conj`
# is the ket->bra involution, which reverses leg order, so it contributes the sign of that
# reversal on top of the permutation's own sign. `op === identity` leaves only the
# permutation sign.
function fermion_permutation_phase(
        op, x::AbstractSectorDelta{<:Any, N}, perm::NTuple{N, Int}
    ) where {N}
    phase = fermion_permutation_phase(x, perm)
    op === conj || return phase
    return phase * fermion_permutation_phase(x, reverse(ntuple(identity, Val(N))))
end
