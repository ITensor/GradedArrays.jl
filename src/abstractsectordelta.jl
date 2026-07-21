"""
    AbstractSectorDelta{T,S,N} <: AbstractArray{T,N}

Abstract supertype for structural (Kronecker/identity) tensors associated to sector labels.
Concrete subtypes:

  - [`AbelianSectorDelta`](@ref): unfused N-D abelian structural tensor (product of Kronecker deltas)
  - [`SectorIdentity`](@ref): fused 2D structural factor (identity matrix per coupled sector)
"""
abstract type AbstractSectorDelta{T, S, N} <: AbstractArray{T, N} end

sectortype(::Type{<:AbstractSectorDelta{T, S}}) where {T, S} = S

Base.copy(A::AbstractSectorDelta) = A
Base.size(A::AbstractSectorDelta) = length.(axes(A))

# A 2D structural delta on a diagonal (coupled) block is the identity over the sector, so its
# trace is the sector's quantum dimension: the length of the diagonal. Only defined in the
# canonical ordering: a non-dual first axis paired with its dual as the second axis.
function LinearAlgebra.tr(A::AbstractSectorDelta{<:Any, <:Any, 2})
    (!isdual(axes(A, 1)) && axes(A, 1) == conj(axes(A, 2))) || throw(
        ArgumentError(
            "trace requires the canonical dual ordering (non-dual first axis), got $(axes(A, 1)) and $(axes(A, 2))"
        )
    )
    return size(A, 1)
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
        x::AbstractSectorDelta{<:Any, <:Any, N},
        perm::NTuple{N, Int}
    ) where {N}
    BS = TKS.BraidingStyle(sectortype(x))
    BS isa TKS.Bosonic && return true
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"
    # Each leg contributes its fermion parity to the swap sign; this is fusion-independent, so it
    # holds for non-abelian symmetric-fermionic sectors as well as abelian ones.
    mask = map(fermionparity, axes(x))
    return masked_inversion_parity(mask, perm)
end

# Fermionic phase for permuting `x` by `perm` under the conjugation flag `op`. `op === conj`
# is the ket->bra involution, which reverses leg order, so it contributes the sign of that
# reversal on top of the permutation's own sign. `op === identity` leaves only the
# permutation sign.
function fermion_permutation_phase(
        op, x::AbstractSectorDelta{<:Any, <:Any, N}, perm::NTuple{N, Int}
    ) where {N}
    phase = fermion_permutation_phase(x, perm)
    op === conj || return phase
    return phase * fermion_permutation_phase(x, reverse(ntuple(identity, Val(N))))
end

function fermion_bend_phase(x::AbstractSectorDelta, dims::NTuple{M, Int}) where {M}
    BS = TKS.BraidingStyle(sectortype(x))
    BS isa TKS.Bosonic && return 1
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"
    dmask = map(d -> fermionparity(axes(x, d)), dims)
    return masked_inversion_parity(dmask, reverse(ntuple(identity, Val(M))))
end
