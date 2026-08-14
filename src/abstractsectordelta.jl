"""
    AbstractSectorDelta{T,S,N} <: AbstractArray{T,N}

Abstract supertype for structural (Kronecker/identity) tensors associated to sector labels.
Concrete subtypes:

  - [`UniqueSectorDelta`](@ref): unfused N-D abelian structural tensor (product of Kronecker deltas)
  - [`SectorIdentity`](@ref): fused 2D structural factor (identity matrix per coupled sector)
"""
abstract type AbstractSectorDelta{T, S, N} <: AbstractArray{T, N} end

sectortype(::Type{<:AbstractSectorDelta{T, S}}) where {T, S} = S

Base.copy(A::AbstractSectorDelta) = A
Base.size(A::AbstractSectorDelta) = map(length, axes(A))

# Display through the dense form. Base's array-display machinery scalar-indexes, which a delta only
# supports under unique fusion; it is a small structural factor, so densifying it for display is fine.
Base.print_array(io::IO, A::AbstractSectorDelta) = Base.print_array(io, Array(A))
Base.show(io::IO, A::AbstractSectorDelta) = show(io, Array(A))

# Matrix/linear-algebra operations (`*`, `adjoint`, `transpose`, `tr`, `one!`, factorizations,
# matrix functions) are defined only on the matrix storage types (`FusedGradedMatrix` /
# `FusedSectorMatrix`) and, among the structural deltas, only on the identity factor
# `SectorIdentity`. On the array and general-delta types they would fall through to the generic
# `AbstractArray` methods, which scalar-index into a dense, non-graded result. Error instead; the
# specific methods (the matrix-storage types, `SectorIdentity`) are more specific, so they still
# win. This helper is the shared fallback for every level (defined here, the earliest sector file).
@noinline function _matrix_op_error(op, A)
    return error(
        "`$op` is a matrix operation and is not defined on `$(nameof(typeof(A)))`; matrix \
        operations live on the matrix storage types (`FusedGradedMatrix` / `FusedSectorMatrix`). \
        Matricize first."
    )
end

Base.adjoint(A::AbstractSectorDelta) = _matrix_op_error(adjoint, A)
Base.transpose(A::AbstractSectorDelta) = _matrix_op_error(transpose, A)
Base.:*(A::AbstractSectorDelta, B::AbstractSectorDelta) = _matrix_op_error(*, A)
function LinearAlgebra.tr(A::AbstractSectorDelta{<:Any, <:Any, 2})
    return _matrix_op_error(LinearAlgebra.tr, A)
end
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::AbstractSectorDelta) = _matrix_op_error($f, A)
end

# `conj` and a transposing `permutedims` would flip the first axis to dual. The fused storage types
# (the matrix trio `FusedGradedMatrix` / `FusedSectorMatrix` / `SectorIdentity` and the vector trio
# `FusedGradedVector` / `FusedSectorVector` / `SectorOnesVector`) always carry a non-dual first axis,
# so both are blocked on them. Conjugate or transpose the `GradedArray` (or matricize) instead.
@noinline function throw_flips_first_axis(op, A)
    return error(
        "`$op` is not supported on `$(nameof(typeof(A)))` because it would flip the first axis to \
        dual; the fused storage types always carry a non-dual first axis. Apply `$op` to the \
        `GradedArray` (or matricize) first."
    )
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
