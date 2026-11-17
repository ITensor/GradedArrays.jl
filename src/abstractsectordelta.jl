# Supertype for the structural factors of sector arrays, determined by the sector labels.
abstract type AbstractSectorDelta{T, S, N} <: AbstractArray{T, N} end

sectortype(::Type{<:AbstractSectorDelta{T, S}}) where {T, S} = S

Base.copy(A::AbstractSectorDelta) = A
Base.size(A::AbstractSectorDelta) = map(length, axes(A))

# Base's array display scalar-indexes, which a delta only supports under unique fusion.
Base.print_array(io::IO, A::AbstractSectorDelta) = Base.print_array(io, Array(A))
Base.show(io::IO, A::AbstractSectorDelta) = show(io, Array(A))

# Matrix operations are defined only on the matrix storage types and `SectorIdentity`. Elsewhere
# they would fall through to the generic `AbstractArray` methods, which scalar-index into a dense
# result, so they error instead.
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

# For `conj` and a transposing `permutedims`, which would flip the first axis to dual. The fused
# storage types always carry a non-dual first axis.
@noinline function throw_flips_first_axis(op, A)
    return error(
        "`$op` is not supported on `$(nameof(typeof(A)))` because it would flip the first axis to \
        dual; the fused storage types always carry a non-dual first axis. Apply `$op` to the \
        `GradedArray` (or matricize) first."
    )
end

# ========================  Fermionic specializations  ========================

# The parity of the number of inversions of a masked permutation.
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
    # The swap sign depends only on the parity of each leg, so this holds for non-abelian fermionic
    # sectors too.
    mask = map(fermionparity, axes(x))
    return masked_inversion_parity(mask, perm)
end

# Conjugation (ket to bra) reverses the leg order, so it contributes the sign of that reversal on
# top of the permutation's own sign.
function fermion_permutation_phase(
        isconj::Bool, x::AbstractSectorDelta{<:Any, <:Any, N}, perm::NTuple{N, Int}
    ) where {N}
    phase = fermion_permutation_phase(x, perm)
    isconj || return phase
    return phase * fermion_permutation_phase(x, reverse(ntuple(identity, Val(N))))
end

function fermion_bend_phase(x::AbstractSectorDelta, dims::NTuple{M, Int}) where {M}
    BS = TKS.BraidingStyle(sectortype(x))
    BS isa TKS.Bosonic && return 1
    @assert BS isa TKS.Fermionic "Only symmetric braiding is supported"
    dmask = map(d -> fermionparity(axes(x, d)), dims)
    return masked_inversion_parity(dmask, reverse(ntuple(identity, Val(M))))
end
