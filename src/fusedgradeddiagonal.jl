# ===========================================================================
#  FusedGradedDiagonal — block-diagonal fused matrix with `Diagonal` blocks
# ===========================================================================

using LinearAlgebra: Diagonal
using MatrixAlgebraKit: MatrixAlgebraKit as MAK

"""
    FusedGradedDiagonal{T,S<:SectorRange,V<:DenseVector{T}} <: AbstractFusedGradedMatrix{T,S}

Square block-diagonal fused matrix whose every coupled-sector block is a `Diagonal`, the diagonal
factor produced by a factorization (SVD singular values, eigenvalues). Analogous to TensorKit's
`DiagonalTensorMap`. Wraps a [`FusedGradedVector`](@ref) of the diagonals; the `Diagonal` blocks are
the lazy `sectordata(d)` view over that vector.
"""
struct FusedGradedDiagonal{T, S <: SectorRange, V <: DenseVector{T}} <:
    AbstractFusedGradedMatrix{T, S}
    diag::FusedGradedVector{T, S, V}
end

"""
    FusedGradedDiagonal(buffer, axis)

Wrap a contiguous `buffer` (shared, not copied) as a `FusedGradedDiagonal` with the given `axis`; the
`Diagonal` blocks are the lazy `sectordata` view over the buffer. The `axis` is fused into canonical
form. To build from per-sector diagonal data instead, use [`fusedgradeddiagonal`](@ref).
"""
function FusedGradedDiagonal(buffer::DenseVector, axis::AbstractGradedOneTo)
    return FusedGradedDiagonal(FusedGradedVector(buffer, axis))
end

function FusedGradedDiagonal{T}(::UndefInitializer, axis::AbstractGradedOneTo) where {T}
    return FusedGradedDiagonal(FusedGradedVector{T}(undef, axis))
end

"""
    fusedgradeddiagonal(sectors .=> data)
    fusedgradeddiagonal(sectordata::Dictionary)

Build a `FusedGradedDiagonal` from the per-sector diagonal data (`sector => data` pairs, any iterator
of pairs, or a `Dictionary` keyed by sector): the pair `sectors[i] => data[i]` gives the diagonal
entries of the block at `sectors[i]`. The axis is derived from the blocks, as for
[`fusedgradedvector`](@ref). To wrap an existing contiguous buffer instead, use
[`FusedGradedDiagonal`](@ref).
"""
fusedgradeddiagonal(sectordata) = FusedGradedDiagonal(fusedgradedvector(sectordata))

sectordata(d::FusedGradedDiagonal) = map(Diagonal, sectordata(MAK.diagview(d)))

# ---- accessors ----

function datatype(::Type{<:FusedGradedDiagonal{T, S, V}}) where {T, S, V}
    return Diagonal{T, Base.promote_op(view, V, UnitRange{Int})}
end

axes_codomain(d::FusedGradedDiagonal) = (axis(MAK.diagview(d)),)
axes_domain(d::FusedGradedDiagonal) = (axis(MAK.diagview(d)),)

function Base.similar(d::FusedGradedDiagonal, ::Type{T}) where {T}
    return FusedGradedDiagonal(similar(MAK.diagview(d), T))
end

LinearAlgebra.diag(d::FusedGradedDiagonal) = copy(MAK.diagview(d))
function LinearAlgebra.diag(d::FusedGradedDiagonal, k::Integer)
    return error("`diag` on a `FusedGradedDiagonal` supports only the main diagonal")
end

# ---- permutation ----

# A transpose would dualize the codomain axis, which a `FusedGradedDiagonal` cannot represent.
function TensorAlgebra.permuteddims(d::FusedGradedDiagonal, perm)
    perm == (1, 2) || throw(
        ArgumentError(
            "permuting a `FusedGradedDiagonal` by `$perm` is not supported; only the identity permutation is allowed"
        )
    )
    return d
end

# ---- densify / matricize ----

# Densify to a full `FusedGradedMatrix`: each `Diagonal` block becomes a dense block (`copyto!` zeros
# the off-diagonal). The block-wise `copyto!` keeps the block backend rather than routing through
# `Array`/`collect`, which would force a CPU result. Used where a diagonal result is not representable
# as a `FusedGradedDiagonal` (a non-`{1,1}` matricize / bond-split unmatricize).
function densematrix(d::FusedGradedDiagonal)
    m = FusedGradedMatrix{eltype(d)}(undef, axis_codomain(d), axis_domain(d))
    return copyto!(m, d)
end

# `matricize(d, Val(1))` is the identity `{1,1}` matricization of the diagonal (handled by the
# `AbstractFusedGradedMatrix` method). Any other pattern is not representable as a
# `FusedGradedDiagonal`, so densify to a `FusedGradedMatrix` rather than throwing.
function TensorAlgebra.matricize(
        ::GradedMatricize,
        d::FusedGradedDiagonal,
        ::Val{K}
    ) where {K}
    K == 1 && return d
    return densematrix(d)
end

# The product of two diagonal fused matrices over a single contracted leg is again diagonal, so
# allocate a `FusedGradedDiagonal` (the block-wise `mul!` fills it via `Diagonal * Diagonal`). Mixed
# diagonal/dense products fall through to the general `AbstractFusedGradedMatrix` method (dense).
function allocate_output(
        ::typeof(*), A::FusedGradedDiagonal, B::FusedGradedDiagonal
    )
    Tout = Base.promote_op(*, eltype(A), eltype(B))
    return FusedGradedDiagonal{Tout}(undef, axis_codomain(A))
end

# ---- broadcasting ----

struct FusedGradedDiagonalStyle <: AbstractFusedGradedStyle{2} end
FusedGradedDiagonalStyle(::Val{2}) = FusedGradedDiagonalStyle()

BC.BroadcastStyle(::Type{<:FusedGradedDiagonal}) = FusedGradedDiagonalStyle()

# Mixed with a dense fused matrix, a diagonal promotes to dense. Only one operand order is needed:
# Base's `result_join` retries the reversed order when the first returns `Unknown`.
BC.BroadcastStyle(style::FusedGradedStyle{2}, ::FusedGradedDiagonalStyle) = style

function Base.similar(bc::BC.Broadcasted{<:FusedGradedDiagonalStyle}, elt::Type)
    # TODO: generalize to non-CPU storage (e.g. GPU arrays). The style should carry a data/storage style
    # (as Base's `ArrayStyle` carries the array type) so `similar` allocates backing storage of the right
    # type rather than always defaulting to a dense CPU array.
    return FusedGradedDiagonal{elt}(undef, first(axes(bc)))
end
