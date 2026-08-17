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

# Allocate with a given per-sector diagonal axis (the reduced/degeneracy dimension of each sector).
function FusedGradedDiagonal{T}(
        ::UndefInitializer, axis::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange}
    return FusedGradedDiagonal(FusedGradedVector{T}(undef, axis))
end

# A `Diagonal` matrix is square, so its codomain and domain share the stored diagonal's axis. Each
# block is a `Diagonal` wrapping that sector's view into the diagonal buffer (sharing storage), so
# the shared block-wise matrix operations can read a diagonal like any fused graded matrix; the axes
# derive from `biaxes` below.
sectordata(d::FusedGradedDiagonal) = map(Diagonal, sectordata(MAK.diagview(d)))

# ---- accessors ----

# Each block is a `Diagonal` over a 1-D `view` into the diagonal buffer (see `_dataview`).
function datatype(::Type{<:FusedGradedDiagonal{T, S, V}}) where {T, S, V}
    return Diagonal{T, Base.promote_op(view, V, UnitRange{Int})}
end

# Square: codomain and domain share the diagonal's axis, the axis of the wrapped diagonal vector
# `diagview(d)`. The derived `biaxes` dualizes the domain half, and the block indexing, reductions,
# predicates, and display all derive from these and `sectordata` generically on
# `AbstractFusedGradedMatrix`.
axes_codomain(d::FusedGradedDiagonal) = (axis(MAK.diagview(d)),)
axes_domain(d::FusedGradedDiagonal) = (axis(MAK.diagview(d)),)

function Base.similar(d::FusedGradedDiagonal, ::Type{T}) where {T}
    return FusedGradedDiagonal(similar(MAK.diagview(d), T))
end

# The main diagonal as an owned `FusedGradedVector` (not densified to a plain vector); off-diagonals
# are not supported.
LinearAlgebra.diag(d::FusedGradedDiagonal) = copy(MAK.diagview(d))
function LinearAlgebra.diag(d::FusedGradedDiagonal, k::Integer)
    return error("`diag` on a `FusedGradedDiagonal` supports only the main diagonal")
end

# ---- structure-preserving permutation ----
#
# Mirrors TensorAlgebra's `diagonal.jl` for `LinearAlgebra.Diagonal`: permuting the two axes of a
# square diagonal (identity or transpose) leaves the stored diagonal unchanged, so it stays a
# `FusedGradedDiagonal` through `bipermutedims` and the matrix-function permute rather than
# densifying to a `GradedArray`. This keeps a factorization's diagonal factor (SVD singular values,
# eigenvalues) diagonal through `sqrth_safe`/`invsqrth_safe`, which then use the fast, write-through
# `diagview(::FusedGradedDiagonal)` path instead of the dense one.

# The lazy permutation of a square diagonal is the diagonal itself.
TensorAlgebra.permuteddims(d::FusedGradedDiagonal, perm) = d

# Accumulate straight onto the destination's stored diagonal, skipping the permutation. `op = conj`
# keeps the (self-dual) square band and conjugates the diagonal entries, matching TensorKit's
# data-wise `conj` on a `DiagonalTensorMap`: it acts on the raw band buffer rather than the graded
# band vector, whose `op = conj` would dualize the axis (which the fused-vector storage forbids).
function TensorAlgebra.bipermutedimsopadd!(
        dest::FusedGradedDiagonal, op, src::FusedGradedDiagonal,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    if op === conj
        axes(dest) == axes(src) ||
            throw(DimensionMismatch("`bipermutedimsopadd!` requires matching axes"))
        d, s = MAK.diagview(dest).buffer, MAK.diagview(src).buffer
        iszero(β) ? (d .= α .* conj.(s)) : (d .= β .* d .+ α .* conj.(s))
        return dest
    end
    check_input(bipermutedimsopadd!, dest, op, src, perm_codomain, perm_domain)
    TensorAlgebra.bipermutedimsopadd!(
        MAK.diagview(dest), op, MAK.diagview(src), (1,), (), α, β
    )
    return dest
end

# The bipermutation of a square diagonal is again a square diagonal of the same size, so allocate the
# `permutedimsop`/`bipermutedims` output as a `FusedGradedDiagonal` (the squareness comes from `src`,
# not from row/column axes alone).
function TensorAlgebra.allocate_output(
        ::typeof(TA.permutedimsop), op, src::FusedGradedDiagonal,
        perm_codomain, perm_domain
    )
    T = Base.promote_op(op, eltype(src))
    return FusedGradedDiagonal(similar(MAK.diagview(src), T))
end

# ---- broadcasting ----
#
# A `FusedGradedDiagonal` is a first-class participant in the graded linear-broadcast machinery, so a
# linear broadcast of a diagonal (`S / norm(S)`, `2 .* S`, `conj.(S)`) stays a `FusedGradedDiagonal`
# instead of densifying. The style is a fused-family sibling of `FusedGradedStyle`. A linear
# combination maps `0 -> 0`, so the diagonal structure is preserved with no zero-preservation gate.

struct FusedGradedDiagonalStyle{N} <: AbstractFusedGradedStyle{N} end
FusedGradedDiagonalStyle{N}(::Val{M}) where {N, M} = FusedGradedDiagonalStyle{M}()

BC.BroadcastStyle(::Type{<:FusedGradedDiagonal}) = FusedGradedDiagonalStyle{2}()

# Within the fused family, a diagonal mixed with a non-diagonal fused matrix promotes to the dense
# fused matrix (the diagonal is scattered into it via the permute-add bridge); a diagonal mixed with
# a diagonal stays diagonal (the generic same-style rule in `broadcast.jl`).
BC.BroadcastStyle(style::FusedGradedStyle{2}, ::FusedGradedDiagonalStyle{2}) = style
BC.BroadcastStyle(::FusedGradedDiagonalStyle{2}, style::FusedGradedStyle{2}) = style

# Rebuild the diagonal from the operand's own (non-dual) band axis. A diagonal is square and stays
# square (same band) under a linear broadcast — including `conj`, which keeps the self-dual band (see
# the `op = conj` branch of `bipermutedimsopadd!` above) rather than dualizing it, and any leg
# reordering, which leaves a square diagonal unchanged. `broadcast_leaf` looks through the
# `PermutedDims` alignment leaf the named-tensor broadcast wraps operands in.
function Base.similar(bc::BC.Broadcasted{<:FusedGradedDiagonalStyle}, elt::Type)
    operands = filter(a -> broadcast_leaf(a) isa FusedGradedDiagonal, BC.flatten(bc).args)
    isempty(operands) && error(
        "no `FusedGradedDiagonal` operand found in a `FusedGradedDiagonalStyle` broadcast"
    )
    d = broadcast_leaf(first(operands))
    return FusedGradedDiagonal{elt}(undef, only(axes_codomain(d)))
end

function Base.copyto!(
        dest::FusedGradedDiagonal, bc::BC.Broadcasted{<:FusedGradedDiagonalStyle}
    )
    copyto!(dest, flattenlinear(bc))
    return dest
end

# Route bare `conj` through the broadcast style so it stays a `FusedGradedDiagonal`.
Base.conj(a::FusedGradedDiagonal) = conj.(a)
