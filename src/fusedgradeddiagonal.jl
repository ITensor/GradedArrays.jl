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

function FusedGradedDiagonal{T}(
        ::UndefInitializer, axis::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange}
    return FusedGradedDiagonal(FusedGradedVector{T}(undef, axis))
end

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

# `check_input` would require `conj` to dualize the axes, but a diagonal is self-dual, so its axes
# just have to match; each block then delegates to the plain-array `bipermutedimsopadd!`.
function TensorAlgebra.bipermutedimsopadd!(
        dest::FusedGradedDiagonal, op, src::FusedGradedDiagonal,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    axes(dest) == axes(src) ||
        throw(DimensionMismatch("`bipermutedimsopadd!` requires matching axes"))
    foreachblock(MAK.diagview(dest), MAK.diagview(src)) do _, (d, s)
        return TensorAlgebra.bipermutedimsopadd!(d, op, s, (1,), (), α, β)
    end
    return dest
end

function TensorAlgebra.allocate_output(
        ::typeof(TA.permutedimsop), op, src::FusedGradedDiagonal, perm_codomain, perm_domain
    )
    return FusedGradedDiagonal(
        TensorAlgebra.allocate_output(TA.permutedimsop, op, MAK.diagview(src), (1,), ())
    )
end

# ---- broadcasting ----

struct FusedGradedDiagonalStyle <: AbstractFusedGradedStyle{2} end
FusedGradedDiagonalStyle(::Val{2}) = FusedGradedDiagonalStyle()

BC.BroadcastStyle(::Type{<:FusedGradedDiagonal}) = FusedGradedDiagonalStyle()

# Mixed with a dense fused matrix, a diagonal promotes to dense.
BC.BroadcastStyle(style::FusedGradedStyle{2}, ::FusedGradedDiagonalStyle) = style
BC.BroadcastStyle(::FusedGradedDiagonalStyle, style::FusedGradedStyle{2}) = style

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

# Without this, `conj` hits the fused-array method that errors; the dot routes it through the style.
Base.conj(a::FusedGradedDiagonal) = conj.(a)
