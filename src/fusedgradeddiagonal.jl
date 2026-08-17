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

"""
    FusedGradedDiagonal(sectors .=> data)

Build a `FusedGradedDiagonal` from the per-sector diagonal data: the pair `sectors[i] => data[i]` gives
the diagonal entries of the block at `sectors[i]`.
"""
function FusedGradedDiagonal(sectordata::AbstractVector{<:Pair})
    return FusedGradedDiagonal(FusedGradedVector(last.(sectordata), first.(sectordata)))
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

function TensorAlgebra.allocate_output(
        ::typeof(TA.permutedimsop), op, src::FusedGradedDiagonal, perm_codomain, perm_domain
    )
    check_input(TA.permutedimsop, op, src, perm_codomain, perm_domain)
    # Both the identity copy and the adjoint allocate the same diagonal (same non-dual axis); `op` acts
    # on the data during the permute-add, not here. Passing `conj` would dualize the diagview's axis, so
    # allocate with `identity`.
    return FusedGradedDiagonal(
        TensorAlgebra.allocate_output(
            TA.permutedimsop,
            identity,
            MAK.diagview(src),
            (1,),
            ()
        )
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
    # TODO: generalize to non-CPU storage (e.g. GPU arrays). The style should carry a data/storage style
    # (as Base's `ArrayStyle` carries the array type) so `similar` allocates backing storage of the right
    # type rather than always defaulting to a dense CPU array.
    return FusedGradedDiagonal{elt}(undef, first(axes(bc)))
end
