# ===========================================================================
#  FusedGradedDiagonal
# ===========================================================================

using LinearAlgebra: Diagonal
using MatrixAlgebraKit: MatrixAlgebraKit as MAK

# `I` is the TensorKitSectors sector type corresponding to the `Sector`, used to aid conversion to
# and from TensorKit. Subject to change.
"""
    FusedGradedDiagonal(v::FusedGradedVector)

A block-diagonal matrix whose blocks are themselves diagonal, with the entries of `v` on its
diagonal. Its codomain and domain are both the axis of `v`. For example, `svd_compact` of a
graded array returns its singular values as a `FusedGradedDiagonal`.
"""
struct FusedGradedDiagonal{T, S <: Sector, V <: DenseVector{T}, I <: TKS.Sector} <:
    AbstractFusedGradedMatrix{T, S}
    diag::FusedGradedVector{T, S, V, I}
end

# Wrap a contiguous `buffer` without copying.
function FusedGradedDiagonal(buffer::DenseVector, axis::AbstractGradedOneTo)
    return FusedGradedDiagonal(FusedGradedVector(buffer, axis))
end

function FusedGradedDiagonal{T}(::UndefInitializer, axis::AbstractGradedOneTo) where {T}
    return FusedGradedDiagonal(FusedGradedVector{T}(undef, axis))
end

# Build from `sector => diagonal` pairs, or a `Dictionary` keyed by sector.
fusedgradeddiagonal(sectordata) = FusedGradedDiagonal(fusedgradedvector(sectordata))

# Forward `V` so the result stays on the same device.
function FusedGradedMatrix(d::FusedGradedDiagonal{T, S, V}) where {T, S, V}
    m = FusedGradedMatrix{T, S, V}(undef, axis_codomain(d), axis_domain(d))
    return copyto!(m, d)
end

# A closure rather than `Diagonal` itself, since `typeof(Diagonal)` is a `UnionAll` and the return
# type would not infer.
function sectordata(d::FusedGradedDiagonal)
    return Iterators.map(b -> Diagonal(b), sectordata(MAK.diagview(d)))
end

function setsectors(d::FusedGradedDiagonal, ss::AbstractVector)
    diag = setsectors(MAK.diagview(d), ss)
    diag === MAK.diagview(d) && return d
    return FusedGradedDiagonal(diag)
end

# ---- accessors ----

function datatype(::Type{<:FusedGradedDiagonal{T, S, V}}) where {T, S, V}
    return Diagonal{T, Base.promote_op(view, V, UnitRange{Int})}
end

axes_codomain(d::FusedGradedDiagonal) = (axis(MAK.diagview(d)),)
axes_domain(d::FusedGradedDiagonal) = (axis(MAK.diagview(d)),)

Base.dataids(d::FusedGradedDiagonal) = Base.dataids(MAK.diagview(d))
Base.parent(d::FusedGradedDiagonal) = MAK.diagview(d)

function Base.similar(d::FusedGradedDiagonal, ::Type{T}) where {T}
    return FusedGradedDiagonal(similar(MAK.diagview(d), T))
end

LinearAlgebra.diag(d::FusedGradedDiagonal) = copy(MAK.diagview(d))
function LinearAlgebra.diag(d::FusedGradedDiagonal, k::Integer)
    return error("`diag` on a `FusedGradedDiagonal` supports only the main diagonal")
end

# ---- permutation ----

# A transpose would make the codomain axis dual, which a `FusedGradedDiagonal` cannot represent.
function TensorAlgebra.permuteddims(d::FusedGradedDiagonal, perm)
    perm == (1, 2) || throw(
        ArgumentError(
            "permuting a `FusedGradedDiagonal` by `$perm` is not supported; only the identity permutation is allowed"
        )
    )
    return d
end

# ---- matricize ----

# A diagonal is already its own `{1,1}` matricization.
function TensorAlgebra.matricizeconjview(
        isconj::Bool, d::FusedGradedDiagonal, perm_codomain, perm_domain
    )
    return d
end
# Only its own square axes can be served, which is what its adjoint asks for.
function Base.similar(
        d::FusedGradedDiagonal, ::Type{T},
        codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {T, S}
    (codomain == axis_codomain(d) && domain == axis_domain(d)) || throw(
        ArgumentError(
            "a `FusedGradedDiagonal` allocates only over its own square codomain/domain axes"
        )
    )
    return similar(d, T)
end

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

# Only one operand order is needed, since Base's `result_join` retries the reversed order.
BC.BroadcastStyle(style::FusedGradedStyle{2}, ::FusedGradedDiagonalStyle) = style

function Base.similar(bc::BC.Broadcasted{<:FusedGradedDiagonalStyle}, elt::Type)
    # TODO: generalize to non-CPU storage (e.g. GPU arrays). The style should carry a data/storage style
    # (as Base's `ArrayStyle` carries the array type) so `similar` allocates backing storage of the right
    # type rather than always defaulting to a dense CPU array.
    return FusedGradedDiagonal{elt}(undef, first(axes(bc)))
end
