# ===========================================================================
#  FusedGradedDiagonal — block-diagonal fused matrix with `Diagonal` blocks
# ===========================================================================

using LinearAlgebra: Diagonal

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
sectordata(d::FusedGradedDiagonal) = map(Diagonal, sectordata(d.diag))

# ---- accessors ----

# Each block is a `Diagonal` over a 1-D `view` into the diagonal buffer (see `_dataview`).
function datatype(::Type{<:FusedGradedDiagonal{T, S, V}}) where {T, S, V}
    return Diagonal{T, Base.promote_op(view, V, UnitRange{Int})}
end

# Square: codomain and domain share the diagonal's axis; `bispace` dualizes the domain half. `biaxes`
# is the core axis accessor (the one place `d.diag.axis` is read directly); the block indexing,
# reductions, predicates, and display all derive from it and `sectordata` generically on
# `AbstractFusedGradedMatrix`.
biaxes(d::FusedGradedDiagonal) = bispace((d.diag.axis,), (d.diag.axis,))

function Base.similar(d::FusedGradedDiagonal, ::Type{T}) where {T}
    return FusedGradedDiagonal(similar(d.diag, T))
end
