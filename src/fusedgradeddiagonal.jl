# ===========================================================================
#  FusedGradedDiagonal — block-diagonal fused matrix with `Diagonal` blocks
# ===========================================================================

using Dictionaries: gettoken, gettokenvalue
using LinearAlgebra: Diagonal

"""
    FusedGradedDiagonal{T,S<:SectorRange,D<:AbstractVector{T},V<:DenseVector{T}} <: AbstractFusedGradedMatrix{T,S}

Square block-diagonal fused matrix whose every coupled-sector block is a `Diagonal`, the diagonal
factor produced by a factorization (SVD singular values, eigenvalues). Analogous to TensorKit's
`DiagonalTensorMap`.
"""
struct FusedGradedDiagonal{
        T,
        S <: SectorRange,
        D <: AbstractVector{T},
        V <: DenseVector{T},
    } <:
    AbstractFusedGradedMatrix{T, S}
    diag::FusedGradedVector{T, S, D, V}
    sectordata::Dictionary{S, Diagonal{T, D}}
end

# Wrap a `FusedGradedVector` as its block-diagonal matrix, sharing storage: each block is a
# `Diagonal` over that sector's view into the contiguous diagonal buffer.
function FusedGradedDiagonal(diag::FusedGradedVector{T, S, D, V}) where {T, S, D, V}
    blocks = map(Diagonal, diag.sectordata)
    return FusedGradedDiagonal{T, S, D, V}(diag, blocks)
end

# Allocate with a given per-sector diagonal axis (the reduced/degeneracy dimension of each sector).
function FusedGradedDiagonal{T}(
        ::UndefInitializer, axis::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange}
    return FusedGradedDiagonal(FusedGradedVector{T}(undef, axis))
end

# A `Diagonal` matrix is square, so its codomain and domain share the stored diagonal's axis.
# `sectordata` lets the shared block-wise matrix operations read a diagonal like any fused graded
# matrix; the axes derive from `biaxes` below.
sectordata(d::FusedGradedDiagonal) = d.sectordata

# ---- accessors ----

function blocktype(::Type{<:FusedGradedDiagonal{T, S, D}}) where {T, S, D}
    return FusedSectorMatrix{T, S, Diagonal{T, D}}
end
blocktype(d::FusedGradedDiagonal) = blocktype(typeof(d))

# Square: codomain and domain share the diagonal's axis; `bispace` dualizes the domain half.
biaxes(d::FusedGradedDiagonal) = bispace((d.diag.axis,), (d.diag.axis,))
Base.axes(d::FusedGradedDiagonal) = Tuple(biaxes(d))
Base.size(d::FusedGradedDiagonal) = map(length, axes(d))

function Base.view(d::FusedGradedDiagonal, I::Block{2})
    i, j = Int.(Tuple(I))
    @boundscheck begin
        (i in 1:blocklength(d.diag.axis) && j in 1:blocklength(d.diag.axis)) ||
            throw(BoundsError(d, I))
    end
    i == j || error("Off-diagonal access not supported for FusedGradedDiagonal")
    s = sectors(d.diag.axis)[i]
    return FusedSectorMatrix(d.sectordata[s], s)
end

function eachblockstoredindex(d::FusedGradedDiagonal)
    ax = sectordatalengths(d.diag.axis)
    return (
        Block(gettoken(ax, c)[2][2], gettoken(ax, c)[2][2]) for
            c in keys(d.sectordata)
    )
end

Base.copy(d::FusedGradedDiagonal) = FusedGradedDiagonal(copy(d.diag))
function Base.similar(d::FusedGradedDiagonal, ::Type{T}) where {T}
    return FusedGradedDiagonal(similar(d.diag, T))
end

# The generic `isposdef` misreads the block structure (returns `false` for a positive-definite
# graded diagonal), so decide it block-wise: positive-definite iff every stored block is.
function LinearAlgebra.isposdef(d::FusedGradedDiagonal)
    return all(LinearAlgebra.isposdef, values(d.sectordata))
end

# ---- show ----

function Base.summary(io::IO, d::FusedGradedDiagonal)
    print(
        io, blocklength(d.diag.axis), "-block ", summary_typename(typeof(d)),
        " with ", length(d.sectordata), " stored block",
        length(d.sectordata) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(d.sectordata), ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", d::FusedGradedDiagonal)
    summary(io, d)
    println(io, ":")
    Base.print_array(io, d)
    return nothing
end

function Base.show(io::IO, d::FusedGradedDiagonal)
    print(
        io, blocklength(d.diag.axis), "-block ", summary_typename(typeof(d)),
        " (", length(d.sectordata), " stored)"
    )
    return nothing
end
