# ===========================================================================
#  FusedGradedDiagonal — block-diagonal fused matrix with `Diagonal` blocks
# ===========================================================================

using Dictionaries: gettoken, gettokenvalue
using LinearAlgebra: Diagonal

"""
    FusedGradedDiagonal{T,S<:SectorRange,D<:AbstractVector{T},V<:DenseVector{T}} <: AbstractFusedGradedMatrix{T,S}

Square block-diagonal fused matrix whose every coupled-sector block is a `Diagonal`, backed by a
contiguous diagonal buffer (a [`FusedGradedVector`](@ref)). This is the diagonal factor produced by a
factorization (SVD singular values, eigenvalues), analogous to TensorKit's `DiagonalTensorMap`.
`MatrixAlgebraKit.diagview` returns the stored diagonal, sharing storage.

Fields:

  - `diag::FusedGradedVector` — the stored diagonal, one value per basis vector of each coupled
    sector's reduced (degeneracy) space. Owns the storage.
  - `blocks::Dictionary{S,Diagonal}` — per-sector `Diagonal` views wrapping `diag`'s blocks, so
    writing a block writes through to `diag` (no copy).
"""
struct FusedGradedDiagonal{
        T,
        S <: SectorRange,
        D <: AbstractVector{T},
        V <: DenseVector{T},
    } <:
    AbstractFusedGradedMatrix{T, S}
    diag::FusedGradedVector{T, S, D, V}
    blocks::Dictionary{S, Diagonal{T, D}}
end

# Wrap a `FusedGradedVector` as its block-diagonal matrix, sharing storage: each block is a
# `Diagonal` over that sector's view into the contiguous diagonal buffer.
function FusedGradedDiagonal(diag::FusedGradedVector{T, S, D, V}) where {T, S, D, V}
    blocks = map(Diagonal, diag.blocks)
    return FusedGradedDiagonal{T, S, D, V}(diag, blocks)
end

# Allocate with a given per-sector diagonal length (the reduced/degeneracy dimension of each sector).
function FusedGradedDiagonal{T}(
        ::UndefInitializer, axis::Dictionary{S, Int}
    ) where {T, S <: SectorRange}
    return FusedGradedDiagonal(FusedGradedVector{T}(undef, axis))
end

# A `Diagonal` matrix is square, so its codomain and domain share the stored diagonal's axis. These
# accessors let the shared block-wise matrix operations read a diagonal like any fused graded matrix.
sectordata(d::FusedGradedDiagonal) = d.blocks
sectordatalengths_codomain(d::FusedGradedDiagonal) = d.diag.axis
sectordatalengths_domain(d::FusedGradedDiagonal) = d.diag.axis

# ---- accessors ----

BlockArrays.blocklength(d::FusedGradedDiagonal) = length(d.blocks)

function blocktype(::Type{<:FusedGradedDiagonal{T, S, D}}) where {T, S, D}
    return FusedSectorMatrix{T, S, Diagonal{T, D}}
end
blocktype(d::FusedGradedDiagonal) = blocktype(typeof(d))

function biaxes(d::FusedGradedDiagonal)
    ax = d.diag.axis
    cod = gradedrange(collect(pairs(ax)))
    return bispace((cod,), (cod,))
end
Base.axes(d::FusedGradedDiagonal) = Tuple(biaxes(d))
Base.size(d::FusedGradedDiagonal) = map(length, axes(d))
Base.eltype(::Type{<:FusedGradedDiagonal{T}}) where {T} = T

function Base.view(d::FusedGradedDiagonal, I::Block{2})
    i, j = Int.(Tuple(I))
    @boundscheck begin
        (i in 1:length(d.diag.axis) && j in 1:length(d.diag.axis)) ||
            throw(BoundsError(d, I))
    end
    i == j || error("Off-diagonal access not supported for FusedGradedDiagonal")
    s = gettokenvalue(keys(d.diag.axis), i)
    return FusedSectorMatrix(d.blocks[s], s)
end

function eachblockstoredindex(d::FusedGradedDiagonal)
    return (
        Block(gettoken(d.diag.axis, c)[2][2], gettoken(d.diag.axis, c)[2][2]) for
            c in keys(d.blocks)
    )
end

Base.copy(d::FusedGradedDiagonal) = FusedGradedDiagonal(copy(d.diag))
function Base.similar(d::FusedGradedDiagonal, ::Type{T}) where {T}
    return FusedGradedDiagonal(similar(d.diag, T))
end

# Block-diagonal boolean queries delegate to the (diagonal) blocks, mirroring `FusedGradedMatrix`.
function LinearAlgebra.isposdef(d::FusedGradedDiagonal)
    return all(LinearAlgebra.isposdef, values(d.blocks))
end
Base.iszero(d::FusedGradedDiagonal) = all(iszero, values(d.blocks))
LinearAlgebra.isdiag(::FusedGradedDiagonal) = true

# ---- show ----

function Base.summary(io::IO, d::FusedGradedDiagonal)
    print(
        io, length(d.diag.axis), "-block ", summary_typename(typeof(d)),
        " with ", length(d.blocks), " stored block",
        length(d.blocks) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(d.blocks), ", ")
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
        io, length(d.diag.axis), "-block ", summary_typename(typeof(d)),
        " (", length(d.blocks), " stored)"
    )
    return nothing
end
