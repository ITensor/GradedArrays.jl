# ===========================================================================
#  FusedSectorMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedSectorMatrix{T,I<:TKS.Sector,D<:AbstractMatrix{T}}

Block-diagonal matrix produced by matricizing an `AbstractGradedArray`.
Each diagonal block corresponds to a coupled sector from fusing codomain/domain legs.

Fields:

  - `labels::Vector{I}` — coupled sector labels, sorted and unique
  - `blocks::Vector{D}` — diagonal blocks, one per sector

The codomain (row) axis is non-dual with sectors `labels[i]` and multiplicities
derived from `size(blocks[i], 1)`. The domain (column) axis is dual with sectors
`dual(labels[i])` and multiplicities from `size(blocks[i], 2)`.
"""
struct FusedSectorMatrix{T, I <: TKS.Sector, D <: AbstractMatrix{T}} <:
    AbstractGradedArray{T, 2}
    labels::Vector{I}
    blocks::Vector{D}
    function FusedSectorMatrix(
            labels::Vector{I},
            blocks::Vector{D}
        ) where {T, I, D <: AbstractMatrix{T}}
        length(labels) == length(blocks) ||
            throw(ArgumentError("labels and blocks must have the same length"))
        issorted(SectorRange.(labels)) ||
            throw(ArgumentError("labels must be sorted"))
        allunique(SectorRange.(labels)) ||
            throw(ArgumentError("labels must be unique"))
        return new{T, I, D}(labels, blocks)
    end
end

# ========================  Accessors  ========================

labels(m::FusedSectorMatrix) = m.labels
BlockArrays.blocklength(m::FusedSectorMatrix) = length(m.labels)
BlockSparseArrays.blocktype(::Type{<:FusedSectorMatrix{T, I, D}}) where {T, I, D} = D
BlockSparseArrays.blocktype(m::FusedSectorMatrix) = blocktype(typeof(m))
sector_type(::Type{<:FusedSectorMatrix{T, I}}) where {T, I} = SectorRange{I}

function Base.axes(m::FusedSectorMatrix{T, I}) where {T, I}
    codomain_sectors = SectorRange.(m.labels)
    domain_sectors = dual.(codomain_sectors)
    codomain = gradedrange(codomain_sectors .=> size.(m.blocks, 1))
    domain = gradedrange(domain_sectors .=> size.(m.blocks, 2))
    return (codomain, domain)
end

Base.size(m::FusedSectorMatrix) = map(length, axes(m))
Base.eltype(::Type{FusedSectorMatrix{T}}) where {T} = T
Base.eltype(::Type{<:FusedSectorMatrix{T}}) where {T} = T

# ========================  Block indexing  ========================

function Base.getindex(m::FusedSectorMatrix{T}, I::Block{2}) where {T}
    i, j = Int.(Tuple(I))
    i == j || return zeros(T, 0, 0)
    return m.blocks[i]
end

function Base.getindex(m::FusedSectorMatrix, i::Block{1}, j::Block{1})
    return m[Block(Int(i), Int(j))]
end

# ========================  eachblockstoredindex  ========================

function BlockSparseArrays.eachblockstoredindex(m::FusedSectorMatrix)
    return (Block(i, i) for i in eachindex(m.labels))
end

# ========================  blocks  ========================

using LinearAlgebra: Diagonal
BlockArrays.blocks(m::FusedSectorMatrix) = Diagonal(m.blocks)

# ========================  fill! / zero!  ========================

function FI.zero!(m::FusedSectorMatrix)
    for b in m.blocks
        fill!(b, zero(eltype(m)))
    end
    return m
end

function Base.fill!(m::FusedSectorMatrix, v)
    iszero(v) || throw(
        ArgumentError("fill! with nonzero value is not supported for FusedSectorMatrix")
    )
    return FI.zero!(m)
end

# ========================  mul!  ========================

function LinearAlgebra.mul!(
        C::FusedSectorMatrix, A::FusedSectorMatrix, B::FusedSectorMatrix,
        α::Number, β::Number
    )
    C.labels == A.labels == B.labels ||
        throw(DimensionMismatch("FusedSectorMatrix sectors must match"))
    for i in eachindex(C.blocks)
        mul!(C.blocks[i], A.blocks[i], B.blocks[i], α, β)
    end
    return C
end

function Base.:(*)(A::FusedSectorMatrix{T₁}, B::FusedSectorMatrix{T₂}) where {T₁, T₂}
    A.labels == B.labels || throw(DimensionMismatch("sectors must match"))
    T = Base.promote_op(LinearAlgebra.matprod, T₁, T₂)
    result_blocks = [A.blocks[i] * B.blocks[i] for i in eachindex(A.blocks)]
    return FusedSectorMatrix(copy(A.labels), result_blocks)
end

# ========================  similar  ========================

function Base.similar(m::FusedSectorMatrix{<:Any, I}, ::Type{T}) where {T, I}
    new_blocks = [similar(b, T) for b in m.blocks]
    return FusedSectorMatrix(copy(m.labels), new_blocks)
end

# ========================  show  ========================

function Base.show(io::IO, ::MIME"text/plain", m::FusedSectorMatrix{T}) where {T}
    nblocks = length(m.labels)
    print(io, nblocks, "-block FusedSectorMatrix{", T, "} with sectors ")
    print(io, "[")
    join(io, m.labels, ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, m::FusedSectorMatrix{T}) where {T}
    nblocks = length(m.labels)
    print(io, nblocks, "-block FusedSectorMatrix{", T, "}")
    return nothing
end

# ========================  Conversion from AbelianArray  ========================

# Identity
FusedSectorMatrix(m::FusedSectorMatrix) = m

"""
    FusedSectorMatrix(a::AbelianMatrix{T})

Convert a 2D block-diagonal `AbelianArray` (as produced by `matricize`) into a
`FusedSectorMatrix`. Extracts diagonal blocks from the stored entries.
"""
function FusedSectorMatrix(a::AbelianMatrix{T}) where {T}
    row_ax, col_ax = axes(a)
    n = blocklength(row_ax)
    blocklength(col_ax) == n ||
        throw(ArgumentError("AbelianMatrix must have matching row/column block counts"))

    row_sectors = sectors(row_ax)
    col_sectors = sectors(col_ax)
    row_sectors == dual.(col_sectors) || throw(
        ArgumentError(
            "AbelianMatrix axes must be canonical duals to convert to FusedSectorMatrix"
        )
    )
    BlockSparseArrays.isblockdiagonal(a) || throw(
        ArgumentError(
            "AbelianMatrix must be block-diagonal to convert to FusedSectorMatrix"
        )
    )

    sector_labels = collect(labels(row_ax))
    # Default: zero blocks with the right dimensions from row/col axes
    row_bls = blocklengths(row_ax)
    col_bls = blocklengths(col_ax)
    diag_blocks = [zeros(T, row_bls[i], col_bls[i]) for i in 1:n]
    for bI in eachblockstoredindex(a)
        i = Int(Tuple(bI)[1])
        diag_blocks[i] = Array(a[bI])
    end
    return FusedSectorMatrix(sector_labels, diag_blocks)
end

"""
    AbelianArray(m::FusedSectorMatrix)

Convert a `FusedSectorMatrix` to a 2D `AbelianArray`.
Inverse of `FusedSectorMatrix(::AbelianArray)`.
"""
function AbelianArray(m::FusedSectorMatrix{T}) where {T}
    codomain, domain = axes(m)
    a = similar(m, (codomain, domain))
    for (i, block) in enumerate(m.blocks)
        iszero(block) && continue
        row_sector = sectors(codomain)[i]
        col_sector = sectors(domain)[i]
        a[Block(i, i)] = SectorArray((row_sector, col_sector), block)
    end
    return a
end
