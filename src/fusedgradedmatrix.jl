# ===========================================================================
#  FusedGradedMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedGradedMatrix{T,D<:AbstractMatrix{T},S<:SectorRange}

Block-diagonal matrix produced by matricizing an `AbstractGradedArray`.
Each diagonal block corresponds to a coupled sector from fusing codomain/domain legs.

Fields:

  - `sectors::Vector{S}` — coupled sectors, sorted and unique (always non-dual, codomain convention)
  - `blocks::Vector{D}` — diagonal blocks, one per sector

The codomain (row) axis is non-dual with sectors `sectors[i]` and multiplicities
derived from `size(blocks[i], 1)`. The domain (column) axis is dual with sectors
`dual(sectors[i])` and multiplicities from `size(blocks[i], 2)`.
"""
struct FusedGradedMatrix{T, D <: AbstractMatrix{T}, S <: SectorRange} <:
    AbstractGradedArray{T, 2}
    sectors::Vector{S}
    blocks::Vector{D}
    function FusedGradedMatrix(
            sectors::Vector{S},
            blocks::Vector{D}
        ) where {T, S <: SectorRange, D <: AbstractMatrix{T}}
        length(sectors) == length(blocks) ||
            throw(ArgumentError("sectors and blocks must have the same length"))
        issorted(sectors) ||
            throw(ArgumentError("sectors must be sorted"))
        allunique(sectors) ||
            throw(ArgumentError("sectors must be unique"))
        return new{T, D, S}(sectors, blocks)
    end
end

# ========================  Accessors  ========================

BlockArrays.blocklength(m::FusedGradedMatrix) = length(m.sectors)
function BlockSparseArrays.blocktype(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S}
    return SectorMatrix{T, D, S}
end
BlockSparseArrays.blocktype(m::FusedGradedMatrix) = BlockSparseArrays.blocktype(typeof(m))
sector_type(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S} = S

function Base.axes(m::FusedGradedMatrix)
    codomain = gradedrange(m.sectors .=> size.(m.blocks, 1))
    domain = gradedrange(dual.(m.sectors) .=> size.(m.blocks, 2))
    return (codomain, domain)
end

Base.size(m::FusedGradedMatrix) = map(length, axes(m))
Base.eltype(::Type{FusedGradedMatrix{T}}) where {T} = T
Base.eltype(::Type{<:FusedGradedMatrix{T}}) where {T} = T

# ========================  Block indexing  ========================

function Base.view(m::FusedGradedMatrix, I::Block{2})
    i, j = Int.(Tuple(I))
    i == j ||
        error("Off-diagonal access not supported for block-diagonal FusedGradedMatrix")
    return SectorMatrix(m.sectors[i], m.blocks[i])
end
function Base.view(m::FusedGradedMatrix, i::Block{1}, j::Block{1})
    return view(m, Block(Int(i), Int(j)))
end

function Base.getindex(m::FusedGradedMatrix, I::Block{2})
    return copy(view(m, I))
end
function Base.getindex(m::FusedGradedMatrix, i::Block{1}, j::Block{1})
    return m[Block(Int(i), Int(j))]
end

# ========================  eachblockstoredindex  ========================

function BlockSparseArrays.eachblockstoredindex(m::FusedGradedMatrix)
    return (Block(i, i) for i in eachindex(m.sectors))
end

# ========================  blocks  ========================

using LinearAlgebra: Diagonal
function BlockArrays.blocks(m::FusedGradedMatrix)
    sector_blocks = [SectorMatrix(s, b) for (s, b) in zip(m.sectors, m.blocks)]
    return Diagonal(sector_blocks)
end

# ========================  fill! / zero!  ========================

function FI.zero!(m::FusedGradedMatrix)
    for b in m.blocks
        fill!(b, zero(eltype(m)))
    end
    return m
end

function Base.fill!(m::FusedGradedMatrix, v)
    iszero(v) || throw(
        ArgumentError("fill! with nonzero value is not supported for FusedGradedMatrix")
    )
    return FI.zero!(m)
end

# ========================  mul!  ========================

function LinearAlgebra.mul!(
        C::FusedGradedMatrix, A::FusedGradedMatrix, B::FusedGradedMatrix,
        α::Number, β::Number
    )
    C.sectors == A.sectors == B.sectors ||
        throw(DimensionMismatch("FusedGradedMatrix sectors must match"))
    for i in eachindex(C.blocks)
        mul!(C.blocks[i], A.blocks[i], B.blocks[i], α, β)
    end
    return C
end

function Base.:(*)(A::FusedGradedMatrix{T₁}, B::FusedGradedMatrix{T₂}) where {T₁, T₂}
    A.sectors == B.sectors || throw(DimensionMismatch("sectors must match"))
    T = Base.promote_op(LinearAlgebra.matprod, T₁, T₂)
    result_blocks = [A.blocks[i] * B.blocks[i] for i in eachindex(A.blocks)]
    return FusedGradedMatrix(copy(A.sectors), result_blocks)
end

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    new_blocks = [similar(b, T) for b in m.blocks]
    return FusedGradedMatrix(copy(m.sectors), new_blocks)
end

# ========================  show  ========================

function Base.show(io::IO, ::MIME"text/plain", m::FusedGradedMatrix{T}) where {T}
    nblocks = length(m.sectors)
    print(io, nblocks, "-block FusedGradedMatrix{", T, "} with sectors ")
    print(io, "[")
    join(io, label.(m.sectors), ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, m::FusedGradedMatrix{T}) where {T}
    nblocks = length(m.sectors)
    print(io, nblocks, "-block FusedGradedMatrix{", T, "}")
    return nothing
end

# ========================  Conversion from AbelianGradedArray  ========================

# Identity
FusedGradedMatrix(m::FusedGradedMatrix) = m

"""
    FusedGradedMatrix(a::AbelianGradedMatrix{T})

Convert a 2D block-diagonal `AbelianGradedArray` (as produced by `matricize`) into a
`FusedGradedMatrix`. Extracts diagonal blocks from the stored entries.
"""
function FusedGradedMatrix(a::AbelianGradedMatrix{T}) where {T}
    row_ax, col_ax = axes(a)
    n = blocklength(row_ax)
    blocklength(col_ax) == n ||
        throw(
        ArgumentError("AbelianGradedMatrix must have matching row/column block counts")
    )

    row_sectors = sectors(row_ax)
    col_sectors = sectors(col_ax)
    row_sectors == dual.(col_sectors) || throw(
        ArgumentError(
            "AbelianGradedMatrix axes must be canonical duals to convert to FusedGradedMatrix"
        )
    )
    BlockSparseArrays.isblockdiagonal(a) || throw(
        ArgumentError(
            "AbelianGradedMatrix must be block-diagonal to convert to FusedGradedMatrix"
        )
    )

    fused_sectors = collect(row_sectors)
    # Default: zero blocks with the right dimensions from row/col axes
    row_bls = blocklengths(row_ax)
    col_bls = blocklengths(col_ax)
    diag_blocks = [zeros(T, row_bls[i], col_bls[i]) for i in 1:n]
    for bI in eachblockstoredindex(a)
        i = Int(Tuple(bI)[1])
        diag_blocks[i] = Array(a[bI])
    end
    return FusedGradedMatrix(fused_sectors, diag_blocks)
end

"""
    AbelianGradedArray(m::FusedGradedMatrix)

Convert a `FusedGradedMatrix` to a 2D `AbelianGradedArray`.
Inverse of `FusedGradedMatrix(::AbelianGradedArray)`.
"""
function AbelianGradedArray(m::FusedGradedMatrix{T}) where {T}
    codomain, domain = axes(m)
    a = similar(m, (codomain, domain))
    for (i, block) in enumerate(m.blocks)
        iszero(block) && continue
        row_sector = sectors(codomain)[i]
        col_sector = sectors(domain)[i]
        a[Block(i, i)] = AbelianSectorArray((row_sector, col_sector), block)
    end
    return a
end
