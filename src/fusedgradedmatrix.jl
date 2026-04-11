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

The codomain (row) axis is non-dual with sectors `sectors[i]` and sector lengths
derived from `size(blocks[i], 1)`. The domain (column) axis is dual with sectors
`dual(sectors[i])` and sector lengths from `size(blocks[i], 2)`.
"""
struct FusedGradedMatrix{T, D <: AbstractMatrix{T}, S <: SectorRange} <:
    AbstractGradedMatrix{T}
    sectors::Vector{S}
    blocks::Vector{D}
    function FusedGradedMatrix{T, D, S}(
            sectors::Vector{S},
            blocks::Vector{D}
        ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
        length(sectors) == length(blocks) ||
            throw(ArgumentError("sectors and blocks must have the same length"))
        issorted(sectors) ||
            throw(ArgumentError("sectors must be sorted"))
        allunique(sectors) ||
            throw(ArgumentError("sectors must be unique"))
        return new{T, D, S}(sectors, blocks)
    end
end
function FusedGradedMatrix(
        sectors::Vector{S},
        blocks::Vector{D}
    ) where {T, S <: SectorRange, D <: AbstractMatrix{T}}
    return FusedGradedMatrix{T, D, S}(sectors, blocks)
end

# ========================  undef constructors  ========================

function FusedGradedMatrix{T, D, S}(
        ::UndefInitializer,
        sectors::Vector{S},
        axes::Tuple{BlockedOneTo, BlockedOneTo}
    ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
    cod_axes = eachblockaxis(axes[1])
    dom_axes = eachblockaxis(axes[2])
    length(cod_axes) == length(dom_axes) == length(sectors) ||
        throw(ArgumentError("axes block counts must match sectors length"))
    blks = [similar(D, (cod_axes[i], dom_axes[i])) for i in eachindex(sectors)]
    return FusedGradedMatrix{T, D, S}(sectors, blks)
end

# Convenience: default D = Matrix{T}.
function FusedGradedMatrix{T}(
        ::UndefInitializer,
        sectors::Vector{S},
        axes::Tuple{BlockedOneTo, BlockedOneTo}
    ) where {T, S <: SectorRange}
    return FusedGradedMatrix{T, Matrix{T}, S}(undef, sectors, axes)
end

# Vector{Int} convenience: wraps into BlockedOneTo and delegates.
function FusedGradedMatrix{T}(
        ::UndefInitializer,
        sectors::Vector{<:SectorRange},
        codomain_blocklengths::Vector{Int},
        domain_blocklengths::Vector{Int}
    ) where {T}
    return FusedGradedMatrix{T}(
        undef, sectors,
        blockedrange.((codomain_blocklengths, domain_blocklengths))
    )
end

# ========================  Accessors  ========================

BlockArrays.blocklength(m::FusedGradedMatrix) = length(m.sectors)
function BlockSparseArrays.blocktype(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S}
    return SectorMatrix{T, D, S}
end
BlockSparseArrays.blocktype(m::FusedGradedMatrix) = BlockSparseArrays.blocktype(typeof(m))
sectortype(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S} = S

function Base.axes(m::FusedGradedMatrix)
    codomain = gradedrange(m.sectors .=> size.(m.blocks, 1))
    domain = gradedrange(dual.(m.sectors) .=> size.(m.blocks, 2))
    return (codomain, domain)
end

Base.size(m::FusedGradedMatrix) = map(length, axes(m))
Base.eltype(::Type{FusedGradedMatrix{T}}) where {T} = T
Base.eltype(::Type{<:FusedGradedMatrix{T}}) where {T} = T

# ========================  Block indexing (primitive)  ========================

function Base.view(m::FusedGradedMatrix, I::Block{2})
    i, j = Int.(Tuple(I))
    i == j ||
        error("Off-diagonal access not supported for block-diagonal FusedGradedMatrix")
    return SectorMatrix(m.sectors[i], m.blocks[i])
end

# ========================  eachblockstoredindex  ========================

function BlockSparseArrays.eachblockstoredindex(m::FusedGradedMatrix)
    return (Block(i, i) for i in eachindex(m.sectors))
end

# ========================  blocks  ========================

using LinearAlgebra: Diagonal
function BlockArrays.blocks(m::FusedGradedMatrix)
    diagblocks = map(I -> view(m, I), eachblockstoredindex(m))
    return Diagonal(collect(diagblocks))
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

function check_mul_axes(
        C::FusedGradedMatrix, A::FusedGradedMatrix, B::FusedGradedMatrix
    )
    axes(A, 2) == dual(axes(B, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    axes(C, 1) == axes(A, 1) || throw(DimensionMismatch())
    axes(C, 2) == axes(B, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        C::FusedGradedMatrix, A::FusedGradedMatrix, B::FusedGradedMatrix,
        α::Number, β::Number
    )
    check_mul_axes(C, A, B)
    for I in blockdiagindices(C)
        mul!(view(C, Data(I)), view(A, Data(I)), view(B, Data(I)), α, β)
    end
    return C
end

function Base.:(*)(A::FusedGradedMatrix{T₁}, B::FusedGradedMatrix{T₂}) where {T₁, T₂}
    axes(A, 2) == dual(axes(B, 1)) || throw(DimensionMismatch("sectors must match"))
    T = Base.promote_op(LinearAlgebra.matprod, T₁, T₂)
    result_blocks = [view(A, Data(I)) * view(B, Data(I)) for I in blockdiagindices(A)]
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
    fused_axes = blockedrange.((datalengths(row_ax), datalengths(col_ax)))
    m = FusedGradedMatrix{T}(undef, fused_sectors, fused_axes)
    for I in blockdiagindices(m)
        m[Data(I)] = view(a, Data(I))
    end
    return m
end

"""
    AbelianGradedArray(m::FusedGradedMatrix)

Convert a `FusedGradedMatrix` to a 2D `AbelianGradedArray`.
Inverse of `FusedGradedMatrix(::AbelianGradedArray)`.
"""
function AbelianGradedArray(m::FusedGradedMatrix{T}) where {T}
    a = similar(m, axes(m))
    for I in blockdiagindices(m)
        a[Data(I)] = view(m, Data(I))
    end
    return a
end
