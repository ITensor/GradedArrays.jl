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
sector_type(::Type{<:FusedSectorMatrix{T, I}}) where {T, I} = SectorRange{I}

function Base.axes(m::FusedSectorMatrix{T, I}) where {T, I}
    codomain = gradedrange(
        [
            SectorRange(s) => div(size(m.blocks[i], 1), quantum_dimension(SectorRange(s)))
                for (i, s) in enumerate(m.labels)
        ]
    )
    domain = gradedrange(
        [
            SectorRange(dual(s), true) =>
                div(size(m.blocks[i], 2), quantum_dimension(SectorRange(dual(s))))
                for (i, s) in enumerate(m.labels)
        ]
    )
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

"""
    FusedSectorMatrix(a::AbelianArray{T,2})

Convert a 2D block-diagonal `AbelianArray` (as produced by `matricize`) into a
`FusedSectorMatrix`. Extracts diagonal blocks by matching row sectors with their
zero-flux column counterparts.
"""
# Identity
FusedSectorMatrix(m::FusedSectorMatrix) = m

function FusedSectorMatrix(a::AbelianArray{T, 2}) where {T}
    row_axis = a.axes[1]
    col_axis = a.axes[2]

    # Build lookup: for each col block, map its "matching label" to its position.
    # The matching label is the label that a row sector must have to form zero flux.
    # Zero flux: row_sector ⊗ flip(col_sector) = trivial
    # For abelian: flip(SectorRange(l, d)) = SectorRange(dual(l), !d)
    # So row label must equal dual(col_label) (accounting for duality).
    col_sects = sectors(col_axis)
    # Build lookup: for each col sector, compute the matching row label for zero flux.
    # Zero flux: label(row) == dual(label(col)) if col is non-dual,
    #            label(row) == label(col) if col is dual.
    # Equivalently: the matching row label is label(flip(col_sector)).
    I = eltype(labels(row_axis))
    col_lookup = Dict{I, Int}()
    for (j, cs) in enumerate(col_sects)
        col_lookup[label(flip(cs))] = j
    end

    row_sects = sectors(row_axis)
    sector_labels = I[]
    diag_blocks = Matrix{T}[]

    for (i, rs) in enumerate(row_sects)
        j = get(col_lookup, label(rs), nothing)
        row_len = blocklengths(row_axis)[i]
        if isnothing(j)
            push!(sector_labels, label(rs))
            push!(diag_blocks, zeros(T, row_len, 0))
        else
            col_len = blocklengths(col_axis)[j]
            data = get(a.blockdata, (i, j), nothing)
            push!(sector_labels, label(rs))
            push!(diag_blocks, isnothing(data) ? zeros(T, row_len, col_len) : data)
        end
    end

    return FusedSectorMatrix(sector_labels, diag_blocks)
end

"""
    AbelianArray(m::FusedSectorMatrix)

Convert a `FusedSectorMatrix` to a 2D `AbelianArray` with merged graded axes.
Inverse of `FusedSectorMatrix(::AbelianArray)`.
"""
function AbelianArray(m::FusedSectorMatrix{T}) where {T}
    ax = axes(m)
    a = AbelianArray{T}(undef, ax)

    I = eltype(m.labels)
    col_sects = sectors(ax[2])
    col_lookup = Dict{I, Int}()
    for (j, cs) in enumerate(col_sects)
        col_lookup[label(flip(cs))] = j
    end

    for (i, (s, block)) in enumerate(zip(m.labels, m.blocks))
        j = get(col_lookup, s, nothing)
        isnothing(j) && continue
        a.blockdata[(i, j)] = block
    end
    return a
end
