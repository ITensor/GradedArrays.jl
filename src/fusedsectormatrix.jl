# ===========================================================================
#  FusedSectorMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedSectorMatrix{T,I<:TKS.Sector,D<:AbstractMatrix{T}}

Block-diagonal matrix produced by matricizing an `AbstractGradedArray`.
Each diagonal block corresponds to a coupled sector from fusing codomain/domain legs.

Fields:

  - `sectors::Vector{I}` — coupled sector labels, sorted and unique
  - `blocks::Vector{D}` — diagonal blocks, one per sector

The codomain (row) axis is non-dual with sectors `sectors[i]` and multiplicities
derived from `size(blocks[i], 1)`. The domain (column) axis is dual with sectors
`dual(sectors[i])` and multiplicities from `size(blocks[i], 2)`.
"""
struct FusedSectorMatrix{T, I <: TKS.Sector, D <: AbstractMatrix{T}}
    sectors::Vector{I}
    blocks::Vector{D}
    function FusedSectorMatrix(
            sectors::Vector{I},
            blocks::Vector{D}
        ) where {T, I, D <: AbstractMatrix{T}}
        length(sectors) == length(blocks) ||
            throw(ArgumentError("sectors and blocks must have the same length"))
        issorted(sectors) ||
            throw(ArgumentError("sectors must be sorted"))
        allunique(sectors) ||
            throw(ArgumentError("sectors must be unique"))
        return new{T, I, D}(sectors, blocks)
    end
end

# ========================  Accessors  ========================

labels(m::FusedSectorMatrix) = m.sectors
BlockArrays.blocklength(m::FusedSectorMatrix) = length(m.sectors)
sector_type(::Type{<:FusedSectorMatrix{T, I}}) where {T, I} = SectorRange{I}

function Base.axes(m::FusedSectorMatrix{T, I}) where {T, I}
    codomain = gradedrange(
        [
            SectorRange(s) => div(size(m.blocks[i], 1), TKS.dim(s))
                for (i, s) in enumerate(m.sectors)
        ]
    )
    domain = gradedrange(
        [
            SectorRange(dual(s), true) => div(size(m.blocks[i], 2), TKS.dim(dual(s)))
                for (i, s) in enumerate(m.sectors)
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
    return (Block(i, i) for i in eachindex(m.sectors))
end

# ========================  similar  ========================

function Base.similar(m::FusedSectorMatrix{<:Any, I}, ::Type{T}) where {T, I}
    new_blocks = [similar(b, T) for b in m.blocks]
    return FusedSectorMatrix(copy(m.sectors), new_blocks)
end

# ========================  show  ========================

function Base.show(io::IO, ::MIME"text/plain", m::FusedSectorMatrix{T}) where {T}
    nblocks = length(m.sectors)
    print(io, nblocks, "-block FusedSectorMatrix{", T, "} with sectors ")
    print(io, "[")
    join(io, m.sectors, ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, m::FusedSectorMatrix{T}) where {T}
    nblocks = length(m.sectors)
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
function FusedSectorMatrix(a::AbelianArray{T, 2}) where {T}
    row_axis = a.axes[1]
    col_axis = a.axes[2]

    # Build lookup: for each col block, map its "matching label" to its position.
    # The matching label is the label that a row sector must have to form zero flux.
    # Zero flux: row_sector ⊗ flip(col_sector) = trivial
    # For abelian: flip(SectorRange(l, d)) = SectorRange(dual(l), !d)
    # So row label must equal dual(col_label) (accounting for duality).
    col_ea = eachblockaxis(col_axis)
    col_lookup = Dict{eltype(labels(col_axis)), Int}()
    for (j, si) in enumerate(col_ea)
        # The row label that matches: if col is dual, match label directly;
        # if col is non-dual, match dual(label).
        match_label = isdual(col_axis) ? label(si) : dual(label(si))
        col_lookup[match_label] = j
    end

    row_ea = eachblockaxis(row_axis)
    I = eltype(labels(row_axis))
    sector_labels = I[]
    diag_blocks = Matrix{T}[]

    for (i, si) in enumerate(row_ea)
        l = label(si)
        j = get(col_lookup, l, nothing)
        row_len = blocklengths(row_axis)[i]
        if isnothing(j)
            # No matching column sector — zero-sized column
            push!(sector_labels, l)
            push!(diag_blocks, zeros(T, row_len, 0))
        else
            col_len = blocklengths(col_axis)[j]
            if haskey(a.blockdata, (i, j))
                push!(sector_labels, l)
                push!(diag_blocks, a.blockdata[(i, j)])
            else
                push!(sector_labels, l)
                push!(diag_blocks, zeros(T, row_len, col_len))
            end
        end
    end

    return FusedSectorMatrix(sector_labels, diag_blocks)
end
