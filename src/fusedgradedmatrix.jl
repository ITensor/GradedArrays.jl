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
function FusedGradedMatrix(pairs::AbstractVector{<:Pair})
    sectors = first.(pairs)
    blocks = last.(pairs)
    return FusedGradedMatrix(sectors, blocks)
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

function TensorAlgebra.check_input(::typeof(*), A::FusedGradedMatrix, B::FusedGradedMatrix)
    axes(A, 2) == dual(axes(B, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    return nothing
end

function TensorAlgebra.check_input(
        ::typeof(mul!),
        C::FusedGradedMatrix, A::FusedGradedMatrix, B::FusedGradedMatrix
    )
    TensorAlgebra.check_input(*, A, B)
    axes(C, 1) == axes(A, 1) || throw(DimensionMismatch())
    axes(C, 2) == axes(B, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        C::FusedGradedMatrix, A::FusedGradedMatrix, B::FusedGradedMatrix,
        α::Number, β::Number
    )
    TensorAlgebra.check_input(mul!, C, A, B)
    for I in blockdiagindices(C)
        mul!(view(C, Data(I)), view(A, Data(I)), view(B, Data(I)), α, β)
    end
    return C
end

function allocate_output(::typeof(*), A::FusedGradedMatrix, B::FusedGradedMatrix)
    cod_axes = eachdataaxis(axes(A, 1))
    dom_axes = eachdataaxis(axes(B, 2))
    result_blocks = [
        similar(
                Base.promote_op(*, typeof(view(A, Data(I))), typeof(view(B, Data(I)))),
                (cod_axes[Int(Tuple(I)[1])], dom_axes[Int(Tuple(I)[2])])
            ) for I in blockdiagindices(A)
    ]
    return FusedGradedMatrix(copy(A.sectors), result_blocks)
end

function Base.:(*)(A::FusedGradedMatrix, B::FusedGradedMatrix)
    TensorAlgebra.check_input(*, A, B)
    C = allocate_output(*, A, B)
    return mul!(C, A, B)
end

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    new_blocks = [similar(b, T) for b in m.blocks]
    return FusedGradedMatrix(copy(m.sectors), new_blocks)
end

# ========================  show  ========================

function Base.summary(io::IO, m::FusedGradedMatrix)
    nblocks = length(m.sectors)
    print(io, nblocks, "-block ", typeof(m), " with sectors [")
    join(io, m.sectors, ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::FusedGradedMatrix)
    summary(io, m)
    println(io, ":")
    for (d, g) in pairs(axes(m))
        print(io, "  Dim $d: ")
        show(io, g)
        println(io)
    end
    isempty(m.sectors) && return nothing
    Base.print_array(io, m)
    return nothing
end

function Base.show(io::IO, m::FusedGradedMatrix)
    nblocks = length(m.sectors)
    print(io, nblocks, "-block ", typeof(m))
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
    sectors(axes(a, 1)) == dual.(sectors(axes(a, 2))) || throw(
        ArgumentError(
            "AbelianGradedMatrix axes must be canonical duals to convert to FusedGradedMatrix"
        )
    )
    fused_sectors = collect(sectors(axes(a, 1)))
    fused_axes = blockedrange.(datalengths.(axes(a)))
    m = FusedGradedMatrix{T}(undef, fused_sectors, fused_axes)
    for I in blockdiagindices(m)
        m[Data(I)] = view(a, Data(I))
    end
    return m
end
