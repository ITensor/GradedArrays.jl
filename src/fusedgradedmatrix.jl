# ===========================================================================
#  FusedGradedMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedGradedMatrix{T,D<:AbstractMatrix{T},S<:SectorRange}

Block-diagonal matrix produced by matricizing an `AbstractGradedArray`.
Each stored block corresponds to a coupled sector that lives on both the codomain and the domain.

Fields:

  - `codomain::Dictionary{S,Int}` — codomain (row) axis, mapping each sector to
    its row-block size. Keys are sorted and unique. Sectors are stored
    non-dual (codomain convention).
  - `domain::Dictionary{S,Int}` — domain (column) axis, mapping each sector to
    its column-block size. Keys are sorted and unique. Stored non-dual; the
    actual axis is dual (the keys are dualed by `axes(m, 2)`).
  - `blocks::Dictionary{S,D}` — stored data blocks, keyed by sector. Each key
    must be in both `codomain` and `domain`, and `size(blocks[s])` must equal
    `(codomain[s], domain[s])`.
"""
struct FusedGradedMatrix{T, D <: AbstractMatrix{T}, S <: SectorRange} <:
    AbstractGradedMatrix{T}
    codomain::Dictionary{S, Int}
    domain::Dictionary{S, Int}
    blocks::Dictionary{S, D}

    # Undef constructor
    function FusedGradedMatrix{T, D, S}(
            ::UndefInitializer, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int},
        ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
        issorted(keys(codomain)) || throw(ArgumentError("codomain sectors must be sorted"))
        issorted(keys(domain)) || throw(ArgumentError("domain sectors must be sorted"))

        blocksectors = intersect(keys(codomain), keys(domain))
        blocks = dictionary(c => similar(D, (Base.OneTo(codomain[c]), Base.OneTo(domain[c]))) for c in blocksectors)

        return new{T, D, S}(codomain, domain, blocks)
    end

    # Data constructor
    function FusedGradedMatrix{T, D, S}(
            codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}, blocks::Dictionary{S, D}
        ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
        issorted(keys(codomain)) || throw(ArgumentError("codomain sectors must be sorted"))
        issorted(keys(domain)) || throw(ArgumentError("domain sectors must be sorted"))

        blocksectors = intersect(keys(codomain), keys(domain))
        issetequal(blocksectors, keys(blocks)) || throw(ArgumentError("invalid blocks"))
        for (c, b) in pairs(blocks)
            size(b) == (codomain[c], domain[c]) || throw(DimensionMismatch("invalid block for sector $c"))
        end

        return new{T, D, S}(codomain, domain, blocks)
    end
end

function FusedGradedMatrix(
        codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}, blocks::Dictionary{S, D},
    ) where {S <: SectorRange, D <: AbstractMatrix}
    return FusedGradedMatrix{eltype(D), D, S}(codomain, domain, blocks)
end

"""
    FusedGradedMatrix(sectors::Vector{S}, blocks::Vector{D})

Build a `FusedGradedMatrix` whose codomain and domain carry the same sector list.
`codomain[sectors[i]]` is `size(blocks[i], 1)` and `domain[sectors[i]]` is `size(blocks[i], 2)`.
"""
function FusedGradedMatrix(
        sectors::AbstractVector{S},
        blocks::AbstractVector{D},
    ) where {S <: SectorRange, D <: AbstractMatrix}
    length(sectors) == length(blocks) ||
        throw(ArgumentError("sectors and blocks must have the same length"))
    issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    cod = Dictionary{S, Int}(sectors, [size(b, 1) for b in blocks])
    dom = Dictionary{S, Int}(sectors, [size(b, 2) for b in blocks])
    blks = Dictionary{S, D}(sectors, collect(blocks))
    return FusedGradedMatrix(cod, dom, blks)
end

FusedGradedMatrix(pairs::AbstractVector{<:Pair{<:SectorRange}}) =
    FusedGradedMatrix(first.(pairs), last.(pairs))

# ========================  undef constructors  ========================

# Symmetric `undef` constructor: same sector list on both axes.
function FusedGradedMatrix{T, D, S}(
        ::UndefInitializer,
        sectors::AbstractVector{S},
        axes::Tuple{BlockedOneTo, BlockedOneTo}
    ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
    length(cod_axes) == length(dom_axes) == length(sectors) ||
        throw(ArgumentError("axes block counts must match sectors length"))
    issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    cod = Dictionary{S, Int}(sectors, map(length, eachblockaxis(axes[1])))
    dom = Dictionary{S, Int}(sectors, map(length, eachblockaxis(axes[2])))
    return FusedGradedMatrix{T, D, S}(undef, cod, dom)
end

# Convenience: default D = Matrix{T}.
function FusedGradedMatrix{T}(
        ::UndefInitializer, sectors::AbstractVector{<:SectorRange}, axes::Tuple{BlockedOneTo, BlockedOneTo}
    ) where {T}
    S = eltype(sectors)
    return FusedGradedMatrix{T, Matrix{T}, S}(undef, sectors, axes)
end

# Vector{Int} convenience: wraps into BlockedOneTo and delegates.
function FusedGradedMatrix{T}(
        ::UndefInitializer,
        sectors::AbstractVector{<:SectorRange},
        codomain_blocklengths::AbstractVector{Int},
        domain_blocklengths::AbstractVector{Int}
    ) where {T}
    S = eltype(sectors)
    cod = Dictionary{S, Int}(sectors, codomain_blocklengths)
    dom = Dictionary{S, Int}(sectors, domain_blocklengths)
    return FusedGradedMatrix{T}(undef, cod, dom)
end

function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int},
    ) where {T, S <: SectorRange}
    return FusedGradedMatrix{T, Matrix{T}, S}(undef, codomain, domain)
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
    check_input(*, A, B)
    axes(C, 1) == axes(A, 1) || throw(DimensionMismatch())
    axes(C, 2) == axes(B, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        C::FusedGradedMatrix, A::FusedGradedMatrix, B::FusedGradedMatrix,
        α::Number, β::Number
    )
    check_input(mul!, C, A, B)
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
    check_input(*, A, B)
    C = allocate_output(*, A, B)
    return mul!(C, A, B)
end

# ======================== LinearAlgebra ======================

Base.adjoint(A::FusedGradedMatrix) = FusedGradedMatrix(A.sectors, map(adjoint, A.blocks))
# note: not defining transpose here since that has requirements on sectors

function LinearAlgebra.norm(A::FusedGradedMatrix, p::Real = 2)
    if p == Inf
        return maximum(Base.Fix2(LinearAlgebra.norm, p), A.blocks)
    elseif p > 0
        s = zero(float(real(eltype(A))))
        for (c, a) in zip(A.sectors, A.blocks)
            s += length(c) * LinearAlgebra.norm(a, p)^p
        end
        return s^inv(p)
    else
        throw(ArgumentError("Norm with non-positive p ($p) is not defined"))
    end
end

LinearAlgebra.istriu(A::FusedGradedMatrix) = all(LinearAlgebra.istriu, A.blocks)
LinearAlgebra.istril(A::FusedGradedMatrix) = all(LinearAlgebra.istril, A.blocks)
LinearAlgebra.isposdef(A::FusedGradedMatrix) = all(LinearAlgebra.isposdef, A.blocks)

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
