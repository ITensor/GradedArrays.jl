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

BlockArrays.blocklength(m::FusedGradedMatrix) = length(m.blocks)
BlockArrays.blocklength(m::FusedGradedMatrix, dim::Integer) =
    dim == 1 ? length(m.codomain) : dim == 2 ? length(m.domain) : throw(BoundsError(m, dim))

function BlockSparseArrays.blocktype(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S}
    return SectorMatrix{T, D, S}
end
BlockSparseArrays.blocktype(m::FusedGradedMatrix) = BlockSparseArrays.blocktype(typeof(m))
sectortype(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S} = S
datatype(::Type{<:FusedGradedMatrix{T, D, S}}) where {T, D, S} = D
datatype(a::FusedGradedMatrix) = datatype(typeof(a))

function Base.axes(m::FusedGradedMatrix)
    cod = gradedrange(collect(pairs(m.codomain)))
    dom = gradedrange([dual(s) => l for (s, l) in pairs(m.domain)])
    return (cod, dom)
end

Base.size(m::FusedGradedMatrix) = map(length, axes(m))
Base.eltype(::Type{FusedGradedMatrix{T}}) where {T} = T
Base.eltype(::Type{<:FusedGradedMatrix{T}}) where {T} = T

# ========================  Block indexing (primitive)  ========================

function Base.view(m::FusedGradedMatrix, I::Block{2})
    i, j = Int.(Tuple(I))
    @boundscheck begin
        i in 1:length(m.codomain) && j in 1:length(m.domain) ||
            throw(BoundsError(m, I))
    end
    s_cod = gettokenvalue(keys(m.codomain), i)
    s_dom = gettokenvalue(keys(m.domain), j)
    s_cod == s_dom ||
        error("Off-diagonal access not supported for block-sparse FusedGradedMatrix")
    return SectorMatrix(s_cod, m.blocks[s_cod])
end

# ========================  eachblockstoredindex  ========================

function BlockSparseArrays.eachblockstoredindex(m::FusedGradedMatrix)
    return (Block(gettoken(m.codomain, c), gettoken(m.domain, c)) for c in keys(m.blocks))
end

# ========================  blocks  ========================

function BlockArrays.blocks(m::FusedGradedMatrix)
    return [view(m, I) for I in eachblockstoredindex(m)]
end

# ========================  fill! / zero!  ========================

function FI.zero!(m::FusedGradedMatrix)
    for b in values(m.blocks)
        fill!(b, zero(eltype(b)))
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
    for (s, c) in pairs(C.blocks)
        if haskey(A.blocks, c) && haskey(B.blocks, c)
            mul!(c, A.blocks[s], B.blocks[s], α, β)
        else
            iszero(β) ? fill!(c, β) : scale!(c, β)
        end
    end
    return C
end

function allocate_output(::typeof(*), A::FusedGradedMatrix, B::FusedGradedMatrix)
    cod = A.codomain
    dom = B.domain
    S = sectortype(typeof(A))
    DA = datatype(BlockSparseArrays.blocktype(A))
    DB = datatype(BlockSparseArrays.blocktype(B))
    Dout = Base.promote_op(*, DA, DB)
    Dout′ = isconcretetype(Dout) ? Dout : Matrix{eltype(Dout)}

    return FusedGradedMatrix{eltype(Dout′), Dout′, S}(undef, cod, dom)
end

function Base.:(*)(A::FusedGradedMatrix, B::FusedGradedMatrix)
    check_input(*, A, B)
    C = allocate_output(*, A, B)
    return mul!(C, A, B)
end

# ======================== LinearAlgebra ======================

function Base.adjoint(A::FusedGradedMatrix)
    new_blocks = map(adjoint, A.blocks)
    return FusedGradedMatrix(A.domain, A.codomain, new_blocks)
end
# note: not defining transpose here since that has requirements on sectors

function LinearAlgebra.norm(A::FusedGradedMatrix, p::Real = 2)
    if p == Inf
        isempty(A.blocks) && return zero(float(real(eltype(A))))
        return maximum(Base.Fix2(LinearAlgebra.norm, p), values(A.blocks))
    elseif p > 0
        s = zero(float(real(eltype(A))))
        for (c, a) in pairs(A.blocks)
            s += length(c) * LinearAlgebra.norm(a, p)^p
        end
        return s^inv(p)
    else
        throw(ArgumentError("Norm with non-positive p ($p) is not defined"))
    end
end

LinearAlgebra.istriu(A::FusedGradedMatrix) = all(LinearAlgebra.istriu, values(A.blocks))
LinearAlgebra.istril(A::FusedGradedMatrix) = all(LinearAlgebra.istril, values(A.blocks))
LinearAlgebra.isposdef(A::FusedGradedMatrix) = all(LinearAlgebra.isposdef, values(A.blocks))

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    new_blocks = map(b -> similar(b, T), m.blocks)
    return FusedGradedMatrix(m.codomain, m.domain, new_blocks)
end

# ========================  show  ========================

function Base.summary(io::IO, m::FusedGradedMatrix)
    print(
        io, length(m.codomain), "×", length(m.domain), " ", typeof(m),
        " with ", length(m.blocks), " stored block",
        length(m.blocks) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(m.blocks), ", ")
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
    isempty(m.blocks) && return nothing
    Base.print_array(io, m)
    return nothing
end

function Base.show(io::IO, m::FusedGradedMatrix)
    print(
        io, length(m.codomain), "×", length(m.domain), " ", typeof(m),
        " (", length(m.blocks), " stored)"
    )
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
