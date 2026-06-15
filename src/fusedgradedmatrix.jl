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
            ::UndefInitializer, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}
        ) where {T, D <: AbstractMatrix{T}, S <: SectorRange}
        issorted(keys(codomain)) || throw(ArgumentError("codomain sectors must be sorted"))
        issorted(keys(domain)) || throw(ArgumentError("domain sectors must be sorted"))

        blocksectors = intersect(keys(codomain), keys(domain))
        blocks = dictionary(
            c => similar(D, (Base.OneTo(codomain[c]), Base.OneTo(domain[c]))) for
                c in blocksectors
        )

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
            size(b) == (codomain[c], domain[c]) ||
                throw(DimensionMismatch("invalid block for sector $c"))
        end

        return new{T, D, S}(codomain, domain, blocks)
    end
end

function FusedGradedMatrix(
        codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}, blocks::Dictionary{S, D}
    ) where {S <: SectorRange, D <: AbstractMatrix}
    return FusedGradedMatrix{eltype(D), D, S}(codomain, domain, blocks)
end

# Block-diagonal by construction (one block per sector), so just check each block.
LinearAlgebra.isdiag(A::FusedGradedMatrix) = all(LinearAlgebra.isdiag, A.blocks)

# Block-diagonal by construction, so any matrix function `f(A) = blkdiag(f(blk_i))` for
# each stored block — covers `sqrt`, `exp`, `log`, etc. Routes around the generic
# `LinearAlgebra` impls that scalar-index for triangular / Hermitian detection.
# Per-block result eltypes may differ (e.g. `sqrt(::Matrix{Float64})` returns
# `Matrix{ComplexF64}` via Schur even when each block is real-PSD), so unify to the
# `promote_type` of all returned blocks before reconstructing.
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval function Base.$f(A::FusedGradedMatrix)
        raw = map(Base.$f, A.blocks)
        T = mapreduce(eltype, promote_type, raw; init = eltype(A))
        blocks = map(b -> eltype(b) === T ? b : convert(AbstractMatrix{T}, b), raw)
        return FusedGradedMatrix(A.codomain, A.domain, blocks)
    end
end

"""
    FusedGradedMatrix(sectors::Vector{S}, blocks::Vector{D})

Build a `FusedGradedMatrix` whose codomain and domain carry the same sector list.
`codomain[sectors[i]]` is `size(blocks[i], 1)` and `domain[sectors[i]]` is `size(blocks[i], 2)`.
"""
function FusedGradedMatrix(
        sectors::AbstractVector{S},
        blocks::AbstractVector{D}
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

function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}
    ) where {T, S <: SectorRange}
    return FusedGradedMatrix{T, Matrix{T}, S}(undef, codomain, domain)
end

# ========================  Accessors  ========================

BlockArrays.blocklength(m::FusedGradedMatrix) = length(m.blocks)
function BlockArrays.blocklength(m::FusedGradedMatrix, dim::Integer)
    return if dim == 1
        length(m.codomain)
    elseif dim == 2
        length(m.domain)
    else
        throw(BoundsError(m, dim))
    end
end

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
    return (
        Block(gettoken(m.codomain, c)[2][2], gettoken(m.domain, c)[2][2]) for
            c in keys(m.blocks)
    )
end

# ========================  blocks  ========================

function BlockArrays.blocks(m::FusedGradedMatrix)
    return [view(m, I) for I in eachblockstoredindex(m)]
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
        if haskey(A.blocks, s) && haskey(B.blocks, s)
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

# ========================  Block-wise +, - ========================
#
# FusedGradedMatrix has no `BroadcastStyle`, so the AbstractArray fallback for `+`/`-`
# (`broadcast_preserving_zero_d`) ends up trying scalar indexing on a sector-block
# structure — which silently produces wrong results. Define the operations directly
# block-by-block, taking the union of sectors so an absent block on one side is
# treated as zero.

function _check_add_axes(A::FusedGradedMatrix, B::FusedGradedMatrix, op)
    A.codomain == B.codomain ||
        throw(DimensionMismatch("$(op): codomain mismatch"))
    A.domain == B.domain ||
        throw(DimensionMismatch("$(op): domain mismatch"))
    return nothing
end

function _block_combine(op, A::FusedGradedMatrix, B::FusedGradedMatrix)
    T = promote_type(eltype(A), eltype(B))
    DA = datatype(BlockSparseArrays.blocktype(A))
    DB = datatype(BlockSparseArrays.blocktype(B))
    D = Base.promote_op(op, DA, DB)
    D′ = isconcretetype(D) ? D : Matrix{T}
    S = sectortype(typeof(A))
    sectors = sort!(collect(union(keys(A.blocks), keys(B.blocks))))
    blocks = Dictionary{S, D′}()
    for c in sectors
        rows = get(A.codomain, c, get(B.codomain, c, 0))
        cols = get(A.domain, c, get(B.domain, c, 0))
        a = get(() -> zeros(eltype(A), rows, cols), A.blocks, c)
        b = get(() -> zeros(eltype(B), rows, cols), B.blocks, c)
        insert!(blocks, c, op(a, b))
    end
    return FusedGradedMatrix(A.codomain, A.domain, blocks)
end

function Base.:(+)(A::FusedGradedMatrix, B::FusedGradedMatrix)
    _check_add_axes(A, B, :+)
    return _block_combine(+, A, B)
end

function Base.:(-)(A::FusedGradedMatrix, B::FusedGradedMatrix)
    _check_add_axes(A, B, :-)
    return _block_combine(-, A, B)
end

# TODO: these explicit scalar-op methods exist only because broadcasting is
# disabled for `FusedGradedMatrix`. Once structure-preserving broadcasting is
# supported, drop them and let Base's `AbstractArray`-scalar `*` / `/` forward to
# broadcasting (as `AbelianGradedArray` already does). See the
# `fusedgradedmatrix_broadcasting` project.
function Base.:(*)(A::FusedGradedMatrix, x::Number)
    new_blocks = map(b -> b * x, A.blocks)
    return FusedGradedMatrix(A.codomain, A.domain, new_blocks)
end
Base.:(*)(x::Number, A::FusedGradedMatrix) = A * x
function Base.:(/)(A::FusedGradedMatrix, x::Number)
    new_blocks = map(b -> b / x, A.blocks)
    return FusedGradedMatrix(A.codomain, A.domain, new_blocks)
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

function LinearAlgebra.dot(A::FusedGradedMatrix, B::FusedGradedMatrix)
    A.codomain == B.codomain ||
        throw(DimensionMismatch("dot: codomain mismatch"))
    A.domain == B.domain ||
        throw(DimensionMismatch("dot: domain mismatch"))
    T = promote_type(eltype(A), eltype(B))
    s = zero(T)
    for c in intersect(keys(A.blocks), keys(B.blocks))
        s += length(c) * LinearAlgebra.dot(A.blocks[c], B.blocks[c])
    end
    return s
end

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    new_blocks = map(b -> similar(b, T), m.blocks)
    return FusedGradedMatrix(m.codomain, m.domain, new_blocks)
end
function Base.similar(
        m::FusedGradedMatrix,
        codomain::Dictionary{S, Int},
        domain::Dictionary{S, Int}
    ) where {S}
    return typeof(m)(undef, codomain, domain)
end
function Base.similar(
        m::FusedGradedMatrix,
        ::Type{T},
        codomain::Dictionary{S, Int},
        domain::Dictionary{S, Int}
    ) where {T, S}
    if T <: Number
        return FusedGradedMatrix{T}(undef, codomain, domain)
    elseif T <: AbstractMatrix
        return FusedGradedMatrix{eltype(T), T, S}(undef, codomain, domain)
    else
        throw(ArgumentError("invalid type $T"))
    end
end
function Base.similar(
        m::FusedGradedMatrix,
        ::Type{T},
        axis::Dictionary{S, Int}
    ) where {T <: AbstractVector, S}
    return FusedGradedVector{eltype(T), T, S}(undef, axis)
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

"""
    FusedGradedMatrix(a::AbelianGradedMatrix{T})

Convert a 2D block-sparse `AbelianGradedArray` (as produced by `matricize`)
into a `FusedGradedMatrix`. The codomain dict comes from the row axis sectors
and lengths; the domain dict comes from `dual.(domain_axis_sectors)` and
lengths. Stored entries of `a` populate `blocks`.
"""
function FusedGradedMatrix(a::AbelianGradedMatrix{T}) where {T}
    S = sectortype(a)
    cod_sectors = sectors(axes(a, 1))
    issorted(cod_sectors) ||
        throw(ArgumentError("codomain sectors of input must be sorted"))
    allunique(cod_sectors) ||
        throw(ArgumentError("codomain sectors of input must be unique"))
    cod = Dictionary{S, Int}(cod_sectors, datalengths(axes(a, 1)))

    dom_sectors = sectors(axes(a, 2))
    issorted(dom_sectors) ||
        throw(ArgumentError("domain sectors of input must have sorted, unique duals"))
    allunique(dom_sectors) ||
        throw(ArgumentError("domain sectors of input must have unique duals"))
    dom = Dictionary{S, Int}(dual.(dom_sectors), datalengths(axes(a, 2)))

    m = FusedGradedMatrix{T, datatype(a), S}(undef, cod, dom)
    for I in eachblockstoredindex(a)
        view(m, I) .= view(a, I)
    end
    return m
end
