# ===========================================================================
#  FusedGradedMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedGradedMatrix{T,S<:SectorRange,D<:AbstractMatrix{T}}

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
struct FusedGradedMatrix{T, S <: SectorRange, D <: AbstractMatrix{T}} <:
    AbstractGradedMatrix{T, S}
    codomain::Dictionary{S, Int}
    domain::Dictionary{S, Int}
    blocks::Dictionary{S, D}

    # Undef constructor
    function FusedGradedMatrix{T, S, D}(
            ::UndefInitializer, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}
        ) where {T, S <: SectorRange, D <: AbstractMatrix{T}}
        issorted(keys(codomain)) || throw(ArgumentError("codomain sectors must be sorted"))
        issorted(keys(domain)) || throw(ArgumentError("domain sectors must be sorted"))

        blocksectors = intersect(keys(codomain), keys(domain))
        blocks = dictionary(
            c => similar(D, (Base.OneTo(codomain[c]), Base.OneTo(domain[c]))) for
                c in blocksectors
        )

        return new{T, S, D}(codomain, domain, blocks)
    end

    # Data constructor
    function FusedGradedMatrix{T, S, D}(
            codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}, blocks::Dictionary{S, D}
        ) where {T, S <: SectorRange, D <: AbstractMatrix{T}}
        issorted(keys(codomain)) || throw(ArgumentError("codomain sectors must be sorted"))
        issorted(keys(domain)) || throw(ArgumentError("domain sectors must be sorted"))

        blocksectors = intersect(keys(codomain), keys(domain))
        issetequal(blocksectors, keys(blocks)) || throw(ArgumentError("invalid blocks"))
        for (c, b) in pairs(blocks)
            size(b) == (codomain[c], domain[c]) ||
                throw(DimensionMismatch("invalid block for sector $c"))
        end

        return new{T, S, D}(codomain, domain, blocks)
    end
end

function FusedGradedMatrix(
        codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}, blocks::Dictionary{S, D}
    ) where {S <: SectorRange, D <: AbstractMatrix}
    return FusedGradedMatrix{eltype(D), S, D}(codomain, domain, blocks)
end

# Block-diagonal by construction (one block per sector), so just check each block.
LinearAlgebra.isdiag(A::FusedGradedMatrix) = all(LinearAlgebra.isdiag, A.blocks)

# Blockwise copy: the generic `AbstractArray` fallback copies elementwise, which
# scalar-indexes (disallowed for graded arrays).
function Base.copy(A::FusedGradedMatrix)
    return FusedGradedMatrix(A.codomain, A.domain, map(copy, A.blocks))
end

# Materialize into a dense `Array` (the generic fallback copies elementwise, which
# scalar-indexes). `_to_blockarray` reintroduces each block's structural factor
# (`SectorIdentity`, i.e. `I ⊗ reduced`), which is the identity for abelian sectors but
# repeats the reduced block over the irrep's quantum dimension for non-abelian ones.
Base.Array(a::FusedGradedMatrix) = Array(_to_blockarray(a))

# Block-diagonal by construction, so any matrix function `f(A) = blkdiag(f(blk_i))` for
# each stored block — covers `sqrt`, `exp`, `log`, etc. Routes around the generic
# `LinearAlgebra` impls that scalar-index for triangular / Hermitian detection.
# Per-block result eltypes may differ (e.g. `sqrt(::Matrix{Float64})` returns
# `Matrix{ComplexF64}` via Schur even when each block is real-PSD), so unify to the
# `promote_type` of all returned blocks before reconstructing.
#
# The target eltype `T` is passed through a type-parameter barrier so the `convert`
# target is concrete to inference. Splicing a runtime `T` straight into
# `convert(AbstractMatrix{T}, b)` makes older Julia widen the block dictionary to an
# abstract `AbstractMatrix`, and the reconstruction then throws a `TypeError`.
function unify_block_eltype(blocks, ::Type{T}) where {T}
    return map(b -> convert(AbstractMatrix{T}, b), blocks)
end

for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval function Base.$f(A::FusedGradedMatrix)
        raw = map(Base.$f, A.blocks)
        T = mapreduce(eltype, promote_type, raw; init = eltype(A))
        return FusedGradedMatrix(A.codomain, A.domain, unify_block_eltype(raw, T))
    end
end

"""
    FusedGradedMatrix(sectors::Vector{S}, blocks::Vector{D})

Build a `FusedGradedMatrix` whose codomain and domain carry the same sector list.
`codomain[sectors[i]]` is `size(blocks[i], 1)` and `domain[sectors[i]]` is `size(blocks[i], 2)`.
"""
function FusedGradedMatrix(
        sectors::AbstractVector,
        blocks::AbstractVector{D}
    ) where {D <: AbstractMatrix}
    length(sectors) == length(blocks) ||
        throw(ArgumentError("sectors and blocks must have the same length"))
    # Accept bare `TKS.Sector`s (e.g. `FermionNumber(1)`) alongside `SectorRange`s, as
    # `gradedrange` does; `SectorRange` wraps the former and is the identity on the latter.
    rs = map(SectorRange, sectors)
    issorted(rs) || throw(ArgumentError("sectors must be sorted"))
    allunique(rs) || throw(ArgumentError("sectors must be unique"))
    S = eltype(rs)
    cod = Dictionary{S, Int}(rs, [size(b, 1) for b in blocks])
    dom = Dictionary{S, Int}(rs, [size(b, 2) for b in blocks])
    blks = Dictionary{S, D}(rs, collect(blocks))
    return FusedGradedMatrix(cod, dom, blks)
end

function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}
    ) where {T, S <: SectorRange}
    return FusedGradedMatrix{T, S, Matrix{T}}(undef, codomain, domain)
end

"""
    FusedGradedMatrix{T}(undef, sectors, rowlengths, collengths)
    FusedGradedMatrix{T}(undef, sectors .=> rowlengths, sectors .=> collengths)
    FusedGradedMatrix{T}(undef, sectors .=> lengths)

Allocate a block-diagonal `FusedGradedMatrix` with uninitialized blocks keyed by a shared set of
`sectors`. `rowlengths[i]`/`collengths[i]` give the reduced row and column lengths of the block at
`sectors[i]`. The pairs forms mirror the `dictionary(pairs)` constructor from `Dictionaries`; the
single-argument pairs form sets the domain equal to the codomain (square blocks). Bare `TKS.Sector`s
are accepted alongside `SectorRange`s. Pair with `randn!`/`rand!` to fill.
"""
function FusedGradedMatrix{T}(
        ::UndefInitializer,
        sectors::AbstractVector, rowlengths::AbstractVector, collengths::AbstractVector
    ) where {T}
    rs = map(SectorRange, sectors)
    S = eltype(rs)
    codomain = Dictionary{S, Int}(rs, collect(Int, rowlengths))
    domain = Dictionary{S, Int}(rs, collect(Int, collengths))
    return FusedGradedMatrix{T}(undef, codomain, domain)
end
function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::AbstractVector{<:Pair}, domain::AbstractVector{<:Pair}
    ) where {T}
    map(SectorRange, first.(codomain)) == map(SectorRange, first.(domain)) ||
        throw(ArgumentError("codomain and domain sectors must match"))
    return FusedGradedMatrix{T}(undef, first.(codomain), last.(codomain), last.(domain))
end
function FusedGradedMatrix{T}(::UndefInitializer, blocks::AbstractVector{<:Pair}) where {T}
    return FusedGradedMatrix{T}(undef, blocks, blocks)
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

function blocktype(::Type{<:FusedGradedMatrix{T, S, D}}) where {T, S, D}
    return SectorMatrix{T, S, D}
end
blocktype(m::FusedGradedMatrix) = blocktype(typeof(m))

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

function eachblockstoredindex(m::FusedGradedMatrix)
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
    DA = datatype(A)
    DB = datatype(B)
    Dout = Base.promote_op(*, DA, DB)
    Dout′ = isconcretetype(Dout) ? Dout : Matrix{eltype(Dout)}

    return FusedGradedMatrix{eltype(Dout′), S, Dout′}(undef, cod, dom)
end

function Base.:(*)(A::FusedGradedMatrix, B::FusedGradedMatrix)
    check_input(*, A, B)
    C = allocate_output(*, A, B)
    return mul!(C, A, B)
end

# ========================  lmul! / rmul! (matrix-matrix)  ========================
#
# MatrixAlgebraKit's SVD-based `left_orth!` / `right_orth!` fold the singular values into the
# orthogonal factor in place with `lmul!(S, C)` / `rmul!(C, S)`, where `S` is the (diagonal)
# singular-value matrix. The scalar-argument `lmul!` / `rmul!` in `abstractgradedarray.jl` do not
# cover this two-matrix form, so define it block-wise: each stored sector block delegates to the
# `LinearAlgebra` method for that block pair, an in-place row / column scaling for the diagonal
# `S` blocks the factorizations feed in. The `check_input(mul!, ...)` call validates the contracted
# axes and that the product fits the mutated operand (the operand plays the role of the `mul!`
# destination `C`: `B` for `lmul!`, `A` for `rmul!`), so the block sectors line up by construction.
function LinearAlgebra.lmul!(A::FusedGradedMatrix, B::FusedGradedMatrix)
    check_input(mul!, B, A, B)
    for (s, b) in pairs(B.blocks)
        LinearAlgebra.lmul!(A.blocks[s], b)
    end
    return B
end
function LinearAlgebra.rmul!(A::FusedGradedMatrix, B::FusedGradedMatrix)
    check_input(mul!, A, A, B)
    for (s, a) in pairs(A.blocks)
        LinearAlgebra.rmul!(a, B.blocks[s])
    end
    return A
end

# ======================== LinearAlgebra ======================

function Base.adjoint(A::FusedGradedMatrix)
    new_blocks = map(adjoint, A.blocks)
    return FusedGradedMatrix(A.domain, A.codomain, new_blocks)
end
# note: not defining transpose here since that has requirements on sectors

# Route eager `conj` through the conjugating broadcast, matching `AbelianGradedArray`: `conj.`
# dualizes the axes (moving each block to its dual coupled sector) and carries the fermionic
# `twist`, handled in the `bipermutedimsopadd!` overload in `tensoralgebra.jl`. This also
# overrides Base's `conj(::AbstractArray{<:Real}) = A` short-circuit so a real-eltype fused matrix
# still dualizes its axes.
Base.conj(A::FusedGradedMatrix) = conj.(A)

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
        return FusedGradedMatrix{eltype(T), S, T}(undef, codomain, domain)
    else
        throw(ArgumentError("invalid type $T"))
    end
end
function Base.similar(
        m::FusedGradedMatrix,
        ::Type{T},
        axis::Dictionary{S, Int}
    ) where {T <: AbstractVector, S}
    return FusedGradedVector{eltype(T), S, T}(undef, axis)
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
    cod_sectors = eachsectoraxis(axes(a, 1))
    issorted(cod_sectors) ||
        throw(ArgumentError("codomain sectors of input must be sorted"))
    allunique(cod_sectors) ||
        throw(ArgumentError("codomain sectors of input must be unique"))
    cod = Dictionary{S, Int}(cod_sectors, datalengths(axes(a, 1)))

    dom_sectors = eachsectoraxis(axes(a, 2))
    issorted(dom_sectors) ||
        throw(ArgumentError("domain sectors of input must have sorted, unique duals"))
    allunique(dom_sectors) ||
        throw(ArgumentError("domain sectors of input must have unique duals"))
    dom = Dictionary{S, Int}(dual.(dom_sectors), datalengths(axes(a, 2)))

    m = FusedGradedMatrix{T, S, datatype(a)}(undef, cod, dom)
    for I in eachblockstoredindex(a)
        copy!(view(m, I), view(a, I))
    end
    return m
end
