# ===========================================================================
#  FusedGradedMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedGradedMatrix{T,S<:SectorRange,D<:AbstractMatrix{T},V<:DenseVector{T}}

Block-diagonal matrix produced by matricizing a `FusionArray`. Each stored block corresponds to a
coupled sector that lives on both the codomain and the domain.
"""
struct FusedGradedMatrix{
        T,
        S <: SectorRange,
        D <: AbstractMatrix{T},
        V <: DenseVector{T},
    } <:
    AbstractFusedGradedMatrix{T, S}
    buffer::V
    sectordata::Dictionary{S, D}
    axis_codomain::FusedGradedOneTo{S}
    axis_domain::FusedGradedOneTo{S}

    # Primitive constructor: wrap a contiguous buffer already in TensorKit `.data` layout. `blocks`
    # are carved from `data` as reshaped views, so this shares (does not copy) the buffer. The axes
    # are non-dual `FusedGradedOneTo`s (their constructor enforces sorted, non-dual sectors); the
    # domain's dual arrow is implicit in `axes` (see `biaxes`).
    function FusedGradedMatrix{T, S, D, V}(
            data::V, codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
        ) where {T, S <: SectorRange, D <: AbstractMatrix{T}, V <: DenseVector{T}}
        (isdual(codomain) || isdual(domain)) && throw(
            ArgumentError(
                "FusedGradedMatrix stores non-dual codomain/domain axes; the domain's dual arrow is implicit in `axes` (see `biaxes`)"
            )
        )
        blocks =
            carve_blocks(D, data, sectordatalengths(codomain), sectordatalengths(domain))
        return new{T, S, D, V}(data, blocks, codomain, domain)
    end
end

# Carve a contiguous buffer into per-coupled-sector blocks as reshaped views, in sorted coupled-sector
# order and column-major within each block (TensorKit's `.data` layout). The block type `D` is fixed by
# the buffer type, so pass it explicitly to type the (possibly empty) block dictionary.
function carve_blocks(
        ::Type{D}, data, codomain::Dictionary{S, Int}, domain::Dictionary{S, Int}
    ) where {D, S}
    coupled = intersect(keys(codomain), keys(domain))
    total = sum(c -> codomain[c] * domain[c], coupled; init = 0)
    length(data) == total ||
        throw(
        DimensionMismatch(
            "buffer length $(length(data)) does not match block total $total"
        )
    )
    blocks = Dictionary{S, D}()
    offset = 0
    for c in coupled
        len = codomain[c] * domain[c]
        block = reshape(view(data, (offset + 1):(offset + len)), (codomain[c], domain[c]))
        insert!(blocks, c, block)
        offset += len
    end
    return blocks
end

# The reshaped-view block type is fixed by the buffer type `V`; derive it from a zero-length view so
# callers never spell it out.
blockviewtype(data::AbstractVector) = typeof(reshape(view(data, 1:0), (0, 0)))

# Primitive constructor deriving the block-view type from the buffer.
function FusedGradedMatrix(
        data::V, codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange, V <: DenseVector{T}}
    return FusedGradedMatrix{T, S, blockviewtype(data), V}(data, codomain, domain)
end

# Data constructor: allocate a fresh buffer and copy each given block into its view. Used by `copy`,
# `adjoint`, the matrix-function loop, and the vector-of-blocks form below, none of which need to
# share the passed blocks.
function FusedGradedMatrix(
        blocks::Dictionary{S, <:AbstractMatrix},
        codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {S <: SectorRange}
    cod, dom = sectordatalengths(codomain), sectordatalengths(domain)
    blocksectors = intersect(keys(cod), keys(dom))
    issetequal(blocksectors, keys(blocks)) || throw(ArgumentError("invalid blocks"))
    for (c, b) in pairs(blocks)
        size(b) == (cod[c], dom[c]) ||
            throw(DimensionMismatch("invalid block for sector $c"))
    end
    T = eltype(eltype(blocks))
    m = FusedGradedMatrix{T}(undef, codomain, domain)
    for (c, b) in pairs(blocks)
        copyto!(m.sectordata[c], b)
    end
    return m
end

# Block-diagonal by construction (one block per sector), so just check each block.
# Sum the per-block stored counts over the stored (symmetry-allowed) blocks; each block view is a
# `FusedSectorMatrix`, whose count folds in the sector's quantum dimension. The rest of `length` are
# structural zeros. Without this, the `AbstractArray` fallback reports `length` (i.e. fully dense).
function SparseArraysBase.storedlength(A::FusedGradedMatrix)
    return sum(B -> storedlength(view(A, B)), eachblockstoredindex(A); init = 0)
end

LinearAlgebra.isdiag(A::FusedGradedMatrix) = all(LinearAlgebra.isdiag, A.sectordata)

# Reductions over `Array(A)` without densifying, folding through the per-block `FusedSectorMatrix`
# reductions (each block already accounts for its quantum dimension and its within-block structural
# zeros). The remaining structural zeros are the off-sector (symmetry-forbidden) positions: for
# `maximum`/`minimum` a single `f(0)` folds them in when any is present (`length > storedlength`); for
# `sum` they would each add `f(0)`, so we restrict to zero-preserving `f` for now and reduce only the
# stored blocks. These return a scalar and take no keyword arguments (`dims` is not meaningful for a
# graded reduction, and `init` is just `x + sum(A)` at the call site).
Base.sum(A::FusedGradedMatrix) = sum(identity, A)
function Base.sum(f, A::FusedGradedMatrix)
    z = f(zero(eltype(A)))
    iszero(z) || throw_not_zero_preserving_sum(z)
    return sum(B -> sum(f, view(A, B)), eachblockstoredindex(A); init = z)
end

Base.maximum(A::FusedGradedMatrix) = maximum(identity, A)
function Base.maximum(f, A::FusedGradedMatrix)
    iszero(blockstoredlength(A)) && return f(zero(eltype(A)))
    m = maximum(B -> maximum(f, view(A, B)), eachblockstoredindex(A))
    return length(A) > storedlength(A) ? max(m, f(zero(eltype(A)))) : m
end

Base.minimum(A::FusedGradedMatrix) = minimum(identity, A)
function Base.minimum(f, A::FusedGradedMatrix)
    iszero(blockstoredlength(A)) && return f(zero(eltype(A)))
    m = minimum(B -> minimum(f, view(A, B)), eachblockstoredindex(A))
    return length(A) > storedlength(A) ? min(m, f(zero(eltype(A)))) : m
end

Base.extrema(A::FusedGradedMatrix) = extrema(identity, A)
Base.extrema(f, A::FusedGradedMatrix) = (minimum(f, A), maximum(f, A))

# Blockwise copy: the generic `AbstractArray` fallback copies elementwise, which
# scalar-indexes (disallowed for graded arrays).
function Base.copy(A::FusedGradedMatrix)
    return FusedGradedMatrix(map(copy, A.sectordata), A.axis_codomain, A.axis_domain)
end

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
        raw = map(Base.$f, A.sectordata)
        T = mapreduce(eltype, promote_type, raw; init = eltype(A))
        return FusedGradedMatrix(unify_block_eltype(raw, T), A.axis_codomain, A.axis_domain)
    end
end

"""
    FusedGradedMatrix(blocks::Vector{D}, sectors::Vector{S})

Build a `FusedGradedMatrix` whose codomain and domain carry the same sector list.
`codomain[sectors[i]]` is `size(blocks[i], 1)` and `domain[sectors[i]]` is `size(blocks[i], 2)`.
"""
function FusedGradedMatrix(
        blocks::AbstractVector{D},
        sectors::AbstractVector
    ) where {D <: AbstractMatrix}
    length(sectors) == length(blocks) ||
        throw(ArgumentError("sectors and blocks must have the same length"))
    # Accept bare `TKS.Sector`s (e.g. `FermionNumber(1)`) alongside `SectorRange`s, as
    # `gradedrange` does; `SectorRange` wraps the former and is the identity on the latter.
    rs = map(SectorRange, sectors)
    issorted(rs) || throw(ArgumentError("sectors must be sorted"))
    allunique(rs) || throw(ArgumentError("sectors must be unique"))
    S = eltype(rs)
    cod = FusedGradedOneTo(rs, [size(b, 1) for b in blocks])
    dom = FusedGradedOneTo(rs, [size(b, 2) for b in blocks])
    blks = Dictionary{S, D}(rs, collect(blocks))
    return FusedGradedMatrix(blks, cod, dom)
end

function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange}
    cod, dom = sectordatalengths(codomain), sectordatalengths(domain)
    coupled = intersect(keys(cod), keys(dom))
    total = sum(c -> cod[c] * dom[c], coupled; init = 0)
    return FusedGradedMatrix(Vector{T}(undef, total), codomain, domain)
end

# Build from the codomain and domain graded ranges, both in the stored (non-dual, fused) convention.
function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::GradedOneTo, domain::GradedOneTo
    ) where {T}
    return FusedGradedMatrix{T}(undef, FusedGradedOneTo(codomain), FusedGradedOneTo(domain))
end
# A single graded range sets the domain equal to the codomain (square blocks), mirroring the
# single-argument pairs form below.
function FusedGradedMatrix{T}(::UndefInitializer, codomain::GradedOneTo) where {T}
    return FusedGradedMatrix{T}(undef, codomain, codomain)
end
# Build from the axes as `axes(m)` returns them: `axes(m, 2)` dualizes the domain, so undo it.
function FusedGradedMatrix{T}(
        ::UndefInitializer, axs::Tuple{<:AbstractGradedOneTo, <:AbstractGradedOneTo}
    ) where {T}
    return FusedGradedMatrix{T}(
        undef,
        FusedGradedOneTo(axs[1]),
        FusedGradedOneTo(dual(axs[2]))
    )
end

"""
    FusedGradedMatrix{T}(undef, sectors, rowlengths, collengths)
    FusedGradedMatrix{T}(undef, sectors, lengths)
    FusedGradedMatrix{T}(undef, sectors .=> rowlengths, sectors .=> collengths)
    FusedGradedMatrix{T}(undef, sectors .=> lengths)
    FusedGradedMatrix{T}(undef, codomain::GradedOneTo, domain::GradedOneTo)
    FusedGradedMatrix{T}(undef, codomain::GradedOneTo)

Allocate a block-diagonal `FusedGradedMatrix` with uninitialized blocks keyed by a shared set of
`sectors`. `rowlengths[i]`/`collengths[i]` give the reduced row and column lengths of the block at
`sectors[i]`. The pairs forms mirror the `dictionary(pairs)` constructor from `Dictionaries`; the
forms taking a single `lengths` vector, single-argument pairs, or single-`GradedOneTo` set the domain
equal to the codomain (square blocks). Bare `TKS.Sector`s are accepted alongside `SectorRange`s. Pair
with `randn!`/`rand!` to fill.
"""
function FusedGradedMatrix{T}(
        ::UndefInitializer,
        sectors::AbstractVector, rowlengths::AbstractVector, collengths::AbstractVector
    ) where {T}
    rs = map(SectorRange, sectors)
    codomain = FusedGradedOneTo(rs, collect(Int, rowlengths))
    domain = FusedGradedOneTo(rs, collect(Int, collengths))
    return FusedGradedMatrix{T}(undef, codomain, domain)
end
# A single `lengths` vector sets the domain equal to the codomain (square blocks).
function FusedGradedMatrix{T}(
        ::UndefInitializer, sectors::AbstractVector, lengths::AbstractVector
    ) where {T}
    return FusedGradedMatrix{T}(undef, sectors, lengths, lengths)
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

# `blocklength(m)` / `blocksize(m)` / `blocksize(m, dim)` derive from `axes(m)` (BlockArrays), and the
# stored (block-diagonal) count is `blockstoredlength(m)`, so no custom overrides are needed here.

function blocktype(::Type{<:FusedGradedMatrix{T, S, D}}) where {T, S, D}
    return FusedSectorMatrix{T, S, D}
end
blocktype(m::FusedGradedMatrix) = blocktype(typeof(m))

sectordata(m::FusedGradedMatrix) = m.sectordata

# `biaxes` is the core axis accessor; `axes`, `axes_codomain`, `axis_codomain`, ... derive from it.
# The stored axes are the fused codomain/domain ranges; `bispace` dualizes the domain half.
biaxes(m::FusedGradedMatrix) = bispace((m.axis_codomain,), (m.axis_domain,))
Base.axes(m::FusedGradedMatrix) = Tuple(biaxes(m))

Base.size(m::FusedGradedMatrix) = map(length, axes(m))

# ========================  Block indexing (primitive)  ========================

function Base.view(m::FusedGradedMatrix, I::Block{2})
    i, j = Int.(Tuple(I))
    @boundscheck begin
        i in 1:blocklength(m.axis_codomain) && j in 1:blocklength(m.axis_domain) ||
            throw(BoundsError(m, I))
    end
    s_cod = sectors(m.axis_codomain)[i]
    s_dom = sectors(m.axis_domain)[j]
    s_cod == s_dom ||
        error("Off-diagonal access not supported for block-sparse FusedGradedMatrix")
    return FusedSectorMatrix(m.sectordata[s_cod], s_cod)
end

# ========================  eachblockstoredindex  ========================

function eachblockstoredindex(m::FusedGradedMatrix)
    cod, dom = sectordatalengths(m.axis_codomain), sectordatalengths(m.axis_domain)
    return (
        Block(gettoken(cod, c)[2][2], gettoken(dom, c)[2][2]) for
            c in keys(m.sectordata)
    )
end

# ======================== LinearAlgebra ======================

# `adjoint` is the lazy `AdjointFusedGradedArray` wrapper (defined with that type). `transpose` stays
# undefined here since it has requirements on sectors.

# The full-matrix trace is the sum of the per-coupled-sector block traces, each weighted by the
# sector's quantum dimension (the structural factor's trace). Reuse the block-level
# `tr(::FusedSectorMatrix)`, which carries that weighting, so this matches `tr(Array(A))` without
# scalar-indexing the generic `AbstractMatrix` fallback.
function LinearAlgebra.tr(A::FusedGradedMatrix)
    return sum(
        bI -> LinearAlgebra.tr(view(A, bI)), eachblockstoredindex(A);
        init = zero(eltype(A))
    )
end

LinearAlgebra.istriu(A::FusedGradedMatrix) = all(LinearAlgebra.istriu, values(A.sectordata))
LinearAlgebra.istril(A::FusedGradedMatrix) = all(LinearAlgebra.istril, values(A.sectordata))
function LinearAlgebra.isposdef(A::FusedGradedMatrix)
    return all(LinearAlgebra.isposdef, values(A.sectordata))
end
Base.iszero(A::FusedGradedMatrix) = all(iszero, values(A.sectordata))

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    data = similar(m.buffer, T, length(m.buffer))
    return FusedGradedMatrix(data, m.axis_codomain, m.axis_domain)
end
function Base.similar(
        m::FusedGradedMatrix,
        codomain::FusedGradedOneTo{S},
        domain::FusedGradedOneTo{S}
    ) where {S}
    return FusedGradedMatrix{eltype(m)}(undef, codomain, domain)
end
function Base.similar(
        m::FusedGradedMatrix,
        ::Type{T},
        codomain::FusedGradedOneTo{S},
        domain::FusedGradedOneTo{S}
    ) where {T, S}
    if T <: Number
        return FusedGradedMatrix{T}(undef, codomain, domain)
    elseif T <: AbstractMatrix
        return FusedGradedMatrix{eltype(T)}(undef, codomain, domain)
    else
        throw(ArgumentError("invalid type $T"))
    end
end
function Base.similar(
        m::FusedGradedMatrix,
        ::Type{T},
        axis::FusedGradedOneTo{S}
    ) where {T <: AbstractVector, S}
    return FusedGradedVector{eltype(T)}(undef, axis)
end

# ========================  show  ========================

function Base.summary(io::IO, m::FusedGradedMatrix)
    print(
        io, blocklength(m.axis_codomain), "×", blocklength(m.axis_domain), " ",
        summary_typename(typeof(m)),
        " with ", length(m.sectordata), " stored block",
        length(m.sectordata) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(m.sectordata), ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::FusedGradedMatrix)
    summary(io, m)
    println(io, ":")
    for (d, g) in pairs(axes(m))
        print(io, "  Dim $d: ")
        show_axis(io, g)
        println(io)
    end
    isempty(m.sectordata) && return nothing
    Base.print_array(io, m)
    return nothing
end

function Base.show(io::IO, m::FusedGradedMatrix)
    print(
        io, blocklength(m.axis_codomain), "×", blocklength(m.axis_domain), " ",
        summary_typename(typeof(m)),
        " (", length(m.sectordata), " stored)"
    )
    return nothing
end
