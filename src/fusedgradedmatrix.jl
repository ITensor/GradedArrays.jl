# ===========================================================================
#  FusedGradedMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

"""
    FusedGradedMatrix{T,S<:SectorRange,V<:DenseVector{T}}

Block-diagonal matrix produced by matricizing a `GradedArray`. Stores a contiguous `buffer` in
TensorKit `.data` layout plus the fused codomain/domain axes; the per-coupled-sector blocks are the
lazy `sectordata(m)` view carved from the buffer on demand.
"""
struct FusedGradedMatrix{T, S <: SectorRange, V <: DenseVector{T}} <:
    AbstractFusedGradedMatrix{T, S}
    buffer::V
    axis_codomain::FusedGradedOneTo{S}
    axis_domain::FusedGradedOneTo{S}

    # Primitive constructor: wrap a contiguous buffer already in TensorKit `.data` layout (shared, not
    # copied). The blocks are the lazy `sectordata` view over the buffer, so nothing block-shaped is
    # stored. The axes are non-dual `FusedGradedOneTo`s (their constructor enforces sorted, non-dual
    # sectors); the domain's dual arrow is implicit in `axes` (see `biaxes`).
    function FusedGradedMatrix{T, S, V}(
            data::V, codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
        ) where {T, S <: SectorRange, V <: DenseVector{T}}
        (isdual(codomain) || isdual(domain)) && throw(
            ArgumentError(
                "FusedGradedMatrix stores non-dual codomain/domain axes; the domain's dual arrow is implicit in `axes` (see `biaxes`)"
            )
        )
        # Validate the buffer length against the block total (SectorData does the same check on access).
        cod, dom = sectordatalengths(codomain), sectordatalengths(domain)
        coupled = intersect(keys(cod), keys(dom))
        total = sum(c -> cod[c] * dom[c], coupled; init = 0)
        length(data) == total ||
            throw(
            DimensionMismatch(
                "buffer length $(length(data)) does not match block total $total"
            )
        )
        return new{T, S, V}(data, codomain, domain)
    end
end

# Primitive constructor deriving the parameters from the buffer.
function FusedGradedMatrix(
        data::V, codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange, V <: DenseVector{T}}
    return FusedGradedMatrix{T, S, V}(data, codomain, domain)
end

# Data constructor: allocate a fresh buffer and copy each given block into its view. Used by `copy`,
# `adjoint`, the matrix-function loop, and the vector-of-blocks form below, none of which need to
# share the passed blocks.
function FusedGradedMatrix(
        sectordata::Dictionary{S, <:AbstractMatrix},
        codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {S <: SectorRange}
    cod, dom = sectordatalengths(codomain), sectordatalengths(domain)
    blocksectors = intersect(keys(cod), keys(dom))
    issetequal(blocksectors, keys(sectordata)) || throw(ArgumentError("invalid blocks"))
    for (c, b) in pairs(sectordata)
        size(b) == (cod[c], dom[c]) ||
            throw(DimensionMismatch("invalid block for sector $c"))
    end
    T = eltype(eltype(sectordata))
    m = FusedGradedMatrix{T}(undef, codomain, domain)
    # The `sectordata` argument shadows the accessor here, so reach it through the module alias.
    dest = GA.sectordata(m)
    for (c, b) in pairs(sectordata)
        copyto!(dest[c], b)
    end
    return m
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
# `AbstractGradedOneTo` so either a `GradedOneTo` or an already-fused `FusedGradedOneTo` works (the
# `tensor_product`-fused coupled axes arrive as the latter); both normalize to `FusedGradedOneTo`.
function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    ) where {T}
    return FusedGradedMatrix{T}(undef, FusedGradedOneTo(codomain), FusedGradedOneTo(domain))
end
# A single graded range sets the domain equal to the codomain (square blocks), mirroring the
# single-argument pairs form below.
function FusedGradedMatrix{T}(::UndefInitializer, codomain::AbstractGradedOneTo) where {T}
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

# Each block is a reshaped 2-D `view` into the buffer (see `_dataview`); infer that type.
function datatype(::Type{<:FusedGradedMatrix{T, S, V}}) where {T, S, V}
    return Base.promote_op(
        reshape,
        Base.promote_op(view, V, UnitRange{Int}),
        Tuple{Int, Int}
    )
end

function sectordata(m::FusedGradedMatrix)
    return SectorData(
        m, sectordatalengths(axis_codomain(m)), sectordatalengths(axis_domain(m))
    )
end

# `axes_codomain`/`axes_domain` are the core axis accessors (the one place these fields are read
# directly); `biaxes`, `axes`, `axis_codomain`, `view`, the reductions, and the display all derive
# from them and `sectordata` generically. The stored axes are the fused codomain/domain ranges in
# un-dualized form; the derived `biaxes` dualizes the domain half.
axes_codomain(m::FusedGradedMatrix) = (m.axis_codomain,)
axes_domain(m::FusedGradedMatrix) = (m.axis_domain,)

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    data = similar(m.buffer, T)
    return FusedGradedMatrix(data, axis_codomain(m), axis_domain(m))
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
