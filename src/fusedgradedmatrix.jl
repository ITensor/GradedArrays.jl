# ===========================================================================
#  FusedGradedMatrix — block-diagonal matrix from matricizing a graded array
# ===========================================================================

using MatrixAlgebraKit: MatrixAlgebraKit as MAK

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
    # stored. This is the single place the codomain/domain axes are fused into canonical
    # `FusedGradedOneTo` form; the stored axes are non-dual, with the domain's dual arrow implicit in
    # `axes` (see `biaxes`).
    function FusedGradedMatrix{T, S, V}(
            buffer::V, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
        ) where {T, S <: SectorRange, V <: DenseVector{T}}
        cod = FusedGradedOneTo(codomain)
        dom = FusedGradedOneTo(domain)
        (isdual(cod) || isdual(dom)) && throw(
            ArgumentError(
                "FusedGradedMatrix stores non-dual codomain/domain axes; the domain's dual arrow is implicit in `axes` (see `biaxes`)"
            )
        )
        # Validate the buffer length against the block total (SectorData does the same check on access).
        codl, doml = sectordatalengths(cod), sectordatalengths(dom)
        coupled = intersect(keys(codl), keys(doml))
        total = sum(c -> codl[c] * doml[c], coupled; init = 0)
        length(buffer) == total ||
            throw(
            DimensionMismatch(
                "buffer length $(length(buffer)) does not match block total $total"
            )
        )
        return new{T, S, V}(buffer, cod, dom)
    end
end

"""
    FusedGradedMatrix(buffer, codomain, domain)

Wrap a contiguous `buffer` (shared, not copied), already in TensorKit `.data` layout, as a
`FusedGradedMatrix` with the given codomain and domain axes; the per-coupled-sector blocks are the
lazy `sectordata` view over the buffer. The axes are fused into canonical form. To build from
per-sector block data instead, use [`fusedgradedmatrix`](@ref).
"""
function FusedGradedMatrix(
        buffer::DenseVector, codomain::AbstractGradedOneTo{S}, domain::AbstractGradedOneTo{S}
    ) where {S}
    return FusedGradedMatrix{eltype(buffer), S, typeof(buffer)}(buffer, codomain, domain)
end

# Allocate an uninitialized buffer for the given codomain/domain and wrap it. The block total (sum of
# cod*dom multiplicities over shared sectors) needs the merged per-sector lengths, so fuse the axes to
# size it.
function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    ) where {T}
    cod = FusedGradedOneTo(codomain)
    dom = FusedGradedOneTo(domain)
    codl, doml = sectordatalengths(cod), sectordatalengths(dom)
    coupled = intersect(keys(codl), keys(doml))
    buffer = Vector{T}(undef, sum(c -> codl[c] * doml[c], coupled; init = 0))
    return FusedGradedMatrix(buffer, cod, dom)
end

"""
    fusedgradedmatrix(sectors .=> data, codomain, domain)
    fusedgradedmatrix(sectordata::Dictionary, codomain, domain)

Build a block-diagonal `FusedGradedMatrix` from per-coupled-sector block data (`sector => block`
pairs, any iterator of pairs, or a `Dictionary` keyed by sector) with the given codomain and domain
axes. The codomain and domain sectors need not coincide; the stored blocks are keyed by the sectors
common to both. Bare `TKS.Sector`s are accepted alongside `SectorRange`s; the sectors must be unique.
To wrap an existing contiguous buffer instead, use [`FusedGradedMatrix`](@ref).
"""
function fusedgradedmatrix(
        sectordata, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    )
    ps = collect(sectordata)
    sectors = [SectorRange(first(p)) for p in ps]
    data = [last(p) for p in ps]
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    m = FusedGradedMatrix{eltype(eltype(data))}(undef, codomain, domain)
    codl, doml = sectordatalengths(axis_codomain(m)), sectordatalengths(axis_domain(m))
    blocksectors = intersect(keys(codl), keys(doml))
    issetequal(blocksectors, sectors) || throw(ArgumentError("invalid blocks"))
    # `sectordata` names the argument here; reach the accessor via the module alias.
    dest = GA.sectordata(m)
    for (s, b) in zip(sectors, data)
        size(b) == (codl[s], doml[s]) ||
            throw(DimensionMismatch("invalid block for sector $s"))
        copyto!(dest[s], b)
    end
    return m
end
# A `Dictionary` iterates over its values, not pairs, so route it through `pairs`.
function fusedgradedmatrix(
        sectordata::Dictionary, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    )
    return fusedgradedmatrix(pairs(sectordata), codomain, domain)
end

"""
    fusedgradedmatrix(sectors .=> data)
    fusedgradedmatrix(sectordata::Dictionary)

Build a block-diagonal `FusedGradedMatrix` from per-coupled-sector block data, deriving the codomain
and domain from the blocks' row and column lengths (`codomain[sectors[i]]` is `size(data[i], 1)`,
`domain[sectors[i]]` is `size(data[i], 2)`). Valid only when the codomain, domain, and block sectors
all coincide; pass explicit axes otherwise. Bare `TKS.Sector`s are accepted alongside `SectorRange`s;
the sectors must be sorted and unique.
"""
function fusedgradedmatrix(sectordata)
    ps = collect(sectordata)
    sectors = [SectorRange(first(p)) for p in ps]
    data = [last(p) for p in ps]
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
    codomain = FusedGradedOneTo(sectors, [size(b, 1) for b in data])
    domain = FusedGradedOneTo(sectors, [size(b, 2) for b in data])
    return fusedgradedmatrix(ps, codomain, domain)
end
fusedgradedmatrix(sectordata::Dictionary) = fusedgradedmatrix(pairs(sectordata))

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

# The main diagonal as an owned `FusedGradedVector` whose block at each coupled sector is that block's
# diagonal; the fresh buffer means writing it does not touch `m`. Restricted to equal codomain and
# domain axes (square blocks): only then do the per-block diagonals coincide with the matrix's main
# diagonal. With rectangular blocks the dense diagonal drifts off the blocks into off-diagonal bands,
# so concatenating per-block diagonals is a different operation; iterate blocks explicitly for that. A
# write-through `diagview` of a `FusedGradedMatrix` is not yet supported. Off-diagonals are unsupported.
function LinearAlgebra.diag(m::FusedGradedMatrix)
    checksquare(m)
    return fusedgradedvector(map(MAK.diagview, sectordata(m)))
end
function LinearAlgebra.diag(m::FusedGradedMatrix, k::Integer)
    return error("`diag` on a `FusedGradedMatrix` supports only the main diagonal")
end

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    return FusedGradedMatrix(similar(m.buffer, T), axis_codomain(m), axis_domain(m))
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
