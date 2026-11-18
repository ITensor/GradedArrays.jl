# ===========================================================================
#  FusedGradedMatrix
# ===========================================================================

using MatrixAlgebraKit: MatrixAlgebraKit as MAK

# The blocks are views into a contiguous `buffer` in TensorKit's `.data` layout. `I` is the
# TensorKitSectors sector type corresponding to
# the `Sector`, used to aid conversion to and from TensorKit. Subject to change.
"""
    FusedGradedMatrix

The block-diagonal matrix a graded array matricizes to, as returned by `matricize`. Its two axes
are the fused codomain and domain.
"""
struct FusedGradedMatrix{T, S <: Sector, V <: DenseVector{T}, I <: TKS.Sector} <:
    AbstractFusedGradedMatrix{T, S}
    buffer::V
    axis_codomain::FusedGradedOneTo{S, I}
    axis_domain::FusedGradedOneTo{S, I}
    datalayout::SectorDataLayout{S, 2}

    # `datalayout` must be the layout of the given axes, which is not checked.
    function FusedGradedMatrix{T, S, V}(
            buffer::V, codomain::FusedGradedOneTo{S, I}, domain::FusedGradedOneTo{S, I},
            datalayout
        ) where {T, S <: Sector, V <: DenseVector{T}, I <: TKS.Sector}
        (isdual(codomain) || isdual(domain)) && throw(
            ArgumentError(
                "FusedGradedMatrix stores non-dual codomain/domain axes; the domain's dual arrow is implicit in `axes` (see `biaxes`)"
            )
        )
        total = bufferlength(datalayout)
        length(buffer) == total ||
            throw(
            DimensionMismatch(
                "buffer length $(length(buffer)) does not match block total $total"
            )
        )
        return new{T, S, V, I}(buffer, codomain, domain, datalayout)
    end
end

function FusedGradedMatrix{T, S, V}(
        buffer::V, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    ) where {T, S <: Sector, V <: DenseVector{T}}
    cod = FusedGradedOneTo(codomain)
    dom = FusedGradedOneTo(domain)
    return FusedGradedMatrix{T, S, V}(buffer, cod, dom, sectordatalayout(cod, dom))
end

# Wrap a contiguous `buffer` in TensorKit's `.data` layout without copying.
function FusedGradedMatrix(
        buffer::DenseVector, codomain::AbstractGradedOneTo{S}, domain::AbstractGradedOneTo{S}
    ) where {S}
    return FusedGradedMatrix{eltype(buffer), S, typeof(buffer)}(buffer, codomain, domain)
end

function FusedGradedMatrix(
        buffer::DenseVector, codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S},
        datalayout::AbstractDictionary{S}
    ) where {S}
    return FusedGradedMatrix{eltype(buffer), S, typeof(buffer)}(
        buffer, codomain, domain, datalayout
    )
end

function FusedGradedMatrix{T}(
        ::UndefInitializer, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    ) where {T}
    cod = FusedGradedOneTo(codomain)
    dom = FusedGradedOneTo(domain)
    datalayout = sectordatalayout(cod, dom)
    buffer = Vector{T}(undef, bufferlength(datalayout))
    return FusedGradedMatrix(buffer, cod, dom, datalayout)
end

# A `FusedGradedMatrix` over the given axes, with its buffer from `allocator` and of the type of
# the buffer of `prototype`.
function similar_fusedgradedmatrix(
        allocator, prototype, ::Type{T}, codomain::AbstractGradedOneTo,
        domain::AbstractGradedOneTo
    ) where {T}
    cod = FusedGradedOneTo(codomain)
    dom = FusedGradedOneTo(domain)
    datalayout = sectordatalayout(cod, dom)
    buffer = TA.similar_map(
        allocator, storage_prototype(prototype), T,
        (Base.OneTo(bufferlength(datalayout)),), ()
    )
    return FusedGradedMatrix(buffer, cod, dom, datalayout)
end
# The dense vector holding the data of a graded array.
storage_prototype(buffer::DenseVector) = buffer
storage_prototype(a::AbstractFusedGradedArray) = storage_prototype(parent(a))

# Lets a caller forward its own buffer type, for example a GPU array.
function FusedGradedMatrix{T, S, V}(
        ::UndefInitializer, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    ) where {T, S, V}
    cod = FusedGradedOneTo(codomain)
    dom = FusedGradedOneTo(domain)
    datalayout = sectordatalayout(cod, dom)
    buffer = V(undef, bufferlength(datalayout))
    return FusedGradedMatrix{T, S, V}(buffer, cod, dom, datalayout)
end

# Build from `sector => block` pairs, or a `Dictionary` keyed by sector, with one block for each
# sector the codomain and domain share.
function fusedgradedmatrix(
        sectordata, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    )
    ps = collect(sectordata)
    sectors = [to_sector(first(p)) for p in ps]
    data = [last(p) for p in ps]
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    m = FusedGradedMatrix{eltype(eltype(data))}(undef, codomain, domain)
    # The argument shadows the `sectordata` function.
    dest = GA.sectordata(m)
    issetequal(keys(dest), sectors) || throw(ArgumentError("invalid blocks"))
    for (s, b) in zip(sectors, data)
        size(b) == size(dest[s]) ||
            throw(DimensionMismatch("invalid block for sector $s"))
        copyto!(dest[s], b)
    end
    return m
end
# A `Dictionary` iterates over its values, not pairs.
function fusedgradedmatrix(
        sectordata::Dictionary, codomain::AbstractGradedOneTo, domain::AbstractGradedOneTo
    )
    return fusedgradedmatrix(pairs(sectordata), codomain, domain)
end

# Derives the codomain and domain from the block sizes, so the codomain, domain and block sectors
# must all coincide. The sectors must be sorted and unique.
function fusedgradedmatrix(sectordata)
    ps = collect(sectordata)
    sectors = [to_sector(first(p)) for p in ps]
    data = [last(p) for p in ps]
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
    codomain = FusedGradedOneTo(sectors, [size(b, 1) for b in data])
    domain = FusedGradedOneTo(sectors, [size(b, 2) for b in data])
    return fusedgradedmatrix(ps, codomain, domain)
end
fusedgradedmatrix(sectordata::Dictionary) = fusedgradedmatrix(pairs(sectordata))

# ========================  Accessors  ========================

# Each block is a reshaped `view` into the buffer.
function datatype(::Type{<:FusedGradedMatrix{T, S, V}}) where {T, S, V}
    return Base.promote_op(
        reshape,
        Base.promote_op(view, V, UnitRange{Int}),
        Tuple{Int, Int}
    )
end

sectordata(m::FusedGradedMatrix) = SectorData(m, m.datalayout)

axes_codomain(m::FusedGradedMatrix) = (m.axis_codomain,)
axes_domain(m::FusedGradedMatrix) = (m.axis_domain,)

# Lets `Base.mightalias` detect sharing with the buffer and its wrappers.
Base.dataids(m::FusedGradedMatrix) = Base.dataids(m.buffer)
Base.parent(m::FusedGradedMatrix) = m.buffer

# ========================  setsectors  ========================

function setsectors(m::FusedGradedMatrix, ss::AbstractVector)
    cod = setsectors(axis_codomain(m), ss)
    dom = setsectors(axis_domain(m), ss)
    (cod === axis_codomain(m) && dom === axis_domain(m)) && return m
    return FusedGradedMatrix(m.buffer, cod, dom, sectordatalayout(cod, dom))
end

# ========================  buffer fast paths  ========================
# The blocks tile the buffer exactly and equal axes give the same layout, so these act on the
# buffer directly.

function TensorAlgebra.zero!(m::FusedGradedMatrix)
    fill!(m.buffer, zero(eltype(m)))
    return m
end

function Base.copyto!(dest::FusedGradedMatrix, src::FusedGradedMatrix)
    axes(dest) == axes(src) || throw(DimensionMismatch("`copyto!` requires matching axes"))
    copyto!(dest.buffer, src.buffer)
    return dest
end

# Only for square blocks: with rectangular blocks the main diagonal leaves the blocks, so it is not
# the concatenation of the block diagonals.
function LinearAlgebra.diag(m::FusedGradedMatrix)
    checksquare(m)
    return fusedgradedvector(map(MAK.diagview, sectordata(m)))
end
function LinearAlgebra.diag(m::FusedGradedMatrix, k::Integer)
    return error("`diag` on a `FusedGradedMatrix` supports only the main diagonal")
end

# ========================  similar  ========================

function Base.similar(m::FusedGradedMatrix, ::Type{T}) where {T}
    return FusedGradedMatrix(
        similar(m.buffer, T), axis_codomain(m), axis_domain(m), m.datalayout
    )
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
