# ===========================================================================
#  FusedSectorVector and FusedGradedVector
# ===========================================================================

# ---------------------------------------------------------------------------
#  FusedSectorVector — single-sector tagged vector (one block of a FusedGradedVector)
# ---------------------------------------------------------------------------

"""
    FusedSectorVector{T, S<:SectorRange, D<:AbstractVector{T}} <: AbstractSectorArray{T, S, 1}

A single sector with a data vector. Analogous to [`FusedSectorMatrix`](@ref) but for 1-D data
(eigenvalues, singular values, etc.). Each element is a symmetry scalar — there is no
Wigner-Eckart structural factor; the sector label simply identifies which block the values
belong to.

The stored `SectorRange` is always non-dual (codomain convention).
"""
struct FusedSectorVector{T, S <: SectorRange, D <: AbstractVector{T}} <:
    AbstractSectorArray{T, S, 1}
    data::D
    sector::S
    function FusedSectorVector{T, S, D}(
            data::D, sector::S
        ) where {T, S <: SectorRange, D <: AbstractVector{T}}
        !isdual(sector) ||
            throw(
            ArgumentError(
                "`FusedSectorVector` requires a non-dual sector, got `$sector`"
            )
        )
        return new{T, S, D}(data, sector)
    end
end

# Default the parameters from the data and sector types.
function FusedSectorVector(data::D, sector::S) where {S <: SectorRange, D <: AbstractVector}
    return FusedSectorVector{eltype(D), S, D}(data, sector)
end

# ---- undef constructors ----

# Innermost: fully parameterized, takes an AbstractUnitRange data axis.
function FusedSectorVector{T, S, D}(
        ::UndefInitializer, sector::S, r::AbstractUnitRange
    ) where {T, S <: SectorRange, D <: AbstractVector{T}}
    return FusedSectorVector{T, S, D}(similar(D, (r,)), sector)
end

# Convenience: default D = Vector{T}.
function FusedSectorVector{T}(
        ::UndefInitializer, sector::S, r::AbstractUnitRange
    ) where {T, S <: SectorRange}
    return FusedSectorVector{T, S, Vector{T}}(undef, sector, r)
end

# Int convenience: maps to Base.OneTo.
function FusedSectorVector{T}(::UndefInitializer, sector::SectorRange, n::Int) where {T}
    return FusedSectorVector{T}(undef, sector, Base.OneTo(n))
end

# ---- accessors ----

# Return the structural delta factor (`SectorOnesVector`, the diagonal of the block's
# `SectorIdentity`), mirroring `sector(::FusedSectorMatrix)`. The stored `SectorRange` is `sv.sector`.
# sectoraxes, dataaxes, and axes are derived generically on AbstractSectorArray from sector and data;
# a `FusedSectorVector`'s single axis is thus a `SectorOneTo` carrying the sector (its `size` is the
# block's full graded length, not the reduced data length), matching the matrix blocks.
sector(sv::FusedSectorVector) = SectorOnesVector{eltype(sv)}(sv.sector)

datatype(::Type{FusedSectorVector{T, S, D}}) where {T, S, D} = D

Base.copy(sv::FusedSectorVector) = FusedSectorVector(copy(data(sv)), sv.sector)

function Base.similar(sv::FusedSectorVector{<:Any, S, <:Any}, ::Type{T}) where {T, S}
    new_data = similar(data(sv), T)
    D = typeof(new_data)
    return FusedSectorVector{T, S, D}(new_data, sv.sector)
end

Base.conj(a::FusedSectorVector) = throw_flips_first_axis(conj, a)

# ---- display ----

function Base.print_array(io::IO, sv::FusedSectorVector)
    print(io, sv.sector, ": ")
    show(io, data(sv))
    return nothing
end

function Base.show(io::IO, sv::FusedSectorVector)
    print(io, sv.sector, ": ")
    show(io, data(sv))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sv::FusedSectorVector)
    summary(io, sv)
    println(io, ":")
    Base.print_array(io, sv)
    return nothing
end

# ---------------------------------------------------------------------------
#  FusedGradedVector — block-structured 1-D graded array for per-sector scalars
# ---------------------------------------------------------------------------

"""
    FusedGradedVector{T,S<:SectorRange,V<:DenseVector{T}}

Block-structured 1-D graded array produced by a sector-preserving operation on
a [`FusedGradedMatrix`](@ref) (e.g. `svd_vals`, `eig_vals`, `eigh_vals`). Stores a contiguous
`buffer` plus the fused axis; the per-sector blocks are the lazy `sectordata(v)` view carved from the
buffer on demand.
"""
struct FusedGradedVector{T, S <: SectorRange, V <: DenseVector{T}} <:
    AbstractFusedGradedVector{T, S}
    buffer::V
    axis::FusedGradedOneTo{S}

    # Primitive constructor: wrap a contiguous buffer (shared, not copied); the per-sector blocks are
    # the lazy `sectordata` view over it. The axis is a non-dual `FusedGradedOneTo` (its constructor
    # enforces sorted, non-dual sectors).
    function FusedGradedVector{T, S, V}(
            data::V, axis::FusedGradedOneTo{S}
        ) where {T, S <: SectorRange, V <: DenseVector{T}}
        isdual(axis) && throw(
            ArgumentError("FusedGradedVector stores a non-dual axis")
        )
        # Validate the buffer length against the block total (SectorData does the same check on access).
        total = sum(values(sectordatalengths(axis)); init = 0)
        length(data) == total ||
            throw(
            DimensionMismatch(
                "buffer length $(length(data)) does not match block total $total"
            )
        )
        return new{T, S, V}(data, axis)
    end
end

# Primitive constructor deriving the parameters from the buffer.
function FusedGradedVector(
        data::V, axis::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange, V <: DenseVector{T}}
    return FusedGradedVector{T, S, V}(data, axis)
end

# Data constructor: allocate a fresh buffer and copy each given block into its view.
function FusedGradedVector(
        sectordata::Dictionary{S, <:AbstractVector}, axis::FusedGradedOneTo{S}
    ) where {S <: SectorRange}
    ax = sectordatalengths(axis)
    issetequal(keys(ax), keys(sectordata)) || throw(ArgumentError("invalid blocks"))
    for (s, b) in pairs(sectordata)
        length(b) == ax[s] ||
            throw(DimensionMismatch("invalid block for sector $s"))
    end
    T = eltype(eltype(sectordata))
    v = FusedGradedVector{T}(undef, axis)
    # The `sectordata` argument shadows the accessor here, so reach it through the module alias.
    dest = GA.sectordata(v)
    for (s, b) in pairs(sectordata)
        copyto!(dest[s], b)
    end
    return v
end

"""
    FusedGradedVector(blocks::Vector{D}, sectors::Vector{S})

Build a `FusedGradedVector` whose `axis` and `blocks` carry the same sector
list. `axis[sectors[i]]` is `length(blocks[i])`.
"""
function FusedGradedVector(
        blocks::AbstractVector{D},
        sectors::AbstractVector
    ) where {D <: AbstractVector}
    length(sectors) == length(blocks) ||
        throw(ArgumentError("sectors and blocks must have the same length"))
    # Accept bare `TKS.Sector`s (e.g. `FermionNumber(1)`) alongside `SectorRange`s, as
    # `gradedrange` does; `SectorRange` wraps the former and is the identity on the latter.
    rs = map(SectorRange, sectors)
    issorted(rs) || throw(ArgumentError("sectors must be sorted"))
    allunique(rs) || throw(ArgumentError("sectors must be unique"))
    S = eltype(rs)
    ax = FusedGradedOneTo(rs, [length(b) for b in blocks])
    blks = Dictionary{S, D}(rs, collect(blocks))
    return FusedGradedVector(blks, ax)
end

function FusedGradedVector{T}(
        ::UndefInitializer, axis::FusedGradedOneTo{S}
    ) where {T, S <: SectorRange}
    total = sum(values(sectordatalengths(axis)); init = 0)
    return FusedGradedVector(Vector{T}(undef, total), axis)
end

# Build from the axis graded range, the inverse of `axes(v)`. Used to allocate a broadcast result.
function FusedGradedVector{T}(::UndefInitializer, g::GradedOneTo) where {T}
    return FusedGradedVector{T}(undef, FusedGradedOneTo(g))
end
function FusedGradedVector{T}(
        ::UndefInitializer,
        axs::Tuple{<:AbstractGradedOneTo}
    ) where {T}
    return FusedGradedVector{T}(undef, FusedGradedOneTo(only(axs)))
end

"""
    FusedGradedVector{T}(undef, sectors, datalengths)
    FusedGradedVector{T}(undef, sectors .=> datalengths)

Allocate a `FusedGradedVector` with uninitialized blocks, `datalengths[i]` the reduced length of the
block at `sectors[i]`. The two forms mirror the `Dictionary(keys, values)` and `dictionary(pairs)`
constructors from `Dictionaries`. Bare `TKS.Sector`s are accepted alongside `SectorRange`s. Pair with
`randn!`/`rand!` to fill.
"""
function FusedGradedVector{T}(
        ::UndefInitializer, sectors::AbstractVector, datalengths::AbstractVector
    ) where {T}
    rs = map(SectorRange, sectors)
    return FusedGradedVector{T}(undef, FusedGradedOneTo(rs, collect(Int, datalengths)))
end
function FusedGradedVector{T}(::UndefInitializer, blocks::AbstractVector{<:Pair}) where {T}
    return FusedGradedVector{T}(undef, first.(blocks), last.(blocks))
end

# ========================  Accessors  ========================

# Each block is a 1-D `view` into the buffer (see `_dataview`); infer that type.
function datatype(::Type{<:FusedGradedVector{T, S, V}}) where {T, S, V}
    return Base.promote_op(view, V, UnitRange{Int})
end

sectordata(v::FusedGradedVector) = SectorData(v, sectordatalengths(v.axis))

# The stored axis is the fused codomain range; a vector has an empty domain. `axes_codomain`/
# `axes_domain` are the core axis accessors (the one place `v.axis` is read directly); `biaxes`,
# `axes`, and `size` derive from them generically.
axes_codomain(v::FusedGradedVector) = (v.axis,)
axes_domain(v::FusedGradedVector) = ()

# Block-wise `mapreduce`: reduce each block locally (so GPU blocks stay on the device for
# their reduction kernel) and combine per-block scalars on the CPU. Routes
# `maximum(abs, v; init=…)`, `sum`, `LinearAlgebra.norm`, etc. without ever falling
# through to `getindex(v, ::Int)`. `init` (and any other kwargs) flow only to the outer
# cross-block fold; double-applying `init` per block would be wrong for non-idempotent
# reductions (e.g. `sum(v; init=10)`).
function Base.mapreduce(f, op, v::FusedGradedVector; kwargs...)
    return mapfoldl(b -> mapreduce(f, op, b), op, sectordata(v); kwargs...)
end

# Block-wise `map`: returns a `FusedGradedVector` with the same axis and `f` applied to
# each stored block, instead of falling through to `collect_similar` which would
# scalar-setindex! into a fresh array. Each per-block `map`
# dispatches to the storage backend's `map` (e.g. GPU kernel for `CuVector` blocks).
function Base.map(f, v::FusedGradedVector)
    blockdata = dictionary(s => map(f, b) for (s, b) in pairs(sectordata(v)))
    return FusedGradedVector(blockdata, v.axis)
end

# ========================  Block indexing (primitive)  ========================

function Base.view(v::FusedGradedVector, I::Block{1})
    i = Int(I)
    @boundscheck begin
        i in 1:blocklength(v.axis) || throw(BoundsError(v, I))
    end
    s = sectors(v.axis)[i]
    return FusedSectorVector(sectordata(v)[s], s)
end

# ========================  eachblockstoredindex  ========================

function eachblockstoredindex(v::FusedGradedVector)
    ax = sectordatalengths(v.axis)
    return (Block(gettoken(ax, c)[2][2]) for c in keys(sectordata(v)))
end

# ========================  similar  ========================

function Base.similar(v::FusedGradedVector, ::Type{T}) where {T}
    data = similar(v.buffer, T)
    return FusedGradedVector(data, v.axis)
end
function Base.similar(v::FusedGradedVector, axis::FusedGradedOneTo{S}) where {S}
    return FusedGradedVector{eltype(v)}(undef, axis)
end
function Base.similar(
        v::FusedGradedVector,
        ::Type{T},
        axis::FusedGradedOneTo{S}
    ) where {T, S}
    if T <: Number
        return FusedGradedVector{T}(undef, axis)
    elseif T <: AbstractVector
        return FusedGradedVector{eltype(T)}(undef, axis)
    else
        throw(ArgumentError("invalid type $T"))
    end
end

# ========================  show  ========================

function Base.summary(io::IO, v::FusedGradedVector)
    sd = sectordata(v)
    print(
        io, blocklength(v.axis), "-block ", summary_typename(typeof(v)),
        " with ", length(sd), " stored block",
        length(sd) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(sd), ", ")
    print(io, "]")
    return nothing
end

function Base.print_array(io::IO, v::FusedGradedVector)
    for (s, b) in pairs(sectordata(v))
        print(io, "  ", s, ": ")
        show(io, b)
        println(io)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", v::FusedGradedVector)
    summary(io, v)
    println(io, ":")
    print(io, "  Dim 1: ")
    show_axis(io, axes(v, 1))
    println(io)
    isempty(sectordata(v)) && return nothing
    Base.print_array(io, v)
    return nothing
end

function Base.show(io::IO, v::FusedGradedVector)
    print(
        io, blocklength(v.axis), "-block ", summary_typename(typeof(v)),
        " (", length(sectordata(v)), " stored)"
    )
    return nothing
end

# ========================  FusedGradedVecOrMat  ========================

# Union of the two fused block-structured graded array types, following the
# `Base.AbstractVecOrMat` naming convention.
const FusedGradedVecOrMat = Union{FusedGradedMatrix, FusedGradedVector}
