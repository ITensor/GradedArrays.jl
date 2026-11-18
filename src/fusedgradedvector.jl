# ===========================================================================
#  FusedSectorVector and FusedGradedVector
# ===========================================================================

# ---------------------------------------------------------------------------
#  FusedSectorVector
# ---------------------------------------------------------------------------

# One block of a `FusedGradedVector`. Its structural factor is a `SectorOnesVector`, so each
# reduced value is repeated once per state of the irrep.
struct FusedSectorVector{T, S <: Sector, D <: AbstractVector{T}} <:
    AbstractSectorArray{T, S, 1}
    data::D
    sector::S
end

# ---- undef constructors ----

function FusedSectorVector{T, S, D}(
        ::UndefInitializer, sector::S, r::AbstractUnitRange
    ) where {T, S <: Sector, D <: AbstractVector{T}}
    return FusedSectorVector{T, S, D}(similar(D, (r,)), sector)
end

function FusedSectorVector{T}(
        ::UndefInitializer, sector::S, r::AbstractUnitRange
    ) where {T, S <: Sector}
    return FusedSectorVector{T, S, Vector{T}}(undef, sector, r)
end

function FusedSectorVector{T}(::UndefInitializer, sector::Sector, n::Int) where {T}
    return FusedSectorVector{T}(undef, sector, Base.OneTo(n))
end

# ---- accessors ----

structure(sv::FusedSectorVector) = SectorOnesVector{eltype(sv)}(sector(sv))
sector(sv::FusedSectorVector) = sv.sector

datatype(::Type{FusedSectorVector{T, S, D}}) where {T, S, D} = D

Base.copy(sv::FusedSectorVector) = FusedSectorVector(copy(data(sv)), sector(sv))

function Base.similar(sv::FusedSectorVector{<:Any, S, <:Any}, ::Type{T}) where {T, S}
    new_data = similar(data(sv), T)
    D = typeof(new_data)
    return FusedSectorVector{T, S, D}(new_data, sector(sv))
end

Base.conj(a::FusedSectorVector) = throw_flips_first_axis(conj, a)

# ---- display ----

function Base.print_array(io::IO, sv::FusedSectorVector)
    print(io, sector(sv), ": ")
    show(io, data(sv))
    return nothing
end

function Base.show(io::IO, sv::FusedSectorVector)
    print(io, sector(sv), ": ")
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
#  FusedGradedVector
# ---------------------------------------------------------------------------

# The blocks are views into a contiguous `buffer`. `I` is the TensorKitSectors sector type corresponding to
# the `Sector`, used to aid conversion to and from TensorKit. Subject to change.
"""
    FusedGradedVector

A graded vector of per-sector values, as returned by the value-only factorizations `svd_vals`,
`eig_vals` and `eigh_vals`.
"""
struct FusedGradedVector{T, S <: Sector, V <: DenseVector{T}, I <: TKS.Sector} <:
    AbstractFusedGradedVector{T, S}
    buffer::V
    axis::FusedGradedOneTo{S, I}
    datalayout::SectorDataLayout{S, 1}

    # `datalayout` must be the layout of the given axis, which is not checked.
    function FusedGradedVector{T, S, V}(
            buffer::V, axis::FusedGradedOneTo{S, I}, datalayout
        ) where {T, S <: Sector, V <: DenseVector{T}, I <: TKS.Sector}
        isdual(axis) && throw(
            ArgumentError("FusedGradedVector stores a non-dual axis")
        )
        total = bufferlength(datalayout)
        length(buffer) == total ||
            throw(
            DimensionMismatch(
                "buffer length $(length(buffer)) does not match block total $total"
            )
        )
        return new{T, S, V, I}(buffer, axis, datalayout)
    end
end

function FusedGradedVector{T, S, V}(
        buffer::V, axis::AbstractGradedOneTo
    ) where {T, S <: Sector, V <: DenseVector{T}}
    ax = FusedGradedOneTo(axis)
    return FusedGradedVector{T, S, V}(buffer, ax, sectordatalayout(ax))
end

# Wrap a contiguous `buffer` without copying.
function FusedGradedVector(buffer::DenseVector, axis::AbstractGradedOneTo{S}) where {S}
    return FusedGradedVector{eltype(buffer), S, typeof(buffer)}(buffer, axis)
end

function FusedGradedVector(
        buffer::DenseVector, axis::FusedGradedOneTo{S}, datalayout::AbstractDictionary{S}
    ) where {S}
    return FusedGradedVector{eltype(buffer), S, typeof(buffer)}(buffer, axis, datalayout)
end

function FusedGradedVector{T}(::UndefInitializer, axis::AbstractGradedOneTo) where {T}
    return FusedGradedVector(Vector{T}(undef, sum(datalengths(axis); init = 0)), axis)
end

# Build from `sector => data` pairs, or a `Dictionary` keyed by sector. The sectors must be sorted
# and unique.
function fusedgradedvector(sectordata)
    ps = collect(sectordata)
    sectors = [to_sector(first(p)) for p in ps]
    data = [last(p) for p in ps]
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
    axis = FusedGradedOneTo(sectors, [length(b) for b in data])
    v = FusedGradedVector{eltype(eltype(data))}(undef, axis)
    # The argument shadows the `sectordata` function.
    dest = GA.sectordata(v)
    for (s, b) in zip(sectors, data)
        copyto!(dest[s], b)
    end
    return v
end
# A `Dictionary` iterates over its values, not pairs.
fusedgradedvector(sectordata::Dictionary) = fusedgradedvector(pairs(sectordata))

# ========================  Accessors  ========================

# Each block is a `view` into the buffer.
function datatype(::Type{<:FusedGradedVector{T, S, V}}) where {T, S, V}
    return Base.promote_op(view, V, UnitRange{Int})
end

sectordata(v::FusedGradedVector) = SectorData(v, v.datalayout)

axes_codomain(v::FusedGradedVector) = (v.axis,)
axes_domain(v::FusedGradedVector) = ()

Base.dataids(v::FusedGradedVector) = Base.dataids(v.buffer)
Base.parent(v::FusedGradedVector) = v.buffer

# Reduce each block on its own device, then combine the results. `init` is passed only to the
# outer fold, since applying it per block would be wrong for `sum(v; init = 10)`.
function Base.mapreduce(f, op, v::FusedGradedVector; kwargs...)
    return mapfoldl(b -> mapreduce(f, op, b), op, sectordata(v); kwargs...)
end

# The generic `map` would scalar-index into a fresh array.
function Base.map(f, v::FusedGradedVector)
    blockdata = dictionary(s => map(f, b) for (s, b) in pairs(sectordata(v)))
    return fusedgradedvector(blockdata)
end

# ========================  setsectors  ========================

function setsectors(v::FusedGradedVector, ss::AbstractVector)
    ax = setsectors(axis(v), ss)
    ax === axis(v) && return v
    return FusedGradedVector(v.buffer, ax, sectordatalayout(ax))
end

# ========================  Block indexing (primitive)  ========================

function Base.view(v::FusedGradedVector, I::Block{1})
    i = Int(I)
    @boundscheck begin
        i in 1:blocklength(axis(v)) || throw(BoundsError(v, I))
    end
    s = sectors(axis(v))[i]
    return FusedSectorVector(sectordata(v)[s], s)
end

# ========================  eachblockstoredindex  ========================

function eachblockstoredindex(v::FusedGradedVector)
    ax = axis(v)
    return (Block(sectorindex(ax, c)) for c in keys(sectordata(v)))
end

# ========================  similar  ========================

function Base.similar(v::FusedGradedVector, ::Type{T}) where {T}
    return FusedGradedVector(similar(v.buffer, T), axis(v), v.datalayout)
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

# The generic fallback would allocate a `GradedArray` through `similar_map`.
function TensorAlgebra.allocate_output(
        ::typeof(TA.bipermutedimsconj), isconj::Bool, src::FusedGradedVector, perm_codomain,
        perm_domain
    )
    check_input(TA.bipermutedimsconj, isconj, src, perm_codomain, perm_domain)
    return similar(src)
end

# ========================  show  ========================

function Base.summary(io::IO, v::FusedGradedVector)
    sd = sectordata(v)
    print(
        io, blocklength(axis(v)), "-block ", summary_typename(io, typeof(v)),
        " with ", length(sd), " stored block",
        length(sd) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(sd), ", ")
    print(io, "]")
    return nothing
end

# No trailing newline, matching `Base.print_array` for a dense array.
function Base.print_array(io::IO, v::FusedGradedVector)
    for (i, (s, b)) in enumerate(pairs(sectordata(v)))
        i > 1 && println(io)
        print(io, "  ", s, ": ")
        show(io, b)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", v::FusedGradedVector)
    summary(io, v)
    println(io, ":")
    show_biaxes(io, v)
    isempty(sectordata(v)) && return nothing
    println(io)
    Base.print_array(io, v)
    return nothing
end

function Base.show(io::IO, v::FusedGradedVector)
    print(
        io, blocklength(axis(v)), "-block ", summary_typename(io, typeof(v)),
        " (", length(sectordata(v)), " stored)"
    )
    return nothing
end

# ========================  FusedGradedVecOrMat  ========================

const FusedGradedVecOrMat = Union{FusedGradedMatrix, FusedGradedVector}
