# ===========================================================================
#  SectorVector and FusedGradedVector
# ===========================================================================

# ---------------------------------------------------------------------------
#  SectorVector — single-sector tagged vector (one block of a FusedGradedVector)
# ---------------------------------------------------------------------------

"""
    SectorVector{T, S<:SectorRange, D<:AbstractVector{T}} <: AbstractSectorArray{T, S, 1}

A single sector with a data vector. Analogous to [`SectorMatrix`](@ref) but for 1-D data
(eigenvalues, singular values, etc.). Each element is a symmetry scalar — there is no
Wigner-Eckart structural factor; the sector label simply identifies which block the values
belong to.

The stored `SectorRange` is always non-dual (codomain convention).
"""
struct SectorVector{T, S <: SectorRange, D <: AbstractVector{T}} <:
    AbstractSectorArray{T, S, 1}
    sector::S
    data::D
end

# ---- accessors ----

# Return the structural delta factor (`SectorOnesVector`, the diagonal of the block's
# `SectorIdentity`), mirroring `sector(::SectorMatrix)`. The stored `SectorRange` is `sv.sector`.
sector(sv::SectorVector) = SectorOnesVector{eltype(sv)}(sv.sector)
dataaxes(sv::SectorVector) = axes(data(sv))
sectoraxes(sv::SectorVector) = (sv.sector,)

datatype(::Type{SectorVector{T, S, D}}) where {T, S, D} = D

Base.axes(sv::SectorVector) = axes(data(sv))

Base.copy(sv::SectorVector) = SectorVector(sv.sector, copy(data(sv)))

function Base.similar(sv::SectorVector{<:Any, S, <:Any}, ::Type{T}) where {T, S}
    new_data = similar(data(sv), T)
    D = typeof(new_data)
    return SectorVector{T, S, D}(sv.sector, new_data)
end

function Base.fill!(sv::SectorVector, v)
    fill!(data(sv), v)
    return sv
end

# ---- display ----

function Base.print_array(io::IO, sv::SectorVector)
    print(io, sv.sector, ": ")
    show(io, data(sv))
    return nothing
end

function Base.show(io::IO, sv::SectorVector)
    print(io, sv.sector, ": ")
    show(io, data(sv))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sv::SectorVector)
    summary(io, sv)
    println(io, ":")
    Base.print_array(io, sv)
    return nothing
end

# ---------------------------------------------------------------------------
#  FusedGradedVector — block-structured 1-D graded array for per-sector scalars
# ---------------------------------------------------------------------------

"""
    FusedGradedVector{T,S<:SectorRange,D<:AbstractVector{T}}

Block-structured 1-D graded array produced by a sector-preserving operation on
a [`FusedGradedMatrix`](@ref) (e.g. `svd_vals`, `eig_vals`, `eigh_vals`).

Fields:

  - `axis::Dictionary{S,Int}` — axis layout, mapping each sector to its block
    size. Keys are sorted and unique. Stored non-dual (codomain convention).
  - `blocks::Dictionary{S,D}` — stored data blocks, keyed by sector. Keys match
    `keys(axis)` exactly and `length(blocks[s]) == axis[s]`.
"""
struct FusedGradedVector{T, S <: SectorRange, D <: AbstractVector{T}} <:
    AbstractGradedArray{T, S, 1}
    axis::Dictionary{S, Int}
    blocks::Dictionary{S, D}

    # Undef constructor
    function FusedGradedVector{T, S, D}(
            ::UndefInitializer, axis::Dictionary{S, Int}
        ) where {T, S <: SectorRange, D <: AbstractVector{T}}
        issorted(keys(axis)) || throw(ArgumentError("axis sectors must be sorted"))

        blocks = dictionary(s => similar(D, (Base.OneTo(axis[s]),)) for s in keys(axis))

        return new{T, S, D}(axis, blocks)
    end

    # Data constructor
    function FusedGradedVector{T, S, D}(
            axis::Dictionary{S, Int}, blocks::Dictionary{S, D}
        ) where {T, S <: SectorRange, D <: AbstractVector{T}}
        issorted(keys(axis)) || throw(ArgumentError("axis sectors must be sorted"))

        issetequal(keys(axis), keys(blocks)) || throw(ArgumentError("invalid blocks"))
        for (s, b) in pairs(blocks)
            length(b) == axis[s] ||
                throw(DimensionMismatch("invalid block for sector $s"))
        end

        return new{T, S, D}(axis, blocks)
    end
end

function FusedGradedVector(
        axis::Dictionary{S, Int}, blocks::Dictionary{S, D}
    ) where {S <: SectorRange, D <: AbstractVector}
    return FusedGradedVector{eltype(D), S, D}(axis, blocks)
end

"""
    FusedGradedVector(sectors::Vector{S}, blocks::Vector{D})

Build a `FusedGradedVector` whose `axis` and `blocks` carry the same sector
list. `axis[sectors[i]]` is `length(blocks[i])`.
"""
function FusedGradedVector(
        sectors::AbstractVector,
        blocks::AbstractVector{D}
    ) where {D <: AbstractVector}
    length(sectors) == length(blocks) ||
        throw(ArgumentError("sectors and blocks must have the same length"))
    # Accept bare `TKS.Sector`s (e.g. `FermionNumber(1)`) alongside `SectorRange`s, as
    # `gradedrange` does; `SectorRange` wraps the former and is the identity on the latter.
    rs = map(SectorRange, sectors)
    issorted(rs) || throw(ArgumentError("sectors must be sorted"))
    allunique(rs) || throw(ArgumentError("sectors must be unique"))
    S = eltype(rs)
    ax = Dictionary{S, Int}(rs, [length(b) for b in blocks])
    blks = Dictionary{S, D}(rs, collect(blocks))
    return FusedGradedVector(ax, blks)
end

function FusedGradedVector{T}(
        ::UndefInitializer, axis::Dictionary{S, Int}
    ) where {T, S <: SectorRange}
    return FusedGradedVector{T, S, Vector{T}}(undef, axis)
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
    return FusedGradedVector{T}(
        undef,
        Dictionary{eltype(rs), Int}(rs, collect(Int, datalengths))
    )
end
function FusedGradedVector{T}(::UndefInitializer, blocks::AbstractVector{<:Pair}) where {T}
    return FusedGradedVector{T}(undef, first.(blocks), last.(blocks))
end

# ========================  Accessors  ========================

# Blockwise copyto!: the generic `AbstractArray` fallback copies elementwise, which
# scalar-indexes (disallowed for graded arrays). Also the write path for mutating a
# `MAK.diagview(::FusedGradedMatrix)` (whose blocks alias the matrix diagonals).
function Base.copyto!(dest::FusedGradedVector, src::FusedGradedVector)
    keys(dest.blocks) == keys(src.blocks) ||
        throw(ArgumentError("`copyto!` requires matching sectors"))
    for s in keys(src.blocks)
        copyto!(dest.blocks[s], src.blocks[s])
    end
    return dest
end

BlockArrays.blocklength(v::FusedGradedVector) = length(v.axis)

function blocktype(::Type{<:FusedGradedVector{T, S, D}}) where {T, S, D}
    return SectorVector{T, S, D}
end
blocktype(v::FusedGradedVector) = blocktype(typeof(v))

function Base.axes(v::FusedGradedVector)
    return (gradedrange([s => l for (s, l) in pairs(v.axis)]),)
end

Base.size(v::FusedGradedVector) = map(length, axes(v))
Base.eltype(::Type{FusedGradedVector{T}}) where {T} = T
Base.eltype(::Type{<:FusedGradedVector{T}}) where {T} = T

# Block-wise `mapreduce`: reduce each block locally (so GPU blocks stay on the device for
# their reduction kernel) and combine per-block scalars on the CPU. Routes
# `maximum(abs, v; init=…)`, `sum`, `LinearAlgebra.norm`, etc. without ever falling
# through to `getindex(v, ::Int)`. `init` (and any other kwargs) flow only to the outer
# cross-block fold; double-applying `init` per block would be wrong for non-idempotent
# reductions (e.g. `sum(v; init=10)`).
function Base.mapreduce(f, op, v::FusedGradedVector; kwargs...)
    return mapfoldl(b -> mapreduce(f, op, b), op, values(v.blocks); kwargs...)
end

# Block-wise `map`: returns a `FusedGradedVector` with the same axis and `f` applied to
# each stored block, instead of falling through to `collect_similar` which would
# allocate an `AbelianGradedVector` and scalar-setindex! into it. Each per-block `map`
# dispatches to the storage backend's `map` (e.g. GPU kernel for `CuVector` blocks).
function Base.map(f, v::FusedGradedVector)
    blocks = dictionary(s => map(f, b) for (s, b) in pairs(v.blocks))
    return FusedGradedVector(v.axis, blocks)
end

# ========================  Block indexing (primitive)  ========================

function Base.view(v::FusedGradedVector, I::Block{1})
    i = Int(I)
    @boundscheck begin
        i in 1:length(v.axis) || throw(BoundsError(v, I))
    end
    s = gettokenvalue(keys(v.axis), i)
    return SectorVector(s, v.blocks[s])
end

# ========================  eachblockstoredindex  ========================

function eachblockstoredindex(v::FusedGradedVector)
    return (Block(gettoken(v.axis, c)[2][2]) for c in keys(v.blocks))
end

# ========================  blocks  ========================

function BlockArrays.blocks(v::FusedGradedVector)
    return [view(v, I) for I in eachblockstoredindex(v)]
end

# ========================  similar  ========================

function Base.similar(v::FusedGradedVector, ::Type{T}) where {T}
    new_blocks = map(b -> similar(b, T), v.blocks)
    return FusedGradedVector(v.axis, new_blocks)
end
function Base.similar(v::FusedGradedVector, axis::Dictionary{S, Int}) where {S}
    return typeof(v)(undef, axis)
end
function Base.similar(
        v::FusedGradedVector,
        ::Type{T},
        axis::Dictionary{S, Int}
    ) where {T, S}
    if T <: Number
        return FusedGradedVector{T}(undef, axis)
    elseif T <: AbstractVector
        return FusedGradedVector{eltype(T), S, T}(undef, axis)
    else
        throw(ArgumentError("invalid type $T"))
    end
end

# ========================  show  ========================

function Base.summary(io::IO, v::FusedGradedVector)
    print(
        io, length(v.axis), "-block ", typeof(v),
        " with ", length(v.blocks), " stored block",
        length(v.blocks) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(v.blocks), ", ")
    print(io, "]")
    return nothing
end

function Base.print_array(io::IO, v::FusedGradedVector)
    for (s, b) in pairs(v.blocks)
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
    show(io, axes(v, 1))
    println(io)
    isempty(v.blocks) && return nothing
    Base.print_array(io, v)
    return nothing
end

function Base.show(io::IO, v::FusedGradedVector)
    print(
        io, length(v.axis), "-block ", typeof(v),
        " (", length(v.blocks), " stored)"
    )
    return nothing
end

# ========================  copy  ========================

function Base.copy(v::FusedGradedVector)
    return FusedGradedVector(copy(v.axis), map(copy, v.blocks))
end

# Materialize into a dense `Array` (the generic fallback copies elementwise, which
# scalar-indexes). `_to_blockarray` reintroduces each block's structural factor
# (`SectorOnesVector`, i.e. `ones ⊗ reduced`), which is the identity for abelian sectors but
# repeats each reduced value over the irrep's quantum dimension for non-abelian ones.
Base.Array(v::FusedGradedVector) = Array(_to_blockarray(v))

# ======================== LinearAlgebra ======================

function LinearAlgebra.norm(A::FusedGradedVector, p::Real = 2)
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

# ========================  FusedGradedVecOrMat  ========================

# Union of the two fused block-structured graded array types, following the
# `Base.AbstractVecOrMat` naming convention.
const FusedGradedVecOrMat = Union{FusedGradedMatrix, FusedGradedVector}

# Rebuild a fused sector→length axis dictionary from a graded range, inverting the `gradedrange`
# that `axes` builds. `eachsectoraxis` keeps each sector's `isdual` (unlike `sectors`), so a
# dualized axis (from a `conj` broadcast) round-trips to dualized keys. The codomain stores the
# axis sectors directly; the domain stores their duals, inverting the `dual` in `axes(m, 2)`.
# Used by the `_similar_fused` broadcast allocators.
function _fusedcodomain(g::GradedOneTo)
    return Dictionary(collect(eachsectoraxis(g)), collect(Int, blocklengths(g)))
end
function _fuseddomain(g::GradedOneTo)
    return Dictionary(map(dual, eachsectoraxis(g)), collect(Int, blocklengths(g)))
end

# Block-aware random fills, mirroring `AbelianGradedArray`: fill each stored block, bypassing the
# scalar-indexing `AbstractArray` fallback (disallowed for graded arrays).
function Random.rand!(rng::AbstractRNG, a::FusedGradedVecOrMat, sp::Random.Sampler)
    for b in values(a.blocks)
        Random.rand!(rng, b, sp)
    end
    return a
end
function Random.randn!(rng::AbstractRNG, a::FusedGradedVecOrMat)
    for b in values(a.blocks)
        Random.randn!(rng, b)
    end
    return a
end
