# ===========================================================================
#  SectorVector and FusedGradedVector
# ===========================================================================

# ---------------------------------------------------------------------------
#  SectorVector — single-sector tagged vector (one block of a FusedGradedVector)
# ---------------------------------------------------------------------------

"""
    SectorVector{T, D<:AbstractVector{T}, S<:SectorRange} <: AbstractSectorArray{T, 1}

A single sector with a data vector. Analogous to [`SectorMatrix`](@ref) but for 1-D data
(eigenvalues, singular values, etc.). Each element is a symmetry scalar — there is no
Wigner-Eckart structural factor; the sector label simply identifies which block the values
belong to.

The stored `SectorRange` is always non-dual (codomain convention).
"""
struct SectorVector{T, D <: AbstractVector{T}, S <: SectorRange} <:
    AbstractSectorArray{T, 1}
    sector::S
    data::D
end

# ---- accessors ----

# Return the SectorRange directly (no structural delta for scalar data).
sector(sv::SectorVector) = sv.sector
dataaxes(sv::SectorVector) = axes(data(sv))
sectoraxes(sv::SectorVector) = (sv.sector,)

sectortype(::Type{<:SectorVector{T, D, S}}) where {T, D, S} = S
datatype(::Type{SectorVector{T, D, S}}) where {T, D, S} = D

Base.axes(sv::SectorVector) = axes(data(sv))

Base.copy(sv::SectorVector) = SectorVector(sv.sector, copy(data(sv)))

function Base.similar(sv::SectorVector{<:Any, <:Any, S}, ::Type{T}) where {T, S}
    new_data = similar(data(sv), T)
    D = typeof(new_data)
    return SectorVector{T, D, S}(sv.sector, new_data)
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
    FusedGradedVector{T,D<:AbstractVector{T},S<:SectorRange}

Block-structured 1-D graded array produced by a sector-preserving operation on
a [`FusedGradedMatrix`](@ref) (e.g. `svd_vals`, `eig_vals`, `eigh_vals`).

Fields:

  - `axis::Dictionary{S,Int}` — axis layout, mapping each sector to its block
    size. Keys are sorted and unique. Stored non-dual (codomain convention).
  - `blocks::Dictionary{S,D}` — stored data blocks, keyed by sector. Keys are
    a subset of `keys(axis)` and `length(blocks[s]) == axis[s]`.
"""
struct FusedGradedVector{T, D <: AbstractVector{T}, S <: SectorRange} <:
    AbstractGradedArray{T, 1}
    axis::Dictionary{S, Int}
    blocks::Dictionary{S, D}

    # Undef constructor
    function FusedGradedVector{T, D, S}(
            ::UndefInitializer, axis::Dictionary{S, Int}
        ) where {T, D <: AbstractVector{T}, S <: SectorRange}
        issorted(keys(axis)) || throw(ArgumentError("axis sectors must be sorted"))

        blocks = dictionary(s => similar(D, (Base.OneTo(axis[s]),)) for s in keys(axis))

        return new{T, D, S}(axis, blocks)
    end

    # Data constructor
    function FusedGradedVector{T, D, S}(
            axis::Dictionary{S, Int}, blocks::Dictionary{S, D}
        ) where {T, D <: AbstractVector{T}, S <: SectorRange}
        issorted(keys(axis)) || throw(ArgumentError("axis sectors must be sorted"))

        issetequal(keys(axis), keys(blocks)) || throw(ArgumentError("invalid blocks"))
        for (s, b) in pairs(blocks)
            length(b) == axis[s] ||
                throw(DimensionMismatch("invalid block for sector $s"))
        end

        return new{T, D, S}(axis, blocks)
    end
end

function FusedGradedVector(
        axis::Dictionary{S, Int}, blocks::Dictionary{S, D}
    ) where {S <: SectorRange, D <: AbstractVector}
    return FusedGradedVector{eltype(D), D, S}(axis, blocks)
end

"""
    FusedGradedVector(sectors::Vector{S}, blocks::Vector{D})

Build a `FusedGradedVector` whose `axis` and `blocks` carry the same sector
list. `axis[sectors[i]]` is `length(blocks[i])`.
"""
function FusedGradedVector(
        sectors::AbstractVector{S},
        blocks::AbstractVector{D}
    ) where {S <: SectorRange, D <: AbstractVector}
    length(sectors) == length(blocks) ||
        throw(ArgumentError("sectors and blocks must have the same length"))
    issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
    allunique(sectors) || throw(ArgumentError("sectors must be unique"))
    ax = Dictionary{S, Int}(sectors, [length(b) for b in blocks])
    blks = Dictionary{S, D}(sectors, collect(blocks))
    return FusedGradedVector(ax, blks)
end

function FusedGradedVector{T}(
        ::UndefInitializer, axis::Dictionary{S, Int}
    ) where {T, S <: SectorRange}
    return FusedGradedVector{T, Vector{T}, S}(undef, axis)
end

# ========================  Accessors  ========================

BlockArrays.blocklength(v::FusedGradedVector) = length(v.axis)

function BlockSparseArrays.blocktype(::Type{<:FusedGradedVector{T, D, S}}) where {T, D, S}
    return SectorVector{T, D, S}
end
BlockSparseArrays.blocktype(v::FusedGradedVector) = BlockSparseArrays.blocktype(typeof(v))
sectortype(::Type{<:FusedGradedVector{T, D, S}}) where {T, D, S} = S
datatype(::Type{<:FusedGradedVector{T, D, S}}) where {T, D, S} = D
datatype(v::FusedGradedVector) = datatype(typeof(v))

function Base.axes(v::FusedGradedVector)
    return (gradedrange([s => l for (s, l) in pairs(v.axis)]),)
end

Base.size(v::FusedGradedVector) = map(length, axes(v))
Base.eltype(::Type{FusedGradedVector{T}}) where {T} = T
Base.eltype(::Type{<:FusedGradedVector{T}}) where {T} = T

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

function BlockSparseArrays.eachblockstoredindex(v::FusedGradedVector)
    return (Block(gettoken(v.axis, c)[2][2]) for c in keys(v.blocks))
end

# ========================  blocks  ========================

function BlockArrays.blocks(v::FusedGradedVector)
    return [view(v, I) for I in eachblockstoredindex(v)]
end

# ========================  fill! / zero!  ========================

function FI.zero!(v::FusedGradedVector)
    for b in values(v.blocks)
        fill!(b, zero(eltype(v)))
    end
    return v
end

function Base.fill!(v::FusedGradedVector, val)
    for b in values(v.blocks)
        fill!(b, val)
    end
    return v
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
        return FusedGradedVector{eltype(T), T, S}(undef, axis)
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
