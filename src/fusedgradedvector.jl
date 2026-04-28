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

Block-structured 1-D graded array produced by a sector-preserving operation on a
[`FusedGradedMatrix`](@ref) (e.g. `svd_vals`, `eig_vals`, `eigh_vals`).
Each block holds the scalar values (singular values, eigenvalues) for one coupled sector.

Fields:

  - `sectors::Vector{S}` — coupled sectors, sorted and unique (always non-dual, codomain convention)
  - `blocks::Vector{D}` — per-sector data vectors, one per sector
"""
struct FusedGradedVector{T, D <: AbstractVector{T}, S <: SectorRange} <:
    AbstractGradedArray{T, 1}
    sectors::Vector{S}
    blocks::Vector{D}
    function FusedGradedVector{T, D, S}(
            sectors::Vector{S},
            blocks::Vector{D}
        ) where {T, D <: AbstractVector{T}, S <: SectorRange}
        length(sectors) == length(blocks) || throw(ArgumentError("sectors and blocks must have the same length"))
        issorted(sectors) || throw(ArgumentError("sectors must be sorted"))
        allunique(sectors) || throw(ArgumentError("sectors must be unique"))
        return new{T, D, S}(sectors, blocks)
    end
end

function FusedGradedVector(
        sectors::Vector{S},
        blocks::Vector{D}
    ) where {T, S <: SectorRange, D <: AbstractVector{T}}
    return FusedGradedVector{T, D, S}(sectors, blocks)
end

function FusedGradedVector(pairs::AbstractVector{<:Pair})
    sectors = first.(pairs)
    blocks = last.(pairs)
    return FusedGradedVector(sectors, blocks)
end

# ========================  undef constructors  ========================

function FusedGradedVector{T, D, S}(
        ::UndefInitializer,
        sectors::Vector{S},
        ax::BlockedOneTo
    ) where {T, D <: AbstractVector{T}, S <: SectorRange}
    blk_axes = eachblockaxis(ax)
    length(blk_axes) == length(sectors) ||
        throw(ArgumentError("axis block count must match sectors length"))
    blks = [similar(D, (blk_axes[i],)) for i in eachindex(sectors)]
    return FusedGradedVector{T, D, S}(sectors, blks)
end

function FusedGradedVector{T}(
        ::UndefInitializer,
        sectors::Vector{S},
        ax::BlockedOneTo
    ) where {T, S <: SectorRange}
    return FusedGradedVector{T, Vector{T}, S}(undef, sectors, ax)
end

function FusedGradedVector{T}(
        ::UndefInitializer,
        sectors::Vector{<:SectorRange},
        blocklengths::Vector{Int}
    ) where {T}
    return FusedGradedVector{T}(undef, sectors, blockedrange(blocklengths))
end

# ========================  Accessors  ========================

BlockArrays.blocklength(v::FusedGradedVector) = length(v.sectors)
function BlockSparseArrays.blocktype(::Type{<:FusedGradedVector{T, D, S}}) where {T, D, S}
    return SectorVector{T, D, S}
end
BlockSparseArrays.blocktype(v::FusedGradedVector) = BlockSparseArrays.blocktype(typeof(v))
sectortype(::Type{<:FusedGradedVector{T, D, S}}) where {T, D, S} = S

function Base.axes(v::FusedGradedVector)
    return (gradedrange(v.sectors .=> length.(v.blocks)),)
end

Base.size(v::FusedGradedVector) = map(length, axes(v))
Base.eltype(::Type{FusedGradedVector{T}}) where {T} = T
Base.eltype(::Type{<:FusedGradedVector{T}}) where {T} = T

# ========================  Block indexing (primitive)  ========================

function Base.view(v::FusedGradedVector, I::Block{1})
    i = Int(I)
    return SectorVector(v.sectors[i], v.blocks[i])
end

# ========================  eachblockstoredindex  ========================

function BlockSparseArrays.eachblockstoredindex(v::FusedGradedVector)
    return (Block(i) for i in eachindex(v.sectors))
end

# ========================  fill! / zero!  ========================

function FI.zero!(v::FusedGradedVector)
    for b in v.blocks
        fill!(b, zero(eltype(v)))
    end
    return v
end

function Base.fill!(v::FusedGradedVector, val)
    iszero(val) || throw(
        ArgumentError("fill! with nonzero value is not supported for FusedGradedVector")
    )
    return FI.zero!(v)
end

# ========================  similar  ========================

function Base.similar(v::FusedGradedVector, ::Type{T}) where {T}
    new_blocks = [similar(b, T) for b in v.blocks]
    return FusedGradedVector(copy(v.sectors), new_blocks)
end

# ========================  show  ========================

function Base.summary(io::IO, v::FusedGradedVector)
    nblocks = length(v.sectors)
    print(io, nblocks, "-block ", typeof(v), " with sectors [")
    join(io, v.sectors, ", ")
    print(io, "]")
    return nothing
end

function Base.print_array(io::IO, v::FusedGradedVector)
    for (s, b) in zip(v.sectors, v.blocks)
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
    isempty(v.sectors) && return nothing
    Base.print_array(io, v)
    return nothing
end

function Base.show(io::IO, v::FusedGradedVector)
    nblocks = length(v.sectors)
    print(io, nblocks, "-block ", typeof(v))
    return nothing
end

# ========================  copy  ========================

Base.copy(v::FusedGradedVector) = FusedGradedVector(copy(v.sectors), map(copy, v.blocks))

# ======================== LinearAlgebra ======================
function LinearAlgebra.norm(A::FusedGradedVector, p::Real = 2)
    if p == Inf
        return maximum(Base.Fix2(LinearAlgebra.norm, p), A.blocks)
    elseif p > 0
        s = zero(float(real(eltype(A))))
        for (c, a) in zip(A.sectors, A.blocks)
            s += length(c) * LinearAlgebra.norm(a, p)^p
        end
        return s^inv(p)
    else
        throw(ArgumentError("Norm with non-positive p ($p) is not defined"))
    end
end

# ========================  Identity constructor  ========================

FusedGradedVector(v::FusedGradedVector) = v
