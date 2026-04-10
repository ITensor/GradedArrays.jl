"""
    GradedOneTo{S<:SectorRange}

Represents a graded axis — a collection of sectors with multiplicities and a dual flag.
This is the axis type for `AbelianGradedArray`.

Stores non-dual `SectorRange` values in `nondual_sectors`, multiplicities, and a single
`isdual` flag. The `sectors` accessor applies the `isdual` flag on the fly.
"""
struct GradedOneTo{S <: SectorRange} <: AbstractUnitRange{Int}
    nondual_sectors::Vector{S}
    multiplicities::Vector{Int}
    isdual::Bool
    function GradedOneTo(
            sectors::Vector{S}, multiplicities::Vector{Int}, isdual::Bool
        ) where {S <: SectorRange}
        length(sectors) == length(multiplicities) ||
            throw(ArgumentError("sectors and multiplicities must have the same length"))
        return new{S}(sectors, multiplicities, isdual)
    end
end
function GradedOneTo(
        sectors::Vector{S},
        multiplicities::Vector{Int}
    ) where {S <: SectorRange}
    return GradedOneTo(sectors, multiplicities, false)
end

# Primitive accessors
sector_multiplicities(g::GradedOneTo) = g.multiplicities
isdual(g::GradedOneTo) = g.isdual

# Derived accessors
sectors(g::GradedOneTo) = isdual(g) ? dual.(g.nondual_sectors) : g.nondual_sectors
Base.first(::GradedOneTo) = 1
BlockArrays.blocklength(g::GradedOneTo) = length(g.nondual_sectors)
BlockArrays.eachblockaxes1(g::GradedOneTo) = eachblockaxis(g)
Base.length(g::GradedOneTo) = sum(blocklengths(g); init = 0)

# sector_type, SymmetryStyle
sector_type(::Type{GradedOneTo{S}}) where {S} = S
SymmetryStyle(::Type{<:GradedOneTo{S}}) where {S} = SymmetryStyle(S)

# blocklengths: total length of each block (quantum_dimension * multiplicity)
function BlockArrays.blocklengths(g::GradedOneTo)
    return [
        quantum_dimension(s) * m
            for (s, m) in zip(g.nondual_sectors, sector_multiplicities(g))
    ]
end

quantum_dimension(g::GradedOneTo) = length(g)
dataaxistype(::Type{<:GradedOneTo}) = Base.OneTo{Int}

function trivial(::Type{GradedOneTo{S}}) where {S}
    return gradedrange([trivial(S) => 1])
end
trivial(g::GradedOneTo) = trivial(typeof(g))

"""
    gradedrange(xs::AbstractVector{<:Pair})

Generic fallback that converts sector keys via `to_sector` before constructing `GradedOneTo`.
This supports NamedTuple keys (for sector products) and other non-standard key types.
"""
function gradedrange(xs::AbstractVector{<:Pair})
    isempty(xs) && throw(
        ArgumentError("Cannot create GradedOneTo from empty vector without type info")
    )
    converted = [to_sector(first(p)) => last(p) for p in xs]
    return gradedrange(converted)
end

# ========================  BlockSparseArrays interface  ========================

function BlockSparseArrays.eachblockaxis(g::GradedOneTo)
    return [
        SectorOneTo(s, m)
            for (s, m) in zip(sectors(g), sector_multiplicities(g))
    ]
end

function BlockSparseArrays.mortar_axis(axs::AbstractVector{<:SectorOneTo})
    isempty(axs) && throw(
        ArgumentError("Cannot create GradedOneTo from empty vector without type info")
    )
    allequal(isdual, axs) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    d = isdual(first(axs))
    # Store non-dual sectors; apply isdual via dual() if needed
    ss = [d ? dual(sector(si)) : sector(si) for si in axs]
    ms = [sector_multiplicity(si) for si in axs]
    g = GradedOneTo(ss, ms)
    return d ? dual(g) : g
end

# Non-abelian fusion: flatten GradedOneTo elements into a single GradedOneTo
function BlockSparseArrays.mortar_axis(axs::AbstractVector{<:GradedOneTo})
    isempty(axs) && throw(
        ArgumentError("Cannot create GradedOneTo from empty vector without type info")
    )
    return mortar_axis(mapreduce(eachblockaxis, vcat, axs))
end

# ========================  × with GradedOneTo  ========================

function KroneckerArrays.:×(g::GradedOneTo, s::SectorRange)
    return ×(g, to_gradedrange(s))
end
function KroneckerArrays.:×(s::SectorRange, g::GradedOneTo)
    return ×(to_gradedrange(s), g)
end
function KroneckerArrays.:×(g1::GradedOneTo, g2::GradedOneTo)
    v = vec([a × b for a in eachblockaxis(g1), b in eachblockaxis(g2)])
    return mortar_axis(v)
end

# dual, flip, flip_dual, adjoint
function dual(g::GradedOneTo)
    return GradedOneTo(g.nondual_sectors, sector_multiplicities(g), !isdual(g))
end
function flip(g::GradedOneTo)
    # Conjugate labels but keep stored sectors non-dual
    new_nondual = [SectorRange(dual(label(s))) for s in g.nondual_sectors]
    return GradedOneTo(new_nondual, sector_multiplicities(g), !isdual(g))
end
flip_dual(g::GradedOneTo) = isdual(g) ? flip(g) : g
Base.adjoint(g::GradedOneTo) = dual(g)

to_gradedrange(g::GradedOneTo) = g

# ========================  Block indexing on GradedOneTo  ========================

# Merge groups of blocks into single blocks.
# Each block of `I` groups source blocks that merge into one destination block.
function Base.getindex(
        g::GradedOneTo, I::AbstractBlockVector{<:Block{1}}
    )
    ea = eachblockaxis(g)
    dest_si = map(blocks(I)) do group
        src_sis = [ea[Int(b)] for b in group]
        total_mult = sum(sector_multiplicity, src_sis)
        return SectorOneTo(sector(first(src_sis)), total_mult)
    end
    return mortar_axis(collect(dest_si))
end

# Splitting: each BlockIndexRange{1} selects a sub-range within a source block.
# Produces one dest block per entry.
function Base.getindex(
        g::GradedOneTo, I::AbstractVector{<:BlockIndexRange{1}}
    )
    ea = eachblockaxis(g)
    dest_si = map(I) do bir
        b = Int(bir.block)
        r = only(bir.indices)
        src_si = ea[b]
        qdim = quantum_dimension(sector(src_si))
        # multiplicity of the sub-range: sub-range length / quantum dimension
        sub_mult = div(length(r), qdim)
        return SectorOneTo(sector(src_si), sub_mult)
    end
    return mortar_axis(collect(dest_si))
end

# Bounds checking (needed for AbstractArray scalar indexing)
function Base.checkindex(::Type{Bool}, g::GradedOneTo, i::Int)
    return 1 <= i <= length(g)
end

# Equality and hashing
function Base.isequal(a::GradedOneTo, b::GradedOneTo)
    return isequal(a.nondual_sectors, b.nondual_sectors) &&
        isequal(sector_multiplicities(a), sector_multiplicities(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::GradedOneTo, b::GradedOneTo) = isequal(a, b)
function Base.hash(g::GradedOneTo, h::UInt)
    return hash(g.nondual_sectors, hash(sector_multiplicities(g), hash(isdual(g), h)))
end

# Show
function Base.show(io::IO, g::GradedOneTo)
    print(io, "GradedOneTo(")
    print(io, "[")
    for (i, (s, m)) in enumerate(zip(sectors(g), sector_multiplicities(g)))
        i > 1 && print(io, ", ")
        show(io, label(s))
        print(io, " => ", m)
    end
    print(io, "])")
    isdual(g) && print(io, "'")
    return nothing
end

# ========================  gradedrange constructors  ========================

"""
    gradedrange(xs::AbstractVector{<:Pair{<:SectorRange, <:Integer}})

Construct a `GradedOneTo` from pairs of `SectorRange` to multiplicities.
All `SectorRange` values must have the same `isdual` flag.
Non-dual inputs produce a non-dual axis; dual inputs produce a dual axis.

# Examples

```julia
gradedrange([U1(0) => 2, U1(1) => 3])     # non-dual
gradedrange([U1(0)' => 2, U1(1)' => 3])   # dual
```
"""
function gradedrange(
        xs::AbstractVector{<:Pair{S, <:Integer}}
    ) where {S <: SectorRange}
    isempty(xs) && return GradedOneTo(S[], Int[])
    d = isdual(first(first(xs)))
    all(p -> isdual(first(p)) == d, xs) ||
        throw(ArgumentError("All SectorRange inputs must have the same isdual flag"))
    # Store non-dual sectors; apply isdual on the fly via dual()
    ss = S[d ? dual(first(p)) : first(p) for p in xs]
    ms = Int[last(p) for p in xs]
    g = GradedOneTo(ss, ms)
    return d ? dual(g) : g
end
