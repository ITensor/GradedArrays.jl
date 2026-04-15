"""
    GradedOneTo{S<:SectorRange}

Represents a graded axis — a collection of sectors with sector lengths and a dual flag.
This is the axis type for `AbelianGradedArray`.

Stores non-dual `SectorRange` values in `nondual_sectors`, sector lengths, and a single
`isdual` flag. The `sectors` accessor applies the `isdual` flag on the fly.
"""
struct GradedOneTo{S <: SectorRange} <: AbstractBlockedUnitRange{Int, Vector{Int}}
    nondual_sectors::Vector{S}
    datalengths::Vector{Int}
    isdual::Bool
    function GradedOneTo(
            sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
        ) where {S <: SectorRange}
        length(sectors) == length(datalengths) ||
            throw(ArgumentError("sectors and datalengths must have the same length"))
        return new{S}(sectors, datalengths, isdual)
    end
end
function GradedOneTo(
        sectors::Vector{S},
        datalengths::Vector{Int}
    ) where {S <: SectorRange}
    return GradedOneTo(sectors, datalengths, false)
end

# Primitive accessors
datalengths(g::GradedOneTo) = g.datalengths
isdual(g::GradedOneTo) = g.isdual

# Derived accessors
sectors(g::GradedOneTo) = isdual(g) ? dual.(g.nondual_sectors) : g.nondual_sectors
sectorlengths(g::GradedOneTo) = length.(sectors(g))
Base.first(::GradedOneTo) = 1
BlockArrays.blocklasts(g::GradedOneTo) = cumsum(blocklengths(g))
BlockArrays.blocklength(g::GradedOneTo) = length(g.nondual_sectors)
BlockArrays.eachblockaxes1(g::GradedOneTo) = eachblockaxis(g)

# sectortype, SymmetryStyle
sectortype(::Type{GradedOneTo{S}}) where {S} = S
SymmetryStyle(::Type{<:GradedOneTo{S}}) where {S} = SymmetryStyle(S)

# blocklengths: total length of each block (length(sector) * multiplicity)
function BlockArrays.blocklengths(g::GradedOneTo)
    return [
        length(s) * m for (s, m) in zip(g.nondual_sectors, datalengths(g))
    ]
end
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
            for (s, m) in zip(sectors(g), datalengths(g))
    ]
end
eachdataaxis(g::GradedOneTo) = data.(eachblockaxis(g))
eachsectoraxis(g::GradedOneTo) = sector.(eachblockaxis(g))

function BlockSparseArrays.mortar_axis(axs::AbstractVector{<:SectorOneTo})
    isempty(axs) && throw(
        ArgumentError("Cannot create GradedOneTo from empty vector without type info")
    )
    allequal(isdual, axs) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    d = isdual(first(axs))
    # Store non-dual sectors; apply isdual via dual() if needed
    ss = [d ? dual(sector(r)) : sector(r) for r in axs]
    ms = [datalength(r) for r in axs]
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
    return GradedOneTo(g.nondual_sectors, datalengths(g), !isdual(g))
end
function flip(g::GradedOneTo)
    # Conjugate labels but keep stored sectors non-dual
    new_nondual = [SectorRange(dual(label(s))) for s in g.nondual_sectors]
    return GradedOneTo(new_nondual, datalengths(g), !isdual(g))
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
    dest = map(blocks(I)) do group
        src = [ea[Int(b)] for b in group]
        total_mult = sum(datalength, src)
        return SectorOneTo(sector(first(src)), total_mult)
    end
    return mortar_axis(collect(dest))
end

# Splitting: each BlockIndexRange{1} selects a sub-range within a source block.
# Produces one dest block per entry.
function Base.getindex(
        g::GradedOneTo, I::AbstractVector{<:BlockIndexRange{1}}
    )
    ea = eachblockaxis(g)
    dest = map(I) do bir
        b = Int(bir.block)
        r_range = only(bir.indices)
        src = ea[b]
        # multiplicity of the sub-range: sub-range length / sector length
        sub_mult = div(length(r_range), length(sector(src)))
        return SectorOneTo(sector(src), sub_mult)
    end
    return mortar_axis(collect(dest))
end

# Bounds checking (needed for AbstractArray scalar indexing)
function Base.checkindex(::Type{Bool}, g::GradedOneTo, i::Int)
    return 1 <= i <= length(g)
end

# Equality and hashing
function Base.isequal(a::GradedOneTo, b::GradedOneTo)
    return isequal(a.nondual_sectors, b.nondual_sectors) &&
        isequal(datalengths(a), datalengths(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::GradedOneTo, b::GradedOneTo) = isequal(a, b)
function Base.hash(g::GradedOneTo, h::UInt)
    return hash(g.nondual_sectors, hash(datalengths(g), hash(isdual(g), h)))
end

# Show. Factor the `dual` to the outside — `dual(gradedrange([...]))` — rather
# than decorating each sector, so the printed form is compact and round-trips
# through the constructor.
function Base.show(io::IO, g::GradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "gradedrange([")
    join(
        io,
        (s => m for (s, m) in zip(g.nondual_sectors, datalengths(g))),
        ", "
    )
    print(io, "])")
    isdual(g) && print(io, ")")
    return nothing
end

# Show a "sectors: ..." line between the default AbstractArray summary and the
# block-separated element listing inherited from AbstractBlockedUnitRange. For
# dual axes the sectors are shown as `dual.([...])`.
function Base.show(io::IO, ::MIME"text/plain", g::GradedOneTo)
    summary(io, g)
    isempty(g) && return nothing
    print(io, ":\n  sectors: ")
    isdual(g) && print(io, "dual.(")
    print(io, "[")
    join(io, g.nondual_sectors, ", ")
    print(io, "]")
    isdual(g) && print(io, ")")
    println(io)
    Base.print_array(io, g)
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
