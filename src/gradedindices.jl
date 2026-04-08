"""
    GradedIndices{I<:TKS.Sector}

Represents a graded axis — a collection of sector labels with multiplicities and a dual flag.
This is the axis type for `AbelianArray` and replaces the old `GradedUnitRange` type alias
for new code paths.

Stores raw labels, multiplicities, and a single dual flag. The `sectors` accessor returns
`SectorRange` values on the fly.
"""
struct GradedIndices{I <: TKS.Sector}
    labels::Vector{I}
    multiplicities::Vector{Int}
    isdual::Bool
    function GradedIndices(
            labels::Vector{I}, multiplicities::Vector{Int}, isdual::Bool
        ) where {I <: TKS.Sector}
        length(labels) == length(multiplicities) ||
            throw(ArgumentError("labels and multiplicities must have the same length"))
        return new{I}(labels, multiplicities, isdual)
    end
end
function GradedIndices(
        labels::Vector{I},
        multiplicities::Vector{Int}
    ) where {I <: TKS.Sector}
    return GradedIndices(labels, multiplicities, false)
end

# Primitive accessors
labels(g::GradedIndices) = g.labels
sector_multiplicities(g::GradedIndices) = g.multiplicities
isdual(g::GradedIndices) = g.isdual

# Derived accessors
sectors(g::GradedIndices) = SectorRange.(labels(g), isdual(g))
BlockArrays.blocklength(g::GradedIndices) = length(labels(g))
function Base.length(g::GradedIndices)
    return sum(
        i -> TKS.dim(labels(g)[i]) * sector_multiplicities(g)[i],
        eachindex(labels(g));
        init = 0
    )
end

# sector_type, SymmetryStyle
sector_type(::Type{GradedIndices{I}}) where {I} = SectorRange{I}
SymmetryStyle(::Type{<:GradedIndices{I}}) where {I} = SymmetryStyle(SectorRange{I})

# blocklengths: total length of each block (quantum_dimension * multiplicity)
function BlockArrays.blocklengths(g::GradedIndices)
    return [TKS.dim(l) * m for (l, m) in zip(labels(g), sector_multiplicities(g))]
end

quantum_dimension(g::GradedIndices) = length(g)

function trivial(::Type{GradedIndices{I}}) where {I}
    return gradedrange([one(I) => 1])
end
trivial(g::GradedIndices) = trivial(typeof(g))

"""
    gradedrange(xs::AbstractVector{<:Pair})

Generic fallback that converts sector keys via `to_sector` before constructing `GradedIndices`.
This supports NamedTuple keys (for sector products) and other non-standard key types.
"""
function gradedrange(xs::AbstractVector{<:Pair})
    isempty(xs) && throw(
        ArgumentError("Cannot create GradedIndices from empty vector without type info")
    )
    converted = [to_sector(first(p)) => last(p) for p in xs]
    return gradedrange(converted)
end

# ========================  BlockSparseArrays interface  ========================

function BlockSparseArrays.eachblockaxis(g::GradedIndices)
    return [
        SectorIndices(l, m, isdual(g))
            for (l, m) in zip(labels(g), sector_multiplicities(g))
    ]
end

function BlockSparseArrays.mortar_axis(axs::AbstractVector{<:SectorIndices})
    isempty(axs) && throw(
        ArgumentError("Cannot create GradedIndices from empty vector without type info")
    )
    allequal(isdual, axs) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    ls = [label(si) for si in axs]
    ms = [sector_multiplicity(si) for si in axs]
    return GradedIndices(ls, ms, isdual(first(axs)))
end

# Non-abelian fusion: flatten GradedIndices elements into a single GradedIndices
function BlockSparseArrays.mortar_axis(axs::AbstractVector{<:GradedIndices})
    isempty(axs) && throw(
        ArgumentError("Cannot create GradedIndices from empty vector without type info")
    )
    return mortar_axis(mapreduce(eachblockaxis, vcat, axs))
end

# ========================  × with GradedIndices  ========================

function KroneckerArrays.:×(g::GradedIndices, s::SectorRange)
    return ×(g, to_gradedrange(s))
end
function KroneckerArrays.:×(s::SectorRange, g::GradedIndices)
    return ×(to_gradedrange(s), g)
end
function KroneckerArrays.:×(g1::GradedIndices, g2::GradedIndices)
    v = vec([a × b for a in eachblockaxis(g1), b in eachblockaxis(g2)])
    return mortar_axis(v)
end

# dual, flip, flip_dual, adjoint
dual(g::GradedIndices) = GradedIndices(labels(g), sector_multiplicities(g), !isdual(g))
function flip(g::GradedIndices)
    return GradedIndices(dual.(labels(g)), sector_multiplicities(g), !isdual(g))
end
flip_dual(g::GradedIndices) = isdual(g) ? flip(g) : g
Base.adjoint(g::GradedIndices) = dual(g)

to_gradedrange(g::GradedIndices) = g

# ========================  Block indexing on GradedIndices  ========================

# Merge groups of blocks into single blocks.
# Each block of `I` groups source blocks that merge into one destination block.
function Base.getindex(
        g::GradedIndices, I::AbstractBlockVector{<:Block{1}}
    )
    ea = eachblockaxis(g)
    dest_si = map(blocks(I)) do group
        src_sis = [ea[Int(b)] for b in group]
        total_mult = sum(sector_multiplicity, src_sis)
        return SectorIndices(label(first(src_sis)), total_mult, isdual(first(src_sis)))
    end
    return mortar_axis(collect(dest_si))
end

# Splitting: each BlockIndexRange{1} selects a sub-range within a source block.
# Produces one dest block per entry.
function Base.getindex(
        g::GradedIndices, I::AbstractVector{<:BlockIndexRange{1}}
    )
    ea = eachblockaxis(g)
    dest_si = map(I) do bir
        b = Int(bir.block)
        r = only(bir.indices)
        src_si = ea[b]
        qdim = TKS.dim(label(src_si))
        # multiplicity of the sub-range: sub-range length / quantum dimension
        sub_mult = div(length(r), qdim)
        return SectorIndices(label(src_si), sub_mult, isdual(src_si))
    end
    return mortar_axis(collect(dest_si))
end

# Bounds checking (needed for AbstractArray scalar indexing)
function Base.checkindex(::Type{Bool}, g::GradedIndices, i::Int)
    return 1 <= i <= length(g)
end

# Equality and hashing
function Base.:(==)(a::GradedIndices, b::GradedIndices)
    return labels(a) == labels(b) &&
        sector_multiplicities(a) == sector_multiplicities(b) &&
        isdual(a) == isdual(b)
end
function Base.hash(g::GradedIndices, h::UInt)
    return hash(labels(g), hash(sector_multiplicities(g), hash(isdual(g), h)))
end

# Show
function Base.show(io::IO, g::GradedIndices)
    print(io, "GradedIndices(")
    print(io, "[")
    for (i, (l, m)) in enumerate(zip(labels(g), sector_multiplicities(g)))
        i > 1 && print(io, ", ")
        show(io, l)
        print(io, " => ", m)
    end
    print(io, "])")
    isdual(g) && print(io, "'")
    return nothing
end

# ========================  gradedrange constructors  ========================

"""
    gradedrange(xs::AbstractVector{<:Pair{<:TKS.Sector, <:Integer}})

Construct a `GradedIndices` from pairs of raw sector labels to multiplicities.
The result is non-dual; use `'` (adjoint) or `dual()` to make it dual.

# Examples

```julia
gradedrange([U1(0) => 2, U1(1) => 3])
gradedrange([U1(0) => 2, U1(1) => 3])'  # dual
```
"""
function gradedrange(xs::AbstractVector{<:Pair{I, <:Integer}}) where {I <: TKS.Sector}
    ls = I[first(p) for p in xs]
    ms = Int[last(p) for p in xs]
    return GradedIndices(ls, ms, false)
end

"""
    gradedrange(xs::AbstractVector{<:Pair{<:SectorRange, <:Integer}})

Construct a `GradedIndices` from pairs of `SectorRange` labels to multiplicities.
All `SectorRange` values must have the same `isdual` flag.

# Examples

```julia
gradedrange([U1(0)' => 2, U1(1)' => 3])  # dual
gradedrange([U1(0) => 2, U1(1) => 3])     # non-dual
```
"""
function gradedrange(
        xs::AbstractVector{<:Pair{<:SectorRange{I}, <:Integer}}
    ) where {I <: TKS.Sector}
    isempty(xs) && return GradedIndices(I[], Int[], false)
    d = isdual(first(first(xs)))
    all(p -> isdual(first(p)) == d, xs) ||
        throw(ArgumentError("All SectorRange inputs must have the same isdual flag"))
    ls = I[label(first(p)) for p in xs]
    ms = Int[last(p) for p in xs]
    return GradedIndices(ls, ms, d)
end
