"""
    GradedOneTo{I<:TKS.Sector}

Represents a graded axis — a collection of sector labels with multiplicities and a dual flag.
This is the axis type for `AbelianGradedArray` and replaces the old `GradedOneTo` type alias
for new code paths.

Stores raw labels, multiplicities, and a single dual flag. The `sectors` accessor returns
`SectorRange` values on the fly.
"""
struct GradedOneTo{I <: TKS.Sector} <: AbstractUnitRange{Int}
    labels::Vector{I}
    multiplicities::Vector{Int}
    isdual::Bool
    function GradedOneTo(
            labels::Vector{I}, multiplicities::Vector{Int}, isdual::Bool
        ) where {I <: TKS.Sector}
        length(labels) == length(multiplicities) ||
            throw(ArgumentError("labels and multiplicities must have the same length"))
        return new{I}(labels, multiplicities, isdual)
    end
end
function GradedOneTo(
        labels::Vector{I},
        multiplicities::Vector{Int}
    ) where {I <: TKS.Sector}
    return GradedOneTo(labels, multiplicities, false)
end

# Primitive accessors
labels(g::GradedOneTo) = g.labels
sector_multiplicities(g::GradedOneTo) = g.multiplicities
isdual(g::GradedOneTo) = g.isdual

# Derived accessors
sectors(g::GradedOneTo) = SectorRange.(labels(g), isdual(g))
BlockArrays.blocklength(g::GradedOneTo) = length(labels(g))
function Base.length(g::GradedOneTo)
    return sum(
        i -> quantum_dimension(sectors(g)[i]) * sector_multiplicities(g)[i],
        eachindex(labels(g));
        init = 0
    )
end

# sector_type, SymmetryStyle
sector_type(::Type{GradedOneTo{I}}) where {I} = SectorRange{I}
SymmetryStyle(::Type{<:GradedOneTo{I}}) where {I} = SymmetryStyle(SectorRange{I})

# blocklengths: total length of each block (quantum_dimension * multiplicity)
function BlockArrays.blocklengths(g::GradedOneTo)
    return [
        quantum_dimension(s) * m for (s, m) in zip(sectors(g), sector_multiplicities(g))
    ]
end

quantum_dimension(g::GradedOneTo) = length(g)
dataaxistype(::Type{<:GradedOneTo}) = Base.OneTo{Int}

function trivial(::Type{GradedOneTo{I}}) where {I}
    return gradedrange([one(I) => 1])
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
        SectorOneTo(l, m, isdual(g))
            for (l, m) in zip(labels(g), sector_multiplicities(g))
    ]
end

function BlockSparseArrays.mortar_axis(axs::AbstractVector{<:SectorOneTo})
    isempty(axs) && throw(
        ArgumentError("Cannot create GradedOneTo from empty vector without type info")
    )
    allequal(isdual, axs) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    ls = [label(si) for si in axs]
    ms = [sector_multiplicity(si) for si in axs]
    return GradedOneTo(ls, ms, isdual(first(axs)))
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
dual(g::GradedOneTo) = GradedOneTo(labels(g), sector_multiplicities(g), !isdual(g))
function flip(g::GradedOneTo)
    return GradedOneTo(dual.(labels(g)), sector_multiplicities(g), !isdual(g))
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
        return SectorOneTo(label(first(src_sis)), total_mult, isdual(first(src_sis)))
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
        return SectorOneTo(label(src_si), sub_mult, isdual(src_si))
    end
    return mortar_axis(collect(dest_si))
end

# Bounds checking (needed for AbstractArray scalar indexing)
function Base.checkindex(::Type{Bool}, g::GradedOneTo, i::Int)
    return 1 <= i <= length(g)
end

# Equality and hashing
function Base.isequal(a::GradedOneTo, b::GradedOneTo)
    return isequal(labels(a), labels(b)) &&
        isequal(sector_multiplicities(a), sector_multiplicities(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::GradedOneTo, b::GradedOneTo) = isequal(a, b)
function Base.hash(g::GradedOneTo, h::UInt)
    return hash(labels(g), hash(sector_multiplicities(g), hash(isdual(g), h)))
end

# Show
function Base.show(io::IO, g::GradedOneTo)
    print(io, "GradedOneTo(")
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

Construct a `GradedOneTo` from pairs of raw sector labels to multiplicities.
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
    return GradedOneTo(ls, ms, false)
end

"""
    gradedrange(xs::AbstractVector{<:Pair{<:SectorRange, <:Integer}})

Construct a `GradedOneTo` from pairs of `SectorRange` labels to multiplicities.
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
    isempty(xs) && return GradedOneTo(I[], Int[], false)
    d = isdual(first(first(xs)))
    all(p -> isdual(first(p)) == d, xs) ||
        throw(ArgumentError("All SectorRange inputs must have the same isdual flag"))
    ls = I[label(first(p)) for p in xs]
    ms = Int[last(p) for p in xs]
    return GradedOneTo(ls, ms, d)
end
