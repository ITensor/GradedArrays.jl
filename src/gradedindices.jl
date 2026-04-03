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

# sector_type
sector_type(::Type{GradedIndices{I}}) where {I} = SectorRange{I}

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

# ========================  × with GradedIndices  ========================

function KroneckerArrays.:×(g::GradedIndices, s::SectorRange)
    return ×(g, to_gradedrange(s))
end
function KroneckerArrays.:×(s::SectorRange, g::GradedIndices)
    return ×(to_gradedrange(s), g)
end
function KroneckerArrays.:×(g1::GradedIndices, g2::GradedIndices)
    v = [si1 × si2 for si1 in _each_sectorindices(g1), si2 in _each_sectorindices(g2)]
    ls = [label(si) for si in vec(v)]
    ms = [sector_multiplicity(si) for si in vec(v)]
    d = isempty(v) ? false : isdual(first(v))
    return GradedIndices(ls, ms, d)
end

function _each_sectorindices(g::GradedIndices)
    return [
        SectorIndices(l, m, isdual(g))
            for (l, m) in zip(labels(g), sector_multiplicities(g))
    ]
end

# dual, flip, adjoint
dual(g::GradedIndices) = GradedIndices(labels(g), sector_multiplicities(g), !isdual(g))
function flip(g::GradedIndices)
    return GradedIndices(dual.(labels(g)), sector_multiplicities(g), !isdual(g))
end
Base.adjoint(g::GradedIndices) = dual(g)

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
