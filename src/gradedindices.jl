"""
    GradedIndices{I<:TKS.Sector}

Represents a graded axis — a collection of sector labels with multiplicities and a dual flag.
This is the axis type for `AbelianArray` and replaces the old `GradedUnitRange` type alias
for new code paths.

Stores raw labels, multiplicities, and a single dual flag. Constructs `SectorRange` values
on the fly via the `sectorranges` accessor.
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
sectors(g::GradedIndices) = isdual(g) ? dual.(labels(g)) : labels(g)
sectorranges(g::GradedIndices) = SectorRange.(labels(g), isdual(g))
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
