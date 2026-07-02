"""
    GradedOneTo{S<:SectorRange}

Represents a graded axis — a collection of sectors with sector lengths and a dual flag.
This is the axis type for `AbelianGradedArray`.

Stores non-dual `SectorRange` values in `sectors`, sector lengths, and a single
`isdual` flag. The `sectors` accessor returns those stored non-dual sectors; query the
duality separately with `isdual`. The dual flag is applied per block by `eachblockaxis`
(and hence `eachsectoraxis`).
"""
struct GradedOneTo{S <: SectorRange} <: AbstractBlockedUnitRange{Int, Vector{Int}}
    sectors::Vector{S}
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
sectors(g::GradedOneTo) = g.sectors
sectorlengths(g::GradedOneTo) = length.(sectors(g))
Base.first(::GradedOneTo) = 1
# A `GradedOneTo` is 1-based and acts as its own axis, like `Base.OneTo`. Without this
# `axes` falls back to the `AbstractBlockedUnitRange` default, which returns a plain
# `BlockedOneTo` and drops the sectors — so e.g. `axes(view(dense, ::GradedOneTo...))`
# would lose the grading. Mirrors `axes(::AbelianGradedArray)` returning its graded axes.
Base.axes(g::GradedOneTo) = (g,)
BlockArrays.blocklasts(g::GradedOneTo) = cumsum(blocklengths(g))
BlockArrays.blocklength(g::GradedOneTo) = length(g.sectors)
BlockArrays.eachblockaxes1(g::GradedOneTo) = eachblockaxis(g)

# sectortype, SymmetryStyle
sectortype(::Type{GradedOneTo{S}}) where {S} = S
SymmetryStyle(::Type{<:GradedOneTo{S}}) where {S} = SymmetryStyle(S)

# blocklengths: total length of each block (length(sector) * multiplicity)
function BlockArrays.blocklengths(g::GradedOneTo)
    return [
        length(s) * m for (s, m) in zip(g.sectors, datalengths(g))
    ]
end
dataaxistype(::Type{<:GradedOneTo}) = Base.OneTo{Int}

function trivial(::Type{GradedOneTo{S}}) where {S}
    return gradedrange([trivial(S) => 1])
end
trivial(g::GradedOneTo) = trivial(typeof(g))

TensorAlgebra.trivialrange(R::Type{<:GradedOneTo}) = trivial(R)

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

function eachblockaxis(g::GradedOneTo)
    block_sectors = isdual(g) ? dual.(sectors(g)) : sectors(g)
    return [
        SectorOneTo(s, m)
            for (s, m) in zip(block_sectors, datalengths(g))
    ]
end
eachdataaxis(g::GradedOneTo) = data.(eachblockaxis(g))
eachsectoraxis(g::GradedOneTo) = sector.(eachblockaxis(g))

function mortar_axis(axs::AbstractVector{SectorOneTo{S}}) where {S}
    isempty(axs) && return GradedOneTo(S[], Int[])
    allequal(isdual, axs) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    d = isdual(first(axs))
    # Store non-dual sectors; apply isdual via dual() if needed
    ss = S[d ? dual(sector(r)) : sector(r) for r in axs]
    ms = Int[datalength(r) for r in axs]
    g = GradedOneTo(ss, ms)
    return d ? dual(g) : g
end

# Non-abelian fusion: flatten GradedOneTo elements into a single GradedOneTo
function mortar_axis(axs::AbstractVector{GradedOneTo{S}}) where {S}
    isempty(axs) && return GradedOneTo(S[], Int[])
    return mortar_axis(mapreduce(eachblockaxis, vcat, axs))
end

# ========================  × with GradedOneTo  ========================

function ×(g::GradedOneTo, s::SectorRange)
    return ×(g, to_gradedrange(s))
end
function ×(s::SectorRange, g::GradedOneTo)
    return ×(to_gradedrange(s), g)
end
function ×(g1::GradedOneTo, g2::GradedOneTo)
    v = vec([a × b for a in eachblockaxis(g1), b in eachblockaxis(g2)])
    return mortar_axis(v)
end

# dual, flip, flip_dual, adjoint
function dual(g::GradedOneTo)
    return GradedOneTo(g.sectors, datalengths(g), !isdual(g))
end
function flip(g::GradedOneTo)
    # Conjugate labels but keep stored sectors non-dual
    new_nondual = [SectorRange(dual(label(s))) for s in g.sectors]
    return GradedOneTo(new_nondual, datalengths(g), !isdual(g))
end
flip_dual(g::GradedOneTo) = isdual(g) ? flip(g) : g
Base.conj(g::GradedOneTo) = dual(g)

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

# Equality and hashing. Empty graded ranges have no sectors and the `isdual`
# flag has no observable effect, so any two empty graded ranges of the same
# sector type compare equal.
function Base.isequal(a::GradedOneTo, b::GradedOneTo)
    isempty(a.sectors) && isempty(b.sectors) && return true
    return isequal(a.sectors, b.sectors) &&
        isequal(datalengths(a), datalengths(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::GradedOneTo, b::GradedOneTo) = isequal(a, b)

# Combining graded axes in a broadcast: graded arrays never mix mismatched blocking or
# sectors, so the only valid combination is of equal axes, and the result is that axis.
# This preserves the `GradedOneTo` type, which the generic `BlockArrays.combine_blockaxes`
# would degrade to a plain blocked range (dropping the sectors and duality).
function BlockArrays.combine_blockaxes(a::GradedOneTo, b::GradedOneTo)
    a == b || throw(DimensionMismatch("cannot combine unequal graded axes: $a and $b"))
    return a
end
function Base.hash(g::GradedOneTo, h::UInt)
    isempty(g.sectors) && return hash(GradedOneTo, h)
    return hash(g.sectors, hash(datalengths(g), hash(isdual(g), h)))
end

# Show. Factor the `dual` to the outside — `dual(gradedrange([...]))` — rather
# than decorating each sector, so the printed form is compact and round-trips
# through the constructor.
function Base.show(io::IO, g::GradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "gradedrange([")
    join(
        io,
        (s => m for (s, m) in zip(g.sectors, datalengths(g))),
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
    join(io, g.sectors, ", ")
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
gradedrange([conj(U1(0)) => 2, conj(U1(1)) => 3])   # dual
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

# Build a graded range from a vector of sector-to-multiplicity pairs, e.g.
# `to_range([U1(0) => 2, U1(1) => 3])`, routed by symmetry: abelian sectors build a
# block-sparse `GradedOneTo`, while non-abelian sectors have no block-sparse representation
# and build a native TensorKit `GradedSpace` via `to_tensorkit_space`. Defined over each
# key type separately rather than a `Union` so each method stays specific enough not to
# capture unrelated `Pair` vectors. A raw `TKS.Sector` key is wrapped in `SectorRange` so
# `SymmetryStyle` consults the fusion rule rather than its `AbelianStyle` default.
function TensorAlgebra.to_range(
        space::AbstractVector{<:Pair{K, <:Integer}}
    ) where {K <: SectorRange}
    return if SymmetryStyle(K) === AbelianStyle()
        gradedrange(space)
    else
        to_tensorkit_space(space)
    end
end
function TensorAlgebra.to_range(
        space::AbstractVector{<:Pair{K, <:Integer}}
    ) where {K <: TKS.Sector}
    return if SymmetryStyle(SectorRange{K}) === AbelianStyle()
        gradedrange(space)
    else
        to_tensorkit_space(space)
    end
end

"""
    to_tensorkit_space(sectors)

Convert a vector of non-abelian `sector => multiplicity` pairs into a native TensorKit
`GradedSpace`. Non-abelian symmetries have no block-sparse (`GradedOneTo`) representation,
so `to_range` routes them here. Requires TensorKit to be loaded; the method that builds the
space lives in the GradedArrays–TensorKit extension.
"""
function to_tensorkit_space(space)
    return throw(
        ArgumentError(
            "building a native non-abelian graded space from $(space) requires TensorKit; \
            run `using TensorKit` to enable it"
        )
    )
end
