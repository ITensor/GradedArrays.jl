# Stores `Sector` values in `sectors`, sector lengths, and a single `isdual` flag. The sectors
# carry no arrow of their own, so the flag is the axis's entire duality; it is applied per block
# by `eachblockaxis` (and hence `eachstructureaxis`). The fused (merged-sorted) form of the axis is
# computed once at construction and cached in `fused`, so `fusesectors` is a field read, and the
# `FusedGradedOneTo` conversion compares the stored sectors against that cache and throws for
# a non-canonical axis. `I` is that cache's stored sector type, fixed by `S` and a parameter
# only because a field type cannot be computed from one. Treat it as an implementation detail
# of the TensorKit conversion, liable to change.
"""
    GradedOneTo

A graded axis: a range carrying a sector for each of its blocks, as returned by
[`gradedrange`](@ref). Wrap it in `dual` for a dual axis.
"""
struct GradedOneTo{S <: Sector, I <: TKS.Sector} <: AbstractGradedOneTo{S}
    sectors::Vector{S}
    datalengths::Vector{Int}
    isdual::Bool
    fused::FusedGradedOneTo{S, I}
    function GradedOneTo(
            sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
        ) where {S <: Sector}
        length(sectors) == length(datalengths) ||
            throw(ArgumentError("sectors and datalengths must have the same length"))
        # One axis is graded by one symmetry, so the sectors need a single concrete type. Checked
        # here rather than in a signature because this is the only chokepoint every path goes
        # through, and a mixed vector still satisfies `Vector{S} where {S <: Sector}`.
        isconcretetype(S) || throw(
            ArgumentError(
                "a graded axis is graded by one symmetry, so its sectors need one concrete \
                type, got $(S) from $(sectors)"
            )
        )
        fused = sortmergesectors(sectors, datalengths, isdual)
        return new{S, tensorkit_sectortype(S)}(sectors, datalengths, isdual, fused)
    end
    # `fused` must equal the fused form of the other fields; unchecked.
    global function unchecked_gradedoneto(
            sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool,
            fused::FusedGradedOneTo{S, I}
        ) where {S <: Sector, I <: TKS.Sector}
        return new{S, I}(sectors, datalengths, isdual, fused)
    end
end
# Any sector vector, materialized for storage; the arrow defaults to non-dual.
function GradedOneTo(
        sectors::AbstractVector{S}, datalengths::AbstractVector{<:Integer},
        isdual::Bool = false
    ) where {S <: Sector}
    return GradedOneTo(collect(S, sectors), collect(Int, datalengths), isdual)
end

# Primitive accessors. The derived range-interface methods (`sectorlengths`, `first`, `axes`,
# `blocklength(s)`, `sectortype`, `FusionStyle`, `eachblockaxis`, ...) are shared via
# `AbstractGradedOneTo`.
datalengths(g::GradedOneTo) = g.datalengths
TensorAlgebra.isdual(g::GradedOneTo) = g.isdual
sectors(g::GradedOneTo) = g.sectors
# The fused (merged-sorted) form is precomputed at construction, so this is a field read.
fusesectors(g::GradedOneTo) = g.fused

# ========================  conversions between graded-axis types  ========================

GradedOneTo(g::GradedOneTo) = g
function GradedOneTo(g::AbstractGradedOneTo)
    return GradedOneTo(sectors(g), datalengths(g), isdual(g))
end
# An already-fused axis is its own fused form, so pass it through as the cache.
function GradedOneTo(g::FusedGradedOneTo)
    return unchecked_gradedoneto(collect(sectors(g)), datalengths(g), isdual(g), g)
end
Base.convert(::Type{GradedOneTo}, g::AbstractGradedOneTo) = GradedOneTo(g)

# The strict conversion (reject rather than re-sort a non-canonical axis) reduces to
# comparing the stored sectors with the precomputed fused form.
function FusedGradedOneTo(g::GradedOneTo)
    fused = fusesectors(g)
    sectors(g) == sectors(fused) || throw(
        ArgumentError(
            "FusedGradedOneTo requires fused and sorted sectors: $(sectors(g))"
        )
    )
    return fused
end

function trivial(::Type{<:GradedOneTo{S}}) where {S}
    return gradedrange([trivial(S) => 1])
end
trivial(g::GradedOneTo) = trivial(typeof(g))

TensorAlgebra.trivialrange(R::Type{<:GradedOneTo}) = trivial(R)
function TensorAlgebra.trivialrange(::Type{<:GradedOneTo{S}}, n::Integer) where {S}
    return gradedrange([trivial(S) => n])
end

# The ungraded extent of a graded range is the plain range over its total dimension, dropping
# sectors and the arrow so a range and its `dual` share an ungraded value.
TensorAlgebra.ungrade(g::GradedOneTo) = Base.OneTo(length(g))

# ========================  BlockSparseArrays interface  ========================

function mortar_axis(axs::AbstractVector{SectorOneTo{S}}) where {S}
    isempty(axs) && return GradedOneTo(S[], Int[])
    allequal_compat(isdual, axs) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    ss = S[sector(r) for r in axs]
    ms = Int[datalength(r) for r in axs]
    return GradedOneTo(ss, ms, isdual(first(axs)))
end

# Non-abelian fusion: flatten GradedOneTo elements into a single GradedOneTo
function mortar_axis(axs::AbstractVector{GradedOneTo{S, I}}) where {S, I}
    isempty(axs) && return GradedOneTo(S[], Int[])
    return mortar_axis(mapreduce(eachblockaxis, vcat, axs))
end

# dual, flip, flip_dual, adjoint
# `dual` reuses the parent's fused form with the arrow flipped (the merge is arrow-independent),
# so no re-fusion happens on the hot `conj`/`biaxes` path.
function TensorAlgebra.dual(g::GradedOneTo)
    return unchecked_gradedoneto(
        g.sectors, datalengths(g), !isdual(g), dual(fusesectors(g))
    )
end
function flip(g::GradedOneTo)
    return GradedOneTo(map(dual_sector, sectors(g)), datalengths(g), !isdual(g))
end
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
        return SectorOneTo(structure(first(src)), total_mult)
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
        return SectorOneTo(structure(src), sub_mult)
    end
    return mortar_axis(collect(dest))
end

# Combining graded axes in a broadcast: graded arrays never mix mismatched blocking or
# sectors, so the only valid combination is of equal axes, and the result is that axis.
# This preserves the `GradedOneTo` type, which the generic `BlockArrays.combine_blockaxes`
# would degrade to a plain blocked range (dropping the sectors and duality). Equality and
# hashing are shared via `AbstractGradedOneTo`.
function BlockArrays.combine_blockaxes(a::GradedOneTo, b::GradedOneTo)
    a == b || throw(DimensionMismatch("cannot combine unequal graded axes: $a and $b"))
    return a
end

# Show. Factor the `dual` to the outside — `dual(gradedrange([...]))` — rather
# than decorating each sector, so the printed form is compact and round-trips
# through the constructor.
function Base.show(io::IO, g::GradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "gradedrange(")
    show_sector_pairs(io, g)
    print(io, ")")
    isdual(g) && print(io, ")")
    return nothing
end

# The `[sector => length, ...]` pair list, shared by the round-tripping `show` above (wrapped in
# `gradedrange(...)`) and the compact `Dim` line a graded array prints for its axes.
function show_sector_pairs(io::IO, g::GradedOneTo)
    print(io, "[")
    join(io, (s => m for (s, m) in zip(g.sectors, datalengths(g))), ", ")
    print(io, "]")
    return nothing
end

# A graded array's `Dim` line: the bare pair list, with a trailing `(dual)` for a dual axis. The
# `gradedrange(...)` wrapper the standalone `show` adds is redundant next to the `Dim N:` label.
show_axis(io::IO, g::AbstractUnitRange) = show(io, g)
function show_axis(io::IO, g::GradedOneTo)
    show_sector_pairs(io, g)
    isdual(g) && print(io, " (dual)")
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
    gradedrange(xs::AbstractVector{<:Pair})

Construct a non-dual graded range from `sector => multiplicity` pairs, keyed by anything
[`Sector`](@ref) accepts, `NamedTuple` keys for sector products included. Wrap the result in
`dual` for a dual axis.

# Examples

```julia
gradedrange([U1(0) => 2, U1(1) => 3])          # non-dual
dual(gradedrange([U1(0) => 2, U1(1) => 3]))    # dual
```
"""
function gradedrange(xs::AbstractVector{<:Pair})
    return GradedOneTo(map(p -> Sector(first(p)), xs), Int[last(p) for p in xs], false)
end

# Route every key type `Sector` accepts to `gradedrange`. One method per key type rather than one
# over a `Union` of them, which inside `Pair{<:...}` is slow to subtype and a ready source of
# ambiguities. A container key must be
# non-empty, since an empty `Tuple` satisfies `Tuple{Vararg{E}}` for every `E` and would make the
# two element types' methods overlap. The bare `TKS.Sector` key is type piracy, allowlisted in
# the Aqua piracy test and tracked as a follow-up.
for element in (:Sector, :(TKS.Sector))
    for key in (
            element,
            :(Tuple{$element, Vararg{$element}}),
            :(NamedTuple{<:Any, <:Tuple{$element, Vararg{$element}}}),
        )
        @eval function TensorAlgebra.to_range(
                space::AbstractVector{<:Pair{K, <:Integer}}
            ) where {K <: $key}
            return gradedrange(space)
        end
    end
end
