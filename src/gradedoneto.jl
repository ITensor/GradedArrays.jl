"""
    GradedOneTo{S<:Sector}

Represents a graded axis — a collection of sectors with sector lengths and a dual flag.
This is the axis type for `GradedArray`.

Stores `Sector` values in `sectors`, sector lengths, and a single `isdual` flag. The sectors
carry no arrow of their own, so the flag is the axis's entire duality; it is applied per block
by `eachblockaxis` (and hence `eachstructureaxis`). The fused (merged-sorted) form of the axis is
computed once at construction and cached in `fused`, so `fusesectors` is a field read; the
`FusedGradedOneTo` conversion compares the stored sectors against that cache and throws for
a non-canonical axis.
"""
struct GradedOneTo{S <: Sector} <: AbstractGradedOneTo{S}
    sectors::Vector{S}
    datalengths::Vector{Int}
    isdual::Bool
    fused::FusedGradedOneTo{S}
    function GradedOneTo(
            sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
        ) where {S <: Sector}
        length(sectors) == length(datalengths) ||
            throw(ArgumentError("sectors and datalengths must have the same length"))
        # One axis is graded by one symmetry, so the sectors need a single concrete type. Checked
        # here because it is the only chokepoint every path goes through: `Sector` is abstract, so
        # a mixed vector still satisfies `Vector{S} where {S <: Sector}` with `S` bound to
        # `Sector` itself, which an outer method cannot tell apart from a good one.
        isconcretetype(S) || throw(
            ArgumentError(
                "a graded axis is graded by one symmetry, so its sectors need one concrete \
                type, got $(S) from $(sectors)"
            )
        )
        merged_sectors, merged_datalengths = mergesectors(sectors, datalengths)
        fused = FusedGradedOneTo(merged_sectors, merged_datalengths, isdual)
        return new{S}(sectors, datalengths, isdual, fused)
    end
    # `fused` must equal the fused form of the other fields; unchecked.
    global function unchecked_gradedoneto(
            sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool,
            fused::FusedGradedOneTo{S}
        ) where {S <: Sector}
        return new{S}(sectors, datalengths, isdual, fused)
    end
end
# Arrow defaults to non-dual.
function GradedOneTo(
        sectors::Vector{S},
        datalengths::Vector{Int}
    ) where {S <: Sector}
    return GradedOneTo(sectors, datalengths, false)
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
    return GradedOneTo(collect(sectors(g)), datalengths(g), isdual(g))
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

function trivial(::Type{GradedOneTo{S}}) where {S}
    return gradedrange([trivial(S) => 1])
end
trivial(g::GradedOneTo) = trivial(typeof(g))

TensorAlgebra.trivialrange(R::Type{<:GradedOneTo}) = trivial(R)
function TensorAlgebra.trivialrange(::Type{GradedOneTo{S}}, n::Integer) where {S}
    return gradedrange([trivial(S) => n])
end

# The ungraded extent of a graded range is the plain range over its total dimension, dropping
# sectors and the arrow so a range and its `dual` share an ungraded value.
TensorAlgebra.ungrade(g::GradedOneTo) = Base.OneTo(length(g))

"""
    gradedrange(xs::AbstractVector{<:Pair})

Generic fallback that converts sector keys via `Sector` before constructing `GradedOneTo`.
This supports NamedTuple keys (for sector products) and other non-standard key types.
"""
function gradedrange(xs::AbstractVector{<:Pair})
    isempty(xs) && throw(
        ArgumentError("Cannot create GradedOneTo from empty vector without type info")
    )
    # Built directly rather than by recursing into the typed method above, whose `S` a converted
    # but still mixed key vector would bind to the abstract `Sector`. `GradedOneTo` rejects that.
    sectors = map(p -> Sector(first(p)), xs)
    return GradedOneTo(sectors, Int[last(p) for p in xs], false)
end

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
function mortar_axis(axs::AbstractVector{GradedOneTo{S}}) where {S}
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
    return GradedOneTo(map(charge_conjugate, sectors(g)), datalengths(g), !isdual(g))
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
        return SectorOneTo(sector(first(src)), isdual(g), total_mult)
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
        return SectorOneTo(sector(src), isdual(g), sub_mult)
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
    gradedrange(xs::AbstractVector{<:Pair{<:Sector, <:Integer}})

Construct a non-dual `GradedOneTo` from `sector => multiplicity` pairs. Wrap the result in
`dual` for a dual axis.

# Examples

```julia
gradedrange([U1(0) => 2, U1(1) => 3])          # non-dual
dual(gradedrange([U1(0) => 2, U1(1) => 3]))    # dual
```
"""
function gradedrange(
        xs::AbstractVector{<:Pair{S, <:Integer}}
    ) where {S <: Sector}
    return GradedOneTo(S[first(p) for p in xs], Int[last(p) for p in xs], false)
end

# A `GradedOneTo` stores bare sectors and carries the arrow itself, so an oriented sector vector
# is rejected by `Sector` rather than silently dropping or absorbing the arrows.
function GradedOneTo(
        sectors::Vector{<:OrientedSector}, datalengths::Vector{Int}, isdual::Bool
    )
    return GradedOneTo(map(Sector, sectors), datalengths, isdual)
end

# Build a graded range from a vector of sector-to-multiplicity pairs, e.g.
# `to_range([U1(0) => 2, U1(1) => 3])`. Both abelian and non-abelian sectors build a `GradedOneTo`
# (`GradedArray` represents non-abelian sectors via its coupled `FusedGradedMatrix`). The key
# types are the ones `Sector` accepts, so an axis descriptor takes the spellings a sector does:
# a bare sector from either package, or a tuple or named tuple of them. One method per key type
# rather than one over a `Union` of them, since a union inside `Pair{<:...}` is slow to subtype
# and a ready source of ambiguities. A container key is admitted only when its elements are
# sectors, since a bare `Tuple` says nothing about symmetry and would capture pairs vectors that
# have nothing to do with it, and non-empty, since an empty `Tuple` satisfies `Tuple{Vararg{E}}`
# for every `E` and would make the two element types' methods overlap. The bare `TensorKitSectors.Sector` key is deliberate type piracy
# (GradedArrays owns neither `to_range` nor `TKS.Sector`), kept for now so raw sectors work as
# axis descriptors (e.g. behind `Index([FermionNumber(0) => 2])`); it is allowlisted in the Aqua
# piracy test and tracked as a follow-up to rehome onto a GradedArrays-owned entry point.
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

"""
    to_tensorkit_space(sectors)

Convert a vector of `sector => multiplicity` pairs into a native TensorKit `GradedSpace`, used by
the TensorKit interop layer to move a `GradedOneTo` into TensorKit's space representation. The
method that builds the space is defined in `src/tensorkit.jl`.
"""
function to_tensorkit_space(space)
    return throw(
        ArgumentError(
            "cannot build a TensorKit graded space from $(space): expected a vector of \
            `sector => multiplicity` pairs"
        )
    )
end

# A `Sector` carries no arrow, so a `Sector`-keyed pairs vector always describes a non-dual
# space: hand the TensorKitSectors counterparts to the label-keyed builder in `src/tensorkit.jl`.
# A dual space is `dual` of this one, which is distinct from a space of conjugated sectors and is
# the form a dual index must take for contraction.
function to_tensorkit_space(space::AbstractVector{<:Pair{S}}) where {S <: Sector}
    return to_tensorkit_space([TKS.Sector(first(p)) => last(p) for p in space])
end
