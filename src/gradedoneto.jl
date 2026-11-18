# The sorted and merged form of the axis is computed at construction and cached in `fused`. `I` is
# the TensorKitSectors sector type corresponding to
# the `Sector`, used to aid conversion to and from TensorKit. Subject to change.
# The data is held behind one reference for the same reason as `FusedGradedOneToData`.
mutable struct GradedOneToData{S <: Sector, I <: TKS.Sector}
    const sectors::Vector{S}
    const datalengths::Vector{Int}
    const fused::FusedGradedOneToData{I}
end
"""
    GradedOneTo

A graded axis: a range carrying a sector for each of its blocks, as returned by
[`gradedrange`](@ref). Wrap it in `dual` for a dual axis.
"""
struct GradedOneTo{S <: Sector, I <: TKS.Sector} <: AbstractGradedOneTo{S}
    data::GradedOneToData{S, I}
    isdual::Bool
    # `data.fused` must equal the fused form of `data`; unchecked.
    global function unchecked_gradedoneto(
            data::GradedOneToData{S, I}, isdual::Bool
        ) where {S <: Sector, I <: TKS.Sector}
        return new{S, I}(data, isdual)
    end
end
function GradedOneTo(
        sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
    ) where {S <: Sector}
    length(sectors) == length(datalengths) ||
        throw(ArgumentError("sectors and datalengths must have the same length"))
    # Checked here rather than in the signature, since a vector of mixed sectors still
    # satisfies `Vector{S} where {S <: Sector}`.
    isconcretetype(S) || throw(
        ArgumentError(
            "a graded axis is graded by one symmetry, so its sectors need one concrete \
            type, got $(S) from $(sectors)"
        )
    )
    fused = sortmergesectors(sectors, datalengths, isdual)
    return unchecked_gradedoneto(GradedOneToData(sectors, datalengths, fused.data), isdual)
end
function GradedOneTo(
        sectors::AbstractVector{S}, datalengths::AbstractVector{<:Integer},
        isdual::Bool = false
    ) where {S <: Sector}
    return GradedOneTo(collect(S, sectors), collect(Int, datalengths), isdual)
end

datalengths(g::GradedOneTo) = g.data.datalengths
TensorAlgebra.isdual(g::GradedOneTo) = g.isdual
sectors(g::GradedOneTo) = g.data.sectors
fusesectors(g::GradedOneTo) = unchecked_fusedgradedoneto(g.data.fused, isdual(g))

# ========================  conversions between graded-axis types  ========================

GradedOneTo(g::GradedOneTo) = g
function GradedOneTo(g::AbstractGradedOneTo)
    return GradedOneTo(sectors(g), datalengths(g), isdual(g))
end
# An already-fused axis is its own fused form, so pass it through as the cache.
function GradedOneTo(g::FusedGradedOneTo)
    return unchecked_gradedoneto(
        GradedOneToData(collect(sectors(g)), datalengths(g), g.data), isdual(g)
    )
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

function mortar_axis(axs::AbstractVector{GradedOneTo{S, I}}) where {S, I}
    isempty(axs) && return GradedOneTo(S[], Int[])
    return mortar_axis(mapreduce(eachblockaxis, vcat, axs))
end

# Reuse the fused form, so `dual` does not fuse again.
function TensorAlgebra.dual(g::GradedOneTo)
    return unchecked_gradedoneto(g.data, !isdual(g))
end
function flip(g::GradedOneTo)
    return GradedOneTo(map(dual_sector, sectors(g)), datalengths(g), !isdual(g))
end
to_gradedrange(g::GradedOneTo) = g

# ========================  Block indexing on GradedOneTo  ========================

# Each block of `I` is a group of blocks to merge into one.
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

# Each `BlockIndexRange{1}` selects a sub-range of a block.
function Base.getindex(
        g::GradedOneTo, I::AbstractVector{<:BlockIndexRange{1}}
    )
    ea = eachblockaxis(g)
    dest = map(I) do bir
        b = Int(bir.block)
        r_range = only(bir.indices)
        src = ea[b]
        sub_mult = div(length(r_range), length(sector(src)))
        return SectorOneTo(structure(src), sub_mult)
    end
    return mortar_axis(collect(dest))
end

# The generic method would return a plain blocked range, dropping the sectors.
function BlockArrays.combine_blockaxes(a::GradedOneTo, b::GradedOneTo)
    a == b || throw(DimensionMismatch("cannot combine unequal graded axes: $a and $b"))
    return a
end

# Print `dual(gradedrange([...]))` so the output round-trips through the constructor.
function Base.show(io::IO, g::GradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "gradedrange(")
    show_sector_pairs(io, g)
    print(io, ")")
    isdual(g) && print(io, ")")
    return nothing
end

function show_sector_pairs(io::IO, g::GradedOneTo)
    print(io, "[")
    join(io, (s => m for (s, m) in zip(sectors(g), datalengths(g))), ", ")
    print(io, "]")
    return nothing
end

# The domain axes are shown un-dualized, as the constructor takes them. No trailing newline.
function show_biaxes(io::IO, a::AbstractArray)
    lines = [
        "$(name) Dim $(d):" => g
            for
            (name, group) in (("Codomain", axes_codomain(a)), ("Domain", axes_domain(a)))
            for (d, g) in enumerate(group)
    ]
    width = maximum(length ∘ first, lines; init = 0)
    for (i, (label, g)) in enumerate(lines)
        i > 1 && println(io)
        print(io, "  ", rpad(label, width), " ")
        show(io, g)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", g::GradedOneTo)
    summary(io, g)
    isempty(g) && return nothing
    print(io, ":\n  sectors: ")
    isdual(g) && print(io, "dual.(")
    print(io, "[")
    join(io, sectors(g), ", ")
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
[`Sector`](@ref) accepts. A `Tuple` or `NamedTuple` key is read as the product of its factors,
as with [`sectorproduct`](@ref), so nested products are flattened and `Trivial` factors drop
out. Wrap the result in `dual` for a dual axis.

# Examples

```julia
gradedrange([U1(0) => 2, U1(1) => 3])          # non-dual
dual(gradedrange([U1(0) => 2, U1(1) => 3]))    # dual
gradedrange([(fU1(1), U1(1)) => 2])            # keyed by fU1(1) × U1(1)
```
"""
function gradedrange(xs::AbstractVector{<:Pair})
    return GradedOneTo(map(p -> to_sector(first(p)), xs), Int[last(p) for p in xs], false)
end

# One method per key type, since a `Union` inside `Pair{<:...}` is slow to subtype and causes
# ambiguities. Container keys must be non-empty, since an empty `Tuple` would match both element
# types. The bare `TKS.Sector` key is type piracy, allowed in the Aqua test.
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
