using Dictionaries: Dictionaries, AbstractDictionary, Dictionary
using MappedArrays: mappedarray

# A graded axis whose sectors each appear once, in sorted order, as in a TensorKit `GradedSpace`.
#
# The sectors are stored in their TensorKitSectors form so that a `GradedSpace` can be built from
# the stored vectors without copying. `I` is the TensorKitSectors sector type corresponding to
# the `Sector`, used to aid conversion to and from TensorKit. Subject to change.
#
# The vectors are held behind one reference, so a tuple of axes stays small. Tuple code over
# inline multi-field axes is much slower to compile.
mutable struct FusedGradedOneToData{I <: TKS.Sector}
    const tensorkit_sectors::Vector{I}
    const datalengths::Vector{Int}
end
struct FusedGradedOneTo{S <: Sector, I <: TKS.Sector} <: AbstractGradedOneTo{S}
    data::FusedGradedOneToData{I}
    isdual::Bool
    global function unchecked_fusedgradedoneto(
            data::FusedGradedOneToData{I}, isdual::Bool
        ) where {I <: TKS.Sector}
        return new{gradedarrays_sectortype(I), I}(data, isdual)
    end
end
function FusedGradedOneTo(
        tensorkit_sectors::Vector{I}, datalengths::Vector{Int}, isdual::Bool
    ) where {I <: TKS.Sector}
    length(tensorkit_sectors) == length(datalengths) ||
        throw(ArgumentError("sectors and datalengths must have the same length"))
    issortedunique(tensorkit_sectors) || throw(
        ArgumentError(
            "FusedGradedOneTo sectors must be sorted and unique: $(tensorkit_sectors)"
        )
    )
    return unchecked_fusedgradedoneto(
        FusedGradedOneToData(tensorkit_sectors, datalengths), isdual
    )
end

issortedunique(v) = all(((a, b),) -> isless(a, b), zip(v, Iterators.drop(v, 1)))

function FusedGradedOneTo(
        sectors::AbstractVector{S}, datalengths::AbstractVector{<:Integer},
        isdual::Bool = false
    ) where {S <: Sector}
    return FusedGradedOneTo(
        tensorkit_sectortype(S)[TKS.Sector(s) for s in sectors],
        collect(Int, datalengths), isdual
    )
end

# The keys must already be sorted and unique.
function FusedGradedOneTo(
        sector_datalengths::AbstractDictionary{S, Int}, isdual::Bool = false
    ) where {S <: Sector}
    return FusedGradedOneTo(
        collect(keys(sector_datalengths)), collect(sector_datalengths), isdual
    )
end

# ========================  primitive accessors  ========================

# Callers must not mutate the stored vectors, since a `GradedSpace` built from this axis aliases
# them.
TensorAlgebra.isdual(g::FusedGradedOneTo) = g.isdual
tensorkit_sectors(g::FusedGradedOneTo) = g.data.tensorkit_sectors
# Wrapped in a closure because `mappedarray` reads a bare type argument as the element type of
# the result rather than as the function producing it, and `Sector` is abstract.
sectors(g::FusedGradedOneTo) = mappedarray(c -> Sector(c), tensorkit_sectors(g))
datalengths(g::FusedGradedOneTo) = g.data.datalengths
sectordatalengths(g::FusedGradedOneTo) = SortedArrayDictionary(sectors(g), datalengths(g))

# Length 0 for a sector the axis does not carry.
getsectordatalengths(g::FusedGradedOneTo, c) = get(sectordatalengths(g), c, 0)

# The block index of sector `c`. Throws for an absent sector.
function sectorindex(g::FusedGradedOneTo, c)
    (hassector, t) = gettoken(sectordatalengths(g), c)
    hassector || throw(ArgumentError("sector $c is not in the axis"))
    return t
end

# ========================  setsectors  ========================

function setsectors(g::FusedGradedOneTo{S}, ss::Vector{S}) where {S}
    return setsectors(g, tensorkit_sectortype(S)[TKS.Sector(s) for s in ss])
end

# `cs` is stored as is, so axes set from the same vector share it. The loop doubles as the check
# that `cs` covers every stored sector.
function setsectors(g::FusedGradedOneTo{S, I}, cs::Vector{I}) where {S, I}
    gl, gd = tensorkit_sectors(g), datalengths(g)
    (cs === gl || cs == gl) && return g
    lens = Vector{Int}(undef, length(cs))
    i = 1
    for k in eachindex(cs)
        if i <= length(gl) && isequal(gl[i], cs[k])
            lens[k] = gd[i]
            i += 1
        else
            lens[k] = 0
        end
    end
    i > length(gl) || throw(
        ArgumentError("sectors $(cs) do not cover the axis support $(gl)")
    )
    return FusedGradedOneTo(cs, lens, isdual(g))
end

# ========================  dual, flip  ========================

function TensorAlgebra.dual(g::FusedGradedOneTo)
    return unchecked_fusedgradedoneto(g.data, !isdual(g))
end

# Conjugating the sectors can reorder them, so they are re-sorted.
function flip(g::FusedGradedOneTo)
    flipped = map(TKS.dual, tensorkit_sectors(g))
    perm = sortperm(flipped)
    return FusedGradedOneTo(flipped[perm], datalengths(g)[perm], !isdual(g))
end

# ========================  show  ========================

# Print `dual(fusedgradedrange([...]))` so the output round-trips through the constructor.
function Base.show(io::IO, g::FusedGradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "fusedgradedrange([")
    join(io, (s => m for (s, m) in zip(sectors(g), datalengths(g))), ", ")
    print(io, "])")
    isdual(g) && print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", g::FusedGradedOneTo)
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

# ========================  fusedgradedrange constructors  ========================

# Construct a non-dual `FusedGradedOneTo` from `sector => multiplicity` pairs. The sectors must
# already be sorted and unique.
function fusedgradedrange(xs::AbstractVector{<:Pair})
    return FusedGradedOneTo(
        map(p -> to_sector(first(p)), xs),
        Int[last(p) for p in xs],
        false
    )
end

# ========================  conversions between graded-axis types  ========================

FusedGradedOneTo(g::FusedGradedOneTo) = g

# Rejects unsorted or repeated sectors rather than re-sorting. Use `fusesectors` to merge them.
function FusedGradedOneTo(g::AbstractGradedOneTo)
    return FusedGradedOneTo(sectors(g), datalengths(g), isdual(g))
end

Base.convert(::Type{FusedGradedOneTo}, g::AbstractGradedOneTo) = FusedGradedOneTo(g)

# ========================  fusesectors  ========================

fusesectors(g::FusedGradedOneTo) = g

# Sort the sectors and sum the data lengths of repeated sectors.
function sortmergesectors(
        sectors::AbstractVector{S}, datalengths::AbstractVector{Int}, isdual::Bool
    ) where {S <: Sector}
    merged_sectors = S[]
    merged_datalengths = Int[]
    for p in sortperm(sectors)
        if !isempty(merged_sectors) && isequal(last(merged_sectors), sectors[p])
            merged_datalengths[end] += datalengths[p]
        else
            push!(merged_sectors, sectors[p])
            push!(merged_datalengths, datalengths[p])
        end
    end
    return FusedGradedOneTo(merged_sectors, merged_datalengths, isdual)
end
