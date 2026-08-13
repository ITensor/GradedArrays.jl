"""
    FusedGradedOneTo{S<:SectorRange}

A graded axis whose sectors are **fused and sorted**: each sector appears once and the
sectors are in sorted order, the canonical form matching the requirement TensorKit places on
a `GradedSpace` and the coupled-sector axes of a [`FusedGradedMatrix`](@ref).

Stores a `Dictionary` mapping each (non-dual) `SectorRange` to its data length
(multiplicity), plus a single `isdual` flag — the single-axis analog of a
[`FusedGradedMatrix`](@ref)'s `codomain`/`domain` dictionaries. The dictionary keys are
sorted and unique (the merged invariant is intrinsic to the dictionary; the sorted invariant
is checked at construction). Unlike a [`GradedOneTo`](@ref), which may hold repeated or
unsorted sectors (the intermediate state of a not-yet-merged fusion), a `FusedGradedOneTo`
is always in canonical fused form.
"""
struct FusedGradedOneTo{S <: SectorRange} <: AbstractGradedOneTo{S}
    sector_datalengths::Dictionary{S, Int}
    isdual::Bool
    function FusedGradedOneTo(
            sector_datalengths::Dictionary{S, Int}, isdual::Bool
        ) where {S <: SectorRange}
        issorted(keys(sector_datalengths)) || throw(
            ArgumentError(
                "FusedGradedOneTo sectors must be sorted: $(keys(sector_datalengths))"
            )
        )
        return new{S}(sector_datalengths, isdual)
    end
end

# Default to a non-dual axis.
function FusedGradedOneTo(sector_datalengths::Dictionary{S, Int}) where {S <: SectorRange}
    return FusedGradedOneTo(sector_datalengths, false)
end

# Vector convenience: keys must already be unique (the `Dictionary` constructor enforces it)
# and sorted (checked above).
function FusedGradedOneTo(
        sectors::Vector{S}, datalengths::Vector{Int}, isdual::Bool
    ) where {S <: SectorRange}
    length(sectors) == length(datalengths) ||
        throw(ArgumentError("sectors and datalengths must have the same length"))
    return FusedGradedOneTo(Dictionary{S, Int}(sectors, datalengths), isdual)
end
function FusedGradedOneTo(
        sectors::Vector{S},
        datalengths::Vector{Int}
    ) where {S <: SectorRange}
    return FusedGradedOneTo(sectors, datalengths, false)
end

# Primitive accessors. `sectors`/`datalengths` return vectors (in fused/sorted order) to
# match the `GradedOneTo` interface; the underlying storage is a `Dictionary`. The remaining
# range-interface methods are shared via `AbstractGradedOneTo`.
TensorAlgebra.isdual(g::FusedGradedOneTo) = g.isdual
sectors(g::FusedGradedOneTo) = collect(keys(g.sector_datalengths))
datalengths(g::FusedGradedOneTo) = collect(values(g.sector_datalengths))

# ========================  dual, flip  ========================

# `dual` flips the arrow only; the stored (non-dual) sectors and their order are unchanged,
# so the fused+sorted invariant is preserved.
TensorAlgebra.dual(g::FusedGradedOneTo) = FusedGradedOneTo(g.sector_datalengths, !isdual(g))

# `flip` conjugates the sector labels and flips the arrow (matching `GradedOneTo`), leaving
# the block sectors unchanged. Dualizing the labels generally reorders them, so re-sort to
# restore the canonical fused form.
function flip(g::FusedGradedOneTo)
    new_nondual = [SectorRange(dual(label(s))) for s in sectors(g)]
    perm = sortperm(new_nondual)
    return FusedGradedOneTo(new_nondual[perm], datalengths(g)[perm], !isdual(g))
end

# ========================  equality, hashing  ========================

# Empty graded ranges have no sectors and the `isdual` flag has no observable effect, so any
# two empty ranges of the same sector type compare equal.
function Base.isequal(a::FusedGradedOneTo, b::FusedGradedOneTo)
    isempty(a.sector_datalengths) && isempty(b.sector_datalengths) && return true
    return isequal(sectors(a), sectors(b)) &&
        isequal(datalengths(a), datalengths(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::FusedGradedOneTo, b::FusedGradedOneTo) = isequal(a, b)
function Base.hash(g::FusedGradedOneTo, h::UInt)
    isempty(g.sector_datalengths) && return hash(FusedGradedOneTo, h)
    return hash(sectors(g), hash(datalengths(g), hash(isdual(g), h)))
end

# ========================  show  ========================

# Factor the `dual` to the outside — `dual(fusedgradedrange([...]))` — so the printed form is
# compact and round-trips through the constructor.
function Base.show(io::IO, g::FusedGradedOneTo)
    isdual(g) && print(io, "dual(")
    print(io, "fusedgradedrange([")
    join(io, (s => m for (s, m) in pairs(g.sector_datalengths)), ", ")
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
    join(io, keys(g.sector_datalengths), ", ")
    print(io, "]")
    isdual(g) && print(io, ")")
    println(io)
    Base.print_array(io, g)
    return nothing
end

# ========================  fusedgradedrange constructors  ========================

"""
    fusedgradedrange(xs::AbstractVector{<:Pair{<:SectorRange, <:Integer}})

Construct a [`FusedGradedOneTo`](@ref) from `sector => multiplicity` pairs, merging repeated
sectors and sorting into the canonical fused order. All `SectorRange` keys must share the
same `isdual` flag.
"""
function fusedgradedrange(xs::AbstractVector{<:Pair{S, <:Integer}}) where {S <: SectorRange}
    isempty(xs) && return FusedGradedOneTo(Dictionary{S, Int}(), false)
    d = isdual(first(first(xs)))
    all(p -> isdual(first(p)) == d, xs) ||
        throw(ArgumentError("All SectorRange inputs must have the same isdual flag"))
    # Store non-dual sectors; merge multiplicities of repeated sectors, then sort.
    merged = Dict{S, Int}()
    for p in xs
        s = d ? dual(first(p)) : first(p)
        merged[s] = get(merged, s, 0) + last(p)
    end
    ss = sort!(collect(keys(merged)))
    sl = Dictionary{S, Int}(ss, [merged[s] for s in ss])
    return FusedGradedOneTo(sl, d)
end

# Convert a (possibly unmerged/unsorted) `GradedOneTo` into a `FusedGradedOneTo`. Repeated or
# unsorted sectors are merged and sorted; the arrow is preserved.
function fusedgradedrange(g::GradedOneTo)
    prs = [(isdual(g) ? dual(s) : s) => m for (s, m) in zip(sectors(g), datalengths(g))]
    return fusedgradedrange(prs)
end

# Idempotent: already in canonical fused form.
fusedgradedrange(g::FusedGradedOneTo) = g

to_gradedrange(g::FusedGradedOneTo) = GradedOneTo(sectors(g), datalengths(g), isdual(g))
