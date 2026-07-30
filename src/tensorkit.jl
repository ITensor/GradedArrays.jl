using TensorKit: TensorKit as TK, ElementarySpace, Vect

# Non-abelian `sector => multiplicity` pairs have no block-sparse `GradedOneTo` representation,
# so `to_range` routes them here to build a native TensorKit `GradedSpace`. A raw TensorKit
# sector carries no arrow, so this is the non-dual builder. It is the entry point both for the
# `SectorRange` routing in GradedArrays and for a user-supplied list of TensorKit sectors passed
# to `to_range`. `Vect[S]` takes the pairs as a single iterable (rather than splatting), so a
# long sector list does not build a large tuple or hit vararg dispatch.
function to_tensorkit_space(space::AbstractVector{<:Pair{S}}) where {S <: TK.Sector}
    return Vect[S](space)
end

# A TensorKit `GradedSpace` holds each sector once, in sorted order. GradedArrays' fused-and-sorted
# ranges satisfy this, so a range that is unfused (a sector repeats) or unsorted cannot be converted
# yet. This restriction (and the initial `FusionArray` form) will be lifted once the unfused/unsorted
# case applies an implicit basis change. The sort order is GradedArrays' `SectorRange` order, which
# matches TensorKit's for every sector, so a sorted range maps to a `GradedSpace` with no reordering.
function check_fused_sorted(g::GradedOneTo)
    secs = sectors(g)
    allunique(secs) ||
        throw(ArgumentError("axis sectors must be fused (each sector appears once)"))
    issorted(secs) || throw(ArgumentError("axis sectors must be sorted"))
    return g
end

# `GradedOneTo` <-> `ElementarySpace` converters. `sectors` gives the non-dual sector labels
# (duality is a separate flag), so build the non-dual side and apply the arrow.
function TK.ElementarySpace(g::GradedOneTo)
    check_fused_sorted(g)
    sp = to_tensorkit_space([c => m for (c, m) in zip(sectors(g), datalengths(g))])
    return isdual(g) ? dual(sp) : sp
end

# Sort the pairs into `SectorRange` order.
function GradedOneTo(V::ElementarySpace)
    V0 = TK.isdual(V) ? TK.dual(V) : V
    ps = sort([c => TK.dim(V0, c) for c in TK.sectors(V0)]; by = p -> SectorRange(first(p)))
    g = gradedrange(ps)
    return TK.isdual(V) ? dual(g) : g
end
