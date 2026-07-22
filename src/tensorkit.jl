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

# A TensorKit space is a first-class graded axis under the direct-wrap design, so `dual` on one
# flips its arrow. This is the whole-space dual the `SectorRange` routing applies.
dual(V::ElementarySpace) = TK.dual(V)

# `GradedOneTo` <-> `ElementarySpace` converters. `sectors` gives the non-dual sector labels
# (duality is a separate flag), so build the non-dual side and apply the arrow.
function TK.ElementarySpace(g::GradedOneTo)
    sp = to_tensorkit_space([c => m for (c, m) in zip(sectors(g), datalengths(g))])
    return isdual(g) ? dual(sp) : sp
end

# TensorKit orders sectors by its own convention (e.g. U1 by `|charge|` then sign), which differs
# from the `SectorRange` order a `GradedOneTo` uses (U1 numerically), so re-sort the pairs.
function GradedOneTo(V::ElementarySpace)
    V0 = TK.isdual(V) ? TK.dual(V) : V
    ps = sort([c => TK.dim(V0, c) for c in TK.sectors(V0)]; by = p -> SectorRange(first(p)))
    g = gradedrange(ps)
    return TK.isdual(V) ? dual(g) : g
end
