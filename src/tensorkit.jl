using TensorKit: TensorKit as TK, ElementarySpace, Vect

# Non-abelian `sector => multiplicity` pairs have no block-sparse `GradedOneTo` representation,
# so `to_range` routes them here to build a native TensorKit `GradedSpace`. Nothing here carries
# an arrow, so this is the non-dual builder. It is the entry point both for the `Sector` routing
# in GradedArrays and for a user-supplied list of TensorKit sectors passed to `to_range`.
# `Vect[S]` takes the pairs as a single iterable (rather than splatting), so a long sector list
# does not build a large tuple or hit vararg dispatch.
function to_tensorkit_space(space::AbstractVector{<:Pair{S}}) where {S <: TK.Sector}
    return Vect[S](space)
end

# A TensorKit `GradedSpace` holds each sector once, in sorted order: fused (no sector repeats) and
# sorted in `Sector` order (which matches TensorKit's), so a fused-sorted range maps to a
# `GradedSpace` with no reordering. `GradedArray` axes may be unfused/unsorted, and the `project` / `Array`
# conversions block-permute the dense data into this form at the TensorKit boundary.
is_fused_sorted(g::AbstractGradedOneTo) = (s = sectors(g); allunique(s) && issorted(s))
# Allocation-free via the cached fused form: canonical iff the stored sectors already equal it.
is_fused_sorted(g::GradedOneTo) = sectors(g) == sectors(fusesectors(g))

# Throwing wrapper: `ElementarySpace` demands a fused-sorted range.
function check_fused_sorted(g::AbstractGradedOneTo)
    is_fused_sorted(g) || throw(ArgumentError("axis sectors must be fused and sorted"))
    return g
end

# `GradedOneTo` <-> `ElementarySpace` converters. The sectors carry no arrow (duality is a
# separate flag), so build the non-dual side and apply the arrow.
function TK.ElementarySpace(g::AbstractGradedOneTo)
    check_fused_sorted(g)
    sp = to_tensorkit_space([c => m for (c, m) in zip(sectors(g), datalengths(g))])
    return isdual(g) ? dual(sp) : sp
end

# A `FusedGradedOneTo` stores a `GradedSpace`'s exact data — sorted parallel sector/length
# vectors plus an arrow (`Sector` order is the TensorKitSectors order it delegates to) — so
# transcribe the storage directly instead of re-validating pair by pair.
function TK.ElementarySpace(g::FusedGradedOneTo)
    sp = to_tensorkit_space(g)
    return isdual(g) ? dual(sp) : sp
end

# The non-dual space over the stored sectors; the arrow is applied by `ElementarySpace`.
function to_tensorkit_space(g::FusedGradedOneTo{S}) where {S}
    return to_tensorkit_space(Vect[tensorkitsectortype(S)], g)
end
# Dictionary-backed spaces (unbounded sector sets, e.g. `U1`): hand the labels and lengths to
# TensorKit's trusted already-sorted `SortedVectorDict` constructor — no sort and no per-pair
# insertion. The length vector is shared with the axis, and both sides treat it as immutable.
# TensorKit's pair constructor drops zero dims, so fall back to it in that (rare) case.
function to_tensorkit_space(
        ::Type{TK.GradedSpace{I, TK.SectorDict{I, Int}}}, g::FusedGradedOneTo
    ) where {I}
    labels = I[tensorkitsector(c) for c in sectors(g)]
    all(>(0), datalengths(g)) || return TK.GradedSpace{I, TK.SectorDict{I, Int}}(
        l => m for (l, m) in zip(labels, datalengths(g))
    )
    dims = TK.SectorDict{I, Int}(labels, datalengths(g))
    return TK.GradedSpace{I, TK.SectorDict{I, Int}}(dims, false)
end
# Tuple-backed spaces (finite sector sets, e.g. `Z2`) store a dense dimension tuple; their
# pair constructor is already a flat fill with nothing to skip.
function to_tensorkit_space(::Type{Sp}, g::FusedGradedOneTo) where {Sp <: ElementarySpace}
    return Sp(tensorkitsector(c) => m for (c, m) in zip(sectors(g), datalengths(g)))
end

# A dual space's duality belongs on the range's `isdual`, not on its sectors, so read the
# sectors off the non-dual side and re-apply the arrow to the range. Sort the pairs into
# `Sector` order.
function GradedOneTo(V::ElementarySpace)
    V0 = TK.isdual(V) ? TK.dual(V) : V
    ps = sort([c => TK.dim(V0, c) for c in TK.sectors(V0)]; by = p -> to_sector(first(p)))
    g = gradedrange(ps)
    return TK.isdual(V) ? dual(g) : g
end
