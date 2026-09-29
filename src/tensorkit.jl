using TensorKit: TensorKit as TK, ElementarySpace, Vect

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
    sp = Vect[tensorkit_sectortype(sectortype(g))](
        TKS.Sector(c) => m for (c, m) in zip(sectors(g), datalengths(g))
    )
    return isdual(g) ? dual(sp) : sp
end

# A `FusedGradedOneTo` stores a `GradedSpace`'s exact data, so transcribe the storage instead of
# re-validating pair by pair. TensorKit holds a space's dimensions either as the same sorted
# vectors the axis has (`DictGradedSpace`, for `U1` and the like) or as a dense dimension per
# sector (`TupleGradedSpace`, for `Z2`), and `Vect[I]` fixes which from the sector type, so the
# branch folds at compile time. Only the first can take our vectors as they are, and its pair
# constructor drops zero dims.
function TK.ElementarySpace(g::FusedGradedOneTo{S}) where {S}
    I = tensorkit_sectortype(S)
    Sp = Vect[I]
    cs, ms = tensorkit_sectors(g), datalengths(g)
    sp = if Sp <: TK.DictGradedSpace && all(>(0), ms)
        Sp(TK.SectorDict{I, Int}(cs, ms), false)
    else
        Sp(c => m for (c, m) in zip(cs, ms))
    end
    return isdual(g) ? dual(sp) : sp
end

# A dual space's duality belongs on the range's `isdual`, not on its sectors, so read the
# sectors off the non-dual side and re-apply the arrow to the range. Sort the pairs into
# `Sector` order.
function GradedOneTo(V::ElementarySpace)
    V0 = TK.isdual(V) ? TK.dual(V) : V
    ps = sort([c => TK.dim(V0, c) for c in TK.sectors(V0)]; by = p -> Sector(first(p)))
    g = gradedrange(ps)
    return TK.isdual(V) ? dual(g) : g
end
