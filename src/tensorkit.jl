using TensorKit: TensorKit as TK, ElementarySpace, Vect

# A TensorKit `GradedSpace` holds each sector once, in sorted order.
is_fused_sorted(g::AbstractGradedOneTo) = (s = sectors(g); allunique(s) && issorted(s))
# Allocation-free using the cached fused form.
is_fused_sorted(g::GradedOneTo) = sectors(g) == sectors(fusesectors(g))

function check_fused_sorted(g::AbstractGradedOneTo)
    is_fused_sorted(g) || throw(ArgumentError("axis sectors must be fused and sorted"))
    return g
end

function TK.ElementarySpace(g::AbstractGradedOneTo)
    check_fused_sorted(g)
    sp = Vect[tensorkit_sectortype(sectortype(g))](
        TKS.Sector(c) => m for (c, m) in zip(sectors(g), datalengths(g))
    )
    return isdual(g) ? dual(sp) : sp
end

# Hands TensorKit the stored vectors, so the space aliases the axis. Only the dictionary-backed
# spaces (not `Z2`'s) can take them directly, and only without zero dimensions, which the pair
# constructor would drop.
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

function GradedOneTo(V::ElementarySpace)
    V0 = TK.isdual(V) ? TK.dual(V) : V
    ps = sort([c => TK.dim(V0, c) for c in TK.sectors(V0)]; by = p -> Sector(first(p)))
    g = gradedrange(ps)
    return TK.isdual(V) ? dual(g) : g
end

function FusedGradedOneTo(V::ElementarySpace)
    V0 = TK.isdual(V) ? TK.dual(V) : V
    cs = collect(TK.sectors(V0))
    return FusedGradedOneTo(cs, Int[TK.dim(V0, c) for c in cs], TK.isdual(V))
end
