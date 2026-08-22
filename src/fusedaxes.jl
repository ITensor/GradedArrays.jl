# Fuse a group of leg axes into its coupled fused-sorted axis. The explicit branches avoid
# re-fusing what is already cached: an empty group is the trivial range, a single leg reads the
# cached fused form (`tensor_product` on one axis), and a multi-leg group reduces without a
# trivial init (which would add a pointless trivial×leg merge per call).
function fuseaxes(::Type{S}, axs::Tuple) where {S <: SectorRange}
    return reduce(tensor_product, axs; init = trivial_gradedrange(S))
end
fuseaxes(::Type{S}, axs::Tuple{}) where {S <: SectorRange} = trivial_gradedrange(S)
function fuseaxes(::Type{S}, axs::Tuple{AbstractGradedOneTo}) where {S <: SectorRange}
    return tensor_product(only(axs))
end
function fuseaxes(
        ::Type{S},
        axs::Tuple{AbstractGradedOneTo, AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
    ) where {S <: SectorRange}
    return reduce(tensor_product, axs)
end

"""
    FusedAxes{S,N}

Internal carrier for one side (codomain or domain) of a graded-array allocation: the per-leg
`leaves` together with their fused `root`, so a caller that already holds the fused form can
pass it along instead of re-fusing. The root depends only on the multiset of leaves (fusion is
order-independent), which is what lets a contraction carry an operand's stored coupled axis to
the output.
"""
struct FusedAxes{S <: SectorRange, N}
    leaves::NTuple{N, GradedOneTo{S}}
    root::FusedGradedOneTo{S}
end

# Checked/fusing construction; `FusedAxes(leaves, root)` is the trusted variant for callers that
# already hold the fused form (`root` must equal `fuseaxes` of the leaves).
function FusedAxes{S}(leaves::Tuple{Vararg{GradedOneTo{S}}}) where {S <: SectorRange}
    return FusedAxes(leaves, fuseaxes(S, leaves))
end
function FusedAxes(
        leaves::Tuple{GradedOneTo{S}, Vararg{GradedOneTo{S}}}
    ) where {S <: SectorRange}
    return FusedAxes{S}(leaves)
end

leaves(fa::FusedAxes) = fa.leaves
root(fa::FusedAxes) = fa.root
