module GradedArraysTensorKitExt

using GradedArrays: GradedArrays, SectorRange, isdual, label
using TensorKit: TensorKit, Vect

# Build a native TensorKit `GradedSpace` from non-abelian `sector => multiplicity` pairs.
# `GradedArrays.to_range` routes non-abelian sectors here (they have no block-sparse
# `GradedOneTo` representation). This layer works in GradedArrays terms: normalize keys to
# `SectorRange` (a raw `TKS.Sector` is non-dual) and read off the shared arrow, then hand the
# sector labels and that arrow to the pure-TensorKit builder.
function GradedArrays.to_tensorkit_space(space::AbstractVector{<:Pair})
    reps = SectorRange.(first.(space))
    d = isdual(first(reps))
    all(r -> isdual(r) == d, reps) ||
        throw(ArgumentError("All sectors must have the same isdual flag"))
    return _to_tensorkit_space([label(r) => m for (r, m) in zip(reps, last.(space))], d)
end

# Pure TensorKit: the keys are sector labels and the arrow is a flag. A raw sector carries no
# duality, so the arrow rides inside the space as a whole-space `dual(V)` (distinct from a
# space of dual sectors, and the form a dual index must take for contraction).
function _to_tensorkit_space(
        space::AbstractVector{<:Pair{S}},
        isdual::Bool
    ) where {S <: TensorKit.Sector}
    V = Vect[S](space...)
    return isdual ? TensorKit.dual(V) : V
end

end
