module GradedArraysTensorKitExt

using GradedArrays: GradedArrays, SectorRange, isdual, label
using TensorKit: TensorKit, Vect

# Build a native TensorKit `GradedSpace` from non-abelian `sector => multiplicity` pairs.
# `GradedArrays.to_range` routes non-abelian sectors here (they have no block-sparse
# `GradedOneTo` representation). The sector's arrow rides inside the space, matching the
# TensorKit convention: a shared dual flag across the pairs becomes a dual space. Raw
# `TKS.Sector` keys are normalized to `SectorRange` so both key kinds are handled.
function GradedArrays.to_tensorkit_space(space::AbstractVector{<:Pair})
    reps = SectorRange.(first.(space))
    d = isdual(first(reps))
    all(r -> isdual(r) == d, reps) ||
        throw(ArgumentError("All sectors must have the same isdual flag"))
    irreptype = typeof(label(first(reps)))
    V = Vect[irreptype]((label(r) => m for (r, m) in zip(reps, last.(space)))...)
    return d ? TensorKit.dual(V) : V
end

end
