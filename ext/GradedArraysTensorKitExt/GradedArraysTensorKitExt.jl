module GradedArraysTensorKitExt

using GradedArrays: GradedArrays
using TensorKit: TensorKit, ElementarySpace, Vect

# Non-abelian `sector => multiplicity` pairs have no block-sparse `GradedOneTo` representation,
# so `GradedArrays.to_range` routes them here to build a native TensorKit `GradedSpace`. A raw
# TensorKit sector carries no arrow, so this is the non-dual builder. It is the entry point
# both for the `SectorRange` routing in GradedArrays and for a user-supplied list of TensorKit
# sectors passed to `to_range`. `Vect[S]` takes the pairs as a single iterable (rather than
# splatting), so a long sector list does not build a large tuple or hit vararg dispatch.
function GradedArrays.to_tensorkit_space(
        space::AbstractVector{<:Pair{S}}
    ) where {S <: TensorKit.Sector}
    return Vect[S](space)
end

# A TensorKit space is a first-class graded axis under the direct-wrap design, so `dual` on
# one flips its arrow. This is the whole-space dual the `SectorRange` routing applies.
GradedArrays.dual(V::ElementarySpace) = TensorKit.dual(V)

end
