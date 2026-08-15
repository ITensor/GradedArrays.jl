# Graded concatenation, plugged into the `TensorAlgebra` concatenation hooks. The result axis is
# concat-order (`cat_axis` / `mortar_axis`): the arguments' sectors are kept in order and
# block-appended, never merged or sorted, so a sector can repeat (an unfused axis). Both backends
# place whole symmetry-allowed blocks through the sparse block containers; on the fusion backend the
# unfused result axis is carried by the `GradedArray` while its blocks scatter into the fused-sorted
# backing (via `viewblock`).

using BlockArrays: blocks
using TensorAlgebra: concatenate!

# `dual` on a graded axis is again a graded axis, so this binary method covers dual axes too,
# and `TensorAlgebra` folds `cat_axis` pairwise so binary suffices for any number of arguments.
function TensorAlgebra.cat_axis(a1::AbstractGradedOneTo, a2::AbstractGradedOneTo)
    return mortar_axis([a1, a2])
end

# Override the broadcast-based default: the arguments have different sizes, so allocate from the
# concatenated graded axes directly.
function TensorAlgebra.cat_similar(::AbstractGradedStyle, ::Type{T}, ax, args...) where {T}
    return similar(first(args), T, ax)
end
