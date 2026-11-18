# The sectors of the result axis are those of the arguments in order, never merged or sorted, so a
# sector can repeat.

using BlockArrays: blocks
using TensorAlgebra: concatenate!

# `TensorAlgebra` folds `cat_axis` pairwise, so a binary method suffices.
function TensorAlgebra.cat_axis(a1::AbstractGradedOneTo, a2::AbstractGradedOneTo)
    return mortar_axis([a1, a2])
end

# The broadcast-based default fails since the arguments have different sizes.
function TensorAlgebra.cat_similar(::AbstractGradedStyle, ::Type{T}, ax, args...) where {T}
    return similar(first(args), T, ax)
end
