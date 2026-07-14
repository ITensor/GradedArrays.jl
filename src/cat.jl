# Concatenation of graded arrays through the `TensorAlgebra` concatenation interface.
#
# `TensorAlgebra.cat(args...; dims)` resolves the combined `cat_style` (here `AbelianGradedStyle`),
# allocates the destination with `cat_similar`, and places the arguments with `cat_copyto!`. The
# default `directsum` is a plain `cat` (its `directsum_style` is `ReshapeFusion`), so it shares this
# path. The concatenation is concat-order: sectors of the arguments are kept in order and
# block-appended, never merged or sorted.
#
# This concat-order path is only correct for `AbelianGradedArray`. Fused graded arrays
# (`FusedGradedVector`/`FusedGradedMatrix`, carrying `GradedStyle`) direct-sum onto a merged,
# rotated basis, so they need a different implementation and are deliberately not handled here.

using BlockArrays: blocks
using TensorAlgebra: cat!

# The concatenated axis is the concat-order block-append of the graded ranges, preserving
# sector order (no merging/sorting). `mortar_axis` builds it and checks that the arrows match.
# `dual` on a `GradedOneTo` is again a `GradedOneTo` (with the `isdual` flag set), so this single
# method also covers dual axes. `TensorAlgebra` folds the variadic `cat_axis` pairwise, so this
# binary method suffices for concatenating any number of arguments.
function TensorAlgebra.cat_axis(a1::GradedOneTo, a2::GradedOneTo)
    return mortar_axis([a1, a2])
end

# Allocate the destination from the concat-order graded axes `ax` and the promoted eltype `T`.
# The arguments have different sizes, so the default broadcast-based `cat_similar` (which assumes
# elementwise-matching axes) does not apply here.
function TensorAlgebra.cat_similar(::AbelianGradedStyle, ::Type{T}, ax, args...) where {T}
    return similar(first(args), T, ax)
end

# Materialize by concatenating the block containers, placing each argument's blocks into the
# destination's diagonal hyper-block without scalar indexing. The inner `cat!` over block
# containers routes through the generic (style-`nothing`) placement: because `AbelianBlocks` is an
# `AbstractSparseArray`, its `zero!` and offset range-assignment visit only the stored
# (symmetry-allowed) blocks, whereas the dense generic path would touch forbidden positions and
# fail on `view` of an unstored block.
function TensorAlgebra.cat_copyto!(dest, ::AbelianGradedStyle, dims, args...)
    cat!(blocks(dest), blocks.(args)...; dims = dims)
    return dest
end

# Route `Base.cat` on abelian graded arrays through the same machinery, so plain `cat` matches
# `TensorAlgebra.cat`.
function Base._cat(dims, as::AbelianGradedArray...)
    return TensorAlgebra.concatenate(dims, as...)
end
