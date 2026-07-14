# Concatenation of graded arrays through the `TensorAlgebra` `Concatenate` machinery.
#
# `TensorAlgebra.cat(args...; dims)` builds a lazy `Concatenated{<:AbelianGradedStyle}` (the
# abelian graded broadcast style), whose axes come from `cat_axis` and whose materialization runs
# through `copyto!`. The default `directsum` is a plain `cat` (its `directsum_style` is
# `ReshapeFusion`), so it shares this path. The concatenation is concat-order: sectors of the
# arguments are kept in order and block-appended, never merged or sorted.
#
# This concat-order path is only correct for `AbelianGradedArray`. Fused graded arrays
# (`FusedGradedVector`/`FusedGradedMatrix`, carrying `GradedStyle`) direct-sum onto a merged,
# rotated basis, so they need a different implementation and are deliberately not handled here.

using BlockArrays: blocks
using TensorAlgebra: Concatenated, cat!

# The concatenated axis is the concat-order block-append of the graded ranges, preserving
# sector order (no merging/sorting). `mortar_axis` builds it and checks that the arrows match.
# `dual` on a `GradedOneTo` is again a `GradedOneTo` (with the `isdual` flag set), so this single
# method also covers dual axes. `TensorAlgebra` folds the variadic `cat_axis` pairwise, so this
# binary method suffices for concatenating any number of arguments.
function TensorAlgebra.cat_axis(a1::GradedOneTo, a2::GradedOneTo)
    return mortar_axis([a1, a2])
end

# Allocate the destination from the concat-order graded axes `ax` and the promoted eltype `T`.
# The arguments have different sizes, so the generic broadcast-based `similar` for `Concatenated`
# (which assumes elementwise-matching axes) does not apply here.
function Base.similar(concat::Concatenated{<:AbelianGradedStyle}, ::Type{T}, ax) where {T}
    return similar(first(concat.args), T, ax)
end

# Materialize by concatenating the block containers, placing each argument's blocks into the
# destination's diagonal hyper-block without scalar indexing.
function Base.copyto!(dest::AbstractArray, concat::Concatenated{<:AbelianGradedStyle})
    cat!(blocks(dest), blocks.(concat.args)...; dims = concat.dims)
    return dest
end

# The inner `cat!` over block containers routes through the generic `Concatenated{Nothing}`
# path, which zeros the destination container and then assigns each argument's blocks into an
# offset sub-region. Because `AbelianBlocks` is an `AbstractSparseArray`, the `zero!` and the
# offset range-assignment both run through the SparseArraysBase interface, visiting only the
# stored (symmetry-allowed) blocks; the dense generic path would touch forbidden positions and
# fail on `view` of an unstored block.

# Route `Base.cat` on abelian graded arrays through the same machinery, so plain `cat` matches
# `TensorAlgebra.cat`.
function Base._cat(dims, as::AbelianGradedArray...)
    return TensorAlgebra.concatenate(dims, as...)
end
