# Abelian graded concatenation, plugged into the `TensorAlgebra` concatenation hooks. The result is
# concat-order: sectors of the arguments are kept in order and block-appended, never merged or
# sorted. This is only correct for `AbelianGradedArray`. Fused graded arrays (`GradedStyle`)
# direct-sum onto a merged, rotated basis and need a different implementation, so they are
# deliberately not handled here.

using BlockArrays: blocks
using TensorAlgebra: cat!

# `dual` on a `GradedOneTo` is again a `GradedOneTo`, so this binary method covers dual axes too,
# and `TensorAlgebra` folds `cat_axis` pairwise so binary suffices for any number of arguments.
function TensorAlgebra.cat_axis(a1::GradedOneTo, a2::GradedOneTo)
    return mortar_axis([a1, a2])
end

# Override the broadcast-based default: the arguments have different sizes, so allocate from the
# concatenated graded axes directly.
function TensorAlgebra.cat_similar(::AbelianGradedStyle, ::Type{T}, ax, args...) where {T}
    return similar(first(args), T, ax)
end

# Place whole blocks (no scalar indexing) with the inner `cat!` on the block containers. That works
# because `AbelianBlocks` is an `AbstractSparseArray`, so the placement visits only the stored
# (symmetry-allowed) blocks, whereas a dense path would touch forbidden positions.
function TensorAlgebra.cat_copyto!(dest, ::AbelianGradedStyle, dims, args...)
    cat!(blocks(dest), blocks.(args)...; dims = dims)
    return dest
end

# Route `Base.cat` through the same machinery, so plain `cat` matches `TensorAlgebra.cat`.
function Base._cat(dims, as::AbelianGradedArray...)
    return TensorAlgebra.concatenate(dims, as...)
end
