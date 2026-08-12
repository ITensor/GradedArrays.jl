# =============================================================================
#  Scalar- and block-indexing guards.
#
#  Scalar indexing (`a[i, j, ...]`) and block indexing (`view(a, ::Block)` and its
#  derived get/set surface) of graded/fused arrays are disabled by default. Both are
#  well-defined only in restricted circumstances (scalar indexing needs unique fusion;
#  block access needs the caller to be block-structure-aware), and the generic
#  `AbstractArray` fallbacks reach for them implicitly. Turning them off by default makes
#  any such implicit reliance an error at the point it happens rather than a silent wrong
#  result or a slow elementwise loop.
#
#  Opt back in for a specific call with the do-block forms `with_scalar_indexing` /
#  `with_block_indexing`. The two guards are independent, so one can be allowed without the
#  other. The toggles are `ScopedValue`s, so they apply only for the dynamic extent of the
#  wrapped call and compose across tasks without global mutable state.
# =============================================================================

using ScopedValues: ScopedValue, with

const _scalar_indexing = ScopedValue(false)
const _block_indexing = ScopedValue(false)

scalar_indexing_allowed() = _scalar_indexing[]
block_indexing_allowed() = _block_indexing[]

"""
    with_scalar_indexing(f, allow = true)

Run `f()` with scalar indexing of graded arrays enabled (or, with `allow = false`,
disabled) for its dynamic extent, e.g. `with_scalar_indexing() do ... end` or
`with_scalar_indexing(false) do ... end`.

!!! warning

    Scalar indexing is a convenience for interactive use and correctness checks, not an
    efficient access pattern. It should be avoided in performance-critical code, which should
    operate on whole blocks or the reduced data instead.
"""
with_scalar_indexing(f, allow::Bool = true) = with(f, _scalar_indexing => allow)

"""
    with_block_indexing(f, allow = true)

Run `f()` with block indexing of graded arrays enabled (or, with `allow = false`,
disabled) for its dynamic extent, e.g. `with_block_indexing() do ... end` or
`with_block_indexing(false) do ... end`.

!!! warning

    Block indexing is experimental and its output type may change.

!!! warning

    Like scalar indexing, this is a convenience, not an efficient access pattern, and should
    be avoided in performance-critical code.
"""
with_block_indexing(f, allow::Bool = true) = with(f, _block_indexing => allow)

function assert_scalar_indexing()
    scalar_indexing_allowed() || _throw_scalar_indexing_disabled()
    return nothing
end
@noinline function _throw_scalar_indexing_disabled()
    return error(
        "scalar indexing of a graded array is disabled by default: it is generally not " *
            "efficient (element-wise access defeats the block structure) and should be avoided " *
            "in favor of whole-block or reduced-data operations. To opt in for a specific call, " *
            "wrap it in `with_scalar_indexing() do ... end`."
    )
end

function assert_block_indexing()
    block_indexing_allowed() || _throw_block_indexing_disabled()
    return nothing
end
@noinline function _throw_block_indexing_disabled()
    return error(
        "block indexing of a graded array is disabled by default. Support for it is " *
            "experimental and subject to change (the output format and type may change in a " *
            "future release). To opt in for a specific call, wrap it in " *
            "`with_block_indexing() do ... end`."
    )
end
