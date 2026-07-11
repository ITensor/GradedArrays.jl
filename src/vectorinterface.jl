# `VectorInterface` methods so graded arrays can drive iterative solvers such as
# `KrylovKit.linsolve`, which push their Krylov vectors through `VectorInterface`. The generic
# `AbstractArray` fallbacks broadcast a function over the elements, which the graded broadcast style
# rejects, so the in-place methods forward to the block-wise `TensorAlgebra` methods instead.

function VI.zerovector(a::AbstractGradedArray, ::Type{S}) where {S <: Number}
    return VI.zerovector!(similar(a, S))
end
VI.zerovector!(a::AbstractGradedArray) = zero!(a)
VI.zerovector!!(a::AbstractGradedArray) = VI.zerovector!(a)
# `VectorInterface` derives the rest: `zerovector(a)` defaults `S` to `scalartype(a)`, and
# `zerovector!!(a, S)` recycles via `zerovector!!(a)` or widens via `zerovector(a, S)`.

# Out-of-place `scale`/`add` allocate a destination of the promoted scalar type (via the public
# `Base.promote_op`, since `VectorInterface`'s `promote_scale`/`promote_add` are internal), so
# scaling a real array by a complex coefficient widens to a complex result.
function VI.scale(a::AbstractGradedArray, α::Number)
    T = Base.promote_op(VI.scale, VI.scalartype(a), typeof(α))
    return VI.scale!(similar(a, T), a, α)
end
VI.scale!(a::AbstractGradedArray, α::Number) = TensorAlgebra.scale!(a, α)
function VI.scale!(b::AbstractGradedArray, a::AbstractGradedArray, α::Number)
    return TensorAlgebra.add!(b, a, α, false)
end
# The `!!` methods fall back to out-of-place allocation when the destination can't hold the result.
function VI.scale!!(a::AbstractGradedArray, α::Number)
    T = Base.promote_op(VI.scale, VI.scalartype(a), typeof(α))
    T <: VI.scalartype(a) || return VI.scale(a, α)
    return VI.scale!(a, α)
end
function VI.scale!!(b::AbstractGradedArray, a::AbstractGradedArray, α::Number)
    T = Base.promote_op(VI.scale, VI.scalartype(a), typeof(α))
    T <: VI.scalartype(b) || return VI.scale(a, α)
    return VI.scale!(b, a, α)
end

function VI.add(a::AbstractGradedArray, b::AbstractGradedArray, α::Number, β::Number)
    T = Base.promote_op(VI.add, VI.scalartype(a), VI.scalartype(b), typeof(α), typeof(β))
    return VI.add!(VI.scale!(similar(a, T), a, β), b, α, true)
end
function VI.add!(a::AbstractGradedArray, b::AbstractGradedArray, α::Number, β::Number)
    return TensorAlgebra.add!(a, b, α, β)
end
function VI.add!!(a::AbstractGradedArray, b::AbstractGradedArray, α::Number, β::Number)
    T = Base.promote_op(VI.add, VI.scalartype(a), VI.scalartype(b), typeof(α), typeof(β))
    T <: VI.scalartype(a) || return VI.add(a, b, α, β)
    return VI.add!(a, b, α, β)
end
# `VectorInterface` derives the two- and three-argument `add`/`add!`/`add!!` from these, defaulting
# the omitted coefficients to `One()`.

VI.inner(a::AbstractGradedArray, b::AbstractGradedArray) = dot(a, b)
