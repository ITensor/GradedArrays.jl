# The lazy conjugate of a `GradedArray`, from `TensorAlgebra.conjed`. Conjugation dualizes every axis
# without swapping codomain and domain, so it cannot be a lazy wrapper of the fused storage, whose
# arrow directions are canonical. Everything that goes through the parent generically is defined on
# `TensorAlgebra.ConjArray`. These are the graded-specific pieces.
const ConjGradedArray{T, S, N} = TensorAlgebra.ConjArray{T, N, <:GradedArray{T, S, N}}

axes_codomain(a::ConjGradedArray) = map(conj, axes_codomain(parent(a)))
axes_domain(a::ConjGradedArray) = map(conj, axes_domain(parent(a)))

broadcast_array(a::ConjGradedArray) = broadcast_array(parent(a))

function Base.similar(
        a::ConjGradedArray, ::Type{T},
        axes::Tuple{AbstractGradedOneTo{S}, Vararg{AbstractGradedOneTo{S}}}
    ) where {T, S}
    return similar(parent(a), T, axes)
end
