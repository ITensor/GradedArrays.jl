# The lazy conjugate of a `GradedArray`, from `TensorAlgebra.conjed`. Conjugation dualizes every axis
# without swapping codomain and domain, so it cannot be a lazy wrapper of the fused storage, whose
# arrow directions are canonical. Everything that goes through the parent generically is defined on
# `TensorAlgebra.ConjArray`. These are the graded-specific pieces, including the elementwise methods
# (`fill!`, `==`, `dot`, `mapreduce`) that `TensorAlgebra.ConjArray` only defines for a dense parent.
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

# The conjugate of a fermionic graded array carries per-block signs, so a nonzero fill writes the
# conjugate of a filled array, which `copyto!` applies with the signs, instead of filling the parent
# with `conj(value)`.
# TODO: Remove the temporary. The sign of each block is that of reversing its legs,
# `(-1)^(n_odd * (n_odd - 1) / 2)` for `n_odd` odd legs, so the parent could be filled with
# `conj(value)` and then scaled in place, for example with a cached TensorKit tree transformer as
# `TensorKit.twist!` uses.
function Base.fill!(a::ConjGradedArray, value)
    if TKS.BraidingStyle(sectortype(parent(a))) isa TKS.Fermionic && !iszero(value)
        return copyto!(a, fill!(similar(a), value))
    end
    fill!(parent(a), conj(value))
    return a
end

# The conjugate of a graded array is not elementwise (it dualizes the axes and, for fermions,
# carries per-block signs), so compare against the eager conjugate.
# TODO: Remove the temporary by rematricizing `b` to the parent's split, as `==` does, and comparing
# subblocks directly with the per-block conj sign.
Base.:(==)(a::ConjGradedArray, b::GradedArray) = conj(parent(a)) == b
Base.:(==)(a::GradedArray, b::ConjGradedArray) = b == a

# The parent's elements without the per-block signs are not the conjugate's, so pair against the
# eager conjugate.
# TODO: Remove the temporary, as for `==`.
function LinearAlgebra.dot(a::ConjGradedArray, b::GradedArray)
    return LinearAlgebra.dot(conj(parent(a)), b)
end
function LinearAlgebra.dot(a::GradedArray, b::ConjGradedArray)
    return LinearAlgebra.dot(a, conj(parent(b)))
end

# Reductions over a `GradedArray` error, so give the same error for its lazy conjugate.
function Base.mapreduce(f, op, a::ConjGradedArray; kwargs...)
    return mapreduce(f ∘ conj, op, parent(a); kwargs...)
end
