# Multi-dimensional Kronecker product, used to materialize a dense graded block from
# its sector-delta factor and data array (see `_to_blocksparsearray`). `kron` covers
# the matrix and vector cases; the general N-dimensional case interleaves the factor
# and data axes, multiplies, and permutes back.

function flatten(t::Tuple{Tuple, Tuple, Vararg{Tuple}})
    return (t[1]..., flatten(Base.tail(t))...)
end
flatten(t::Tuple{Tuple}) = t[1]
flatten(::Tuple{}) = ()

function interleave(x::Tuple, y::Tuple)
    length(x) == length(y) || throw(ArgumentError("Tuples must have the same length."))
    xy = ntuple(i -> (x[i], y[i]), length(x))
    return flatten(xy)
end

function kron_nd(a::AbstractArray{<:Any, N}, b::AbstractArray{<:Any, N}) where {N}
    a′ = reshape(a, interleave(size(a), ntuple(one, N)))
    b′ = reshape(b, interleave(ntuple(one, N), size(b)))
    c′ = permutedims(a′ .* b′, reverse(ntuple(identity, 2N)))
    sz = reverse(ntuple(i -> size(a, i) * size(b, i), N))
    return permutedims(reshape(c′, sz), reverse(ntuple(identity, N)))
end
kron_nd(a1::AbstractMatrix, a2::AbstractMatrix) = kron(a1, a2)
kron_nd(a1::AbstractVector, a2::AbstractVector) = kron(a1, a2)
# Rank-0 (scalar) block, e.g. from materializing a full contraction to a scalar: the general
# `permutedims`-based path has no zero-length permutation to reverse, so multiply directly.
kron_nd(a1::AbstractArray{<:Any, 0}, a2::AbstractArray{<:Any, 0}) = fill(a1[] * a2[])
