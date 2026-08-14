# Stored-entry interface functions owned by GradedArrays.
#
# GradedArrays implements a stored-entry interface on its block-container views. These names are
# duplicated with SparseArraysBase by design: GradedArrays owns them here so it does not depend on
# SparseArraysBase. They are internal (not exported); downstream reaches them by qualified import,
# e.g. `using GradedArrays: storedlength`. If GradedArrays depends on SparseArraysBase again later,
# these become overloads of the SparseArraysBase functions instead of owned functions.

function storedlength end
function isstored end
function getstoredindex end
function setstoredindex! end
function getunstoredindex end
function setunstoredindex! end
function eachstoredindex end
function storedvalues end

# Scalar sparse `getindex`/`setindex!`: a stored entry routes through `getstoredindex` /
# `setstoredindex!`, an unstored one through `getunstoredindex` / `setunstoredindex!` (which error for
# a symmetry-forbidden block). Block containers route their `Base.getindex` / `Base.setindex!` here,
# the plain-`AbstractArray` replacement for the `AbstractSparseArray` generic.
function getindex_sparse(a::AbstractArray, I::Int...)
    return isstored(a, I...) ? getstoredindex(a, I...) : getunstoredindex(a, I...)
end
function setindex!_sparse(a::AbstractArray, value, I::Int...)
    isstored(a, I...) ? setstoredindex!(a, value, I...) : setunstoredindex!(a, value, I...)
    return a
end

# Minimal stand-in for the sparse concatenation SparseArraysBase gets for free from the generic
# `TensorAlgebra.concatenate!`. That generic path slices the destination (`@view(a[ranges...])`,
# `a[ranges...] = x`), which our block containers do not implement; instead place each argument's whole
# stored entries at their concatenation offsets, using only the stored-entry interface. Along a
# concatenated dimension the offset accumulates across arguments; along a shared dimension it stays put.
# Multi-dimension cat is block-diagonal, so the allowed-but-unwritten off-diagonal entries are zeroed
# first. `dest`/`args` are block containers (their `size` counts blocks, their stored entries are block
# views).
function concatenate_sparse!(dest::AbstractArray, dims, args::AbstractArray...)
    catdims = TensorAlgebra.dims2cat(dims)
    iscatdim(d) = d <= length(catdims) && catdims[d]
    count(catdims) > 1 && foreach(zero!, storedvalues(dest))
    offset = ntuple(Returns(0), ndims(dest))
    for arg in args
        for (I, block) in zip(eachstoredindex(IndexCartesian(), arg), storedvalues(arg))
            destI = ntuple(
                d -> iscatdim(d) ? Tuple(I)[d] + offset[d] : Tuple(I)[d],
                ndims(dest)
            )
            setstoredindex!(dest, block, destI...)
        end
        offset =
            ntuple(d -> iscatdim(d) ? offset[d] + size(arg, d) : offset[d], ndims(dest))
    end
    return dest
end
