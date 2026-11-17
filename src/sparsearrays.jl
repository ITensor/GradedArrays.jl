# Stored-entry interface functions owned by GradedArrays, so it does not depend on
# SparseArraysBase.

function storedlength end
function isstored end

function isstored(a::AbstractArray, I::Int...)
    @inline
    @boundscheck checkbounds(a, I...)
    return true
end
function getstoredindex end
function setstoredindex! end
function getunstoredindex end
function setunstoredindex! end
function eachstoredindex end
function storedvalues end

function getindex_sparse(a::AbstractArray, I::Int...)
    return isstored(a, I...) ? getstoredindex(a, I...) : getunstoredindex(a, I...)
end
function setindex!_sparse(a::AbstractArray, value, I::Int...)
    isstored(a, I...) ? setstoredindex!(a, value, I...) : setunstoredindex!(a, value, I...)
    return a
end

# The generic `TensorAlgebra.concatenate!` slices the destination, which the block containers do
# not support, so place the stored entries of each argument at their offsets instead. Concatenating
# along several dimensions is block-diagonal, so the destination is zeroed first.
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
