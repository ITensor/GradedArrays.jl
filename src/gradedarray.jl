using BlockArrays: AbstractBlockedUnitRange
using BlockSparseArrays:
  BlockSparseArrays,
  AbstractBlockSparseMatrix,
  AnyAbstractBlockSparseArray,
  BlockSparseArray,
  BlockSparseMatrix,
  BlockSparseVector,
  blocktype
using LinearAlgebra: Adjoint
using TypeParameterAccessors: similartype, unwrap_array_type

const GradedArray{T,M,A,Blocks,Axes} = BlockSparseArray{
  T,M,A,Blocks,Axes
} where {Axes<:Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}}}
const GradedMatrix{T,A,Blocks,Axes} = GradedArray{
  T,2,A,Blocks,Axes
} where {Axes<:Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}}}
const GradedVector{T,A,Blocks,Axes} = GradedArray{
  T,1,A,Blocks,Axes
} where {Axes<:Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}}}

# TODO: Handle this through some kind of trait dispatch, maybe
# a `SymmetryStyle`-like trait to check if the block sparse
# matrix has graded axes.
function Base.axes(a::Adjoint{<:Any,<:AbstractBlockSparseMatrix})
  return dual.(reverse(axes(a')))
end

# TODO: Need to implement this! Will require implementing
# `block_merge(a::AbstractUnitRange, blockmerger::BlockedUnitRange)`.
function BlockSparseArrays.block_merge(
  a::AbstractGradedUnitRange, blockmerger::AbstractBlockedUnitRange
)
  return a
end

# A block spare array similar to the input (dense) array.
# TODO: Make `BlockSparseArrays.blocksparse_similar` more general and use that,
# and also turn it into an DerivableInterfaces.jl-based interface function.
function similar_blocksparse(
  a::AbstractArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  # TODO: Probably need to unwrap the type of `a` in certain cases
  # to make a proper block type.
  return BlockSparseArray{
    elt,length(axes),similartype(unwrap_array_type(blocktype(a)), elt, axes)
  }(
    undef, axes
  )
end

function Base.similar(
  a::AbstractArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  return similar_blocksparse(a, elt, axes)
end

# Fix ambiguity error with `BlockArrays.jl`.
function Base.similar(
  a::StridedArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  return similar_blocksparse(a, elt, axes)
end

# Fix ambiguity error with `BlockSparseArrays.jl`.
# TBD DerivableInterfaces?
function Base.similar(
  a::AnyAbstractBlockSparseArray,
  elt::Type,
  axes::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}},
)
  return similar_blocksparse(a, elt, axes)
end

function Base.zeros(
  elt::Type, ax::Tuple{AbstractGradedUnitRange,Vararg{AbstractGradedUnitRange}}
)
  return BlockSparseArray{elt}(undef, ax)
end

function getindex_blocksparse(a::AbstractArray, I::AbstractUnitRange...)
  a′ = similar(a, only.(axes.(I))...)
  a′ .= a
  return a′
end

function Base.getindex(
  a::AbstractArray, I1::AbstractGradedUnitRange, I_rest::AbstractGradedUnitRange...
)
  return getindex_blocksparse(a, I1, I_rest...)
end

# Fix ambiguity error with Base.
function Base.getindex(a::Vector, I::AbstractGradedUnitRange)
  return getindex_blocksparse(a, I)
end

# Fix ambiguity error with BlockSparseArrays.jl.
function Base.getindex(
  a::AnyAbstractBlockSparseArray,
  I1::AbstractGradedUnitRange,
  I_rest::AbstractGradedUnitRange...,
)
  return getindex_blocksparse(a, I1, I_rest...)
end

# Fix ambiguity error with BlockSparseArrays.jl.
function Base.getindex(
  a::AnyAbstractBlockSparseArray{<:Any,2},
  I1::AbstractGradedUnitRange,
  I2::AbstractGradedUnitRange,
)
  return getindex_blocksparse(a, I1, I2)
end
