using BlockArrays: AbstractBlockedUnitRange
using BlockSparseArrays:
  BlockSparseArrays,
  AbstractBlockSparseMatrix,
  AnyAbstractBlockSparseArray,
  BlockSparseArray,
  blocktype,
  eachblockstoredindex,
  sparsemortar
using LinearAlgebra: Adjoint
using TypeParameterAccessors: similartype, unwrap_array_type

const GradedArray{T,N,A<:AbstractArray{T,N},Blocks<:AbstractArray{A,N},Axes<:Tuple{AbstractGradedUnitRange{<:Integer},Vararg{AbstractGradedUnitRange{<:Integer}}}} = BlockSparseArray{
  T,N,A,Blocks,Axes
}
const GradedMatrix{T,A,Blocks,Axes} = GradedArray{
  T,2,A,Blocks,Axes
} where {Axes<:Tuple{AbstractGradedUnitRange,AbstractGradedUnitRange}}
const GradedVector{T,A,Blocks,Axes} =
  GradedArray{T,1,A,Blocks,Axes} where {Axes<:Tuple{AbstractGradedUnitRange}}

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
  blockaxistypes = map(axes) do axis
    return eltype(Base.promote_op(eachblockaxis, typeof(axis)))
  end
  similar_blocktype = Base.promote_op(
    similar, blocktype(a), Type{elt}, Tuple{blockaxistypes...}
  )
  return BlockSparseArray{elt,length(axes),similar_blocktype}(undef, axes)
end

function Base.similar(
  a::AbstractArray, elt::Type, axes::Tuple{SectorOneTo,Vararg{SectorOneTo}}
)
  return similar(a, elt, Base.OneTo.(length.(axes)))
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

ungrade(a::GradedArray) = sparsemortar(blocks(a), ungrade.(axes(a)))

function flux(a::GradedArray{<:Any,N}, I::Vararg{Block{1},N}) where {N}
  sects = ntuple(N) do d
    return flux(axes(a, d), I[d])
  end
  return ⊗(sects...)
end
function flux(a::GradedArray{<:Any,N}, I::Block{N}) where {N}
  return flux(a, Tuple(I)...)
end
function flux(a::GradedArray)
  sect = nothing
  for I in eachblockstoredindex(a)
    sect_I = flux(a, I)
    isnothing(sect) || sect_I == sect || throw(ArgumentError("Inconsistent flux."))
    sect = sect_I
  end
  return sect
end

# Copy of `Base.dims2string` defined in `show.jl`.
function dims_to_string(d)
  isempty(d) && return "0-dimensional"
  length(d) == 1 && return "$(d[1])-element"
  return join(map(string, d), '×')
end

# Copy of `BlockArrays.block2string` from `BlockArrays.jl`.
block_to_string(b, s) = string(join(map(string, b), '×'), "-blocked ", dims_to_string(s))

using TypeParameterAccessors: type_parameters, unspecify_type_parameters
function base_type_and_params(type::Type)
  alias = Base.make_typealias(type)
  base_type, params = if isnothing(alias)
    unspecify_type_parameters(type), type_parameters(type)
  else
    base_type_globalref, params_svec = alias
    base_type_globalref.name, params_svec
  end
  return base_type, params
end

function base_type_and_params(type::Type{<:GradedArray})
  return :GradedArray, type_parameters(type)
end
function base_type_and_params(type::Type{<:GradedVector})
  params = type_parameters(type)
  params′ = [params[1:1]..., params[3:end]...]
  return :GradedVector, params′
end
function base_type_and_params(type::Type{<:GradedMatrix})
  params = type_parameters(type)
  params′ = [params[1:1]..., params[3:end]...]
  return :GradedMatrix, params′
end

# Modified version of `BlockSparseArrays.concretetype_to_string_truncated`.
# This accounts for the fact that the GradedArray alias is not defined in
# BlockSparseArrays so for the sake of printing, Julia doesn't show it as
# an alias: https://github.com/JuliaLang/julia/issues/40448
function concretetype_to_string_truncated(type::Type; param_truncation_length=typemax(Int))
  isconcretetype(type) || throw(ArgumentError("Type must be concrete."))
  base_type, params = base_type_and_params(type)
  str = string(base_type)
  if isempty(params)
    return str
  end
  str *= '{'
  param_strings = map(params) do param
    param_string = string(param)
    if length(param_string) > param_truncation_length
      return "…"
    end
    return param_string
  end
  str *= join(param_strings, ", ")
  str *= '}'
  return str
end

using BlockArrays: blocksize
function Base.summary(io::IO, a::GradedArray)
  print(io, block_to_string(blocksize(a), size(a)))
  print(io, ' ')
  print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length=40))
  return nothing
end

function Base.showarg(io::IO, a::GradedArray, toplevel::Bool)
  !toplevel && print(io, "::")
  print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length=40))
  return nothing
end
