module LabelledNumbersBlockArraysExt

using BlockArrays: BlockArrays, Block, BlockBoundsError
using LabelledNumbers: LabelledUnitRange, unlabel

# Fixes ambiguity error with:
# ```julia
# getindex(::LabelledUnitRange, ::Any...)
# getindex(::AbstractArray{<:Any,N}, ::Block{N}) where {N}
# getindex(::AbstractArray, ::Block{1}, ::Any...)
# ```
function Base.getindex(a::LabelledUnitRange, index::Block{1})
  @boundscheck index == Block(1) || throw(BlockBoundsError(a, index))
  return a
end

BlockArrays.blockaxes(a::LabelledUnitRange) = BlockArrays.blockaxes(unlabel(a))
BlockArrays.blockfirsts(a::LabelledUnitRange) = BlockArrays.blockfirsts(unlabel(a))
BlockArrays.blocklasts(a::LabelledUnitRange) = BlockArrays.blocklasts(unlabel(a))

end
