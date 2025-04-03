module GradedArrays

include("LabelledNumbers/LabelledNumbers.jl")
using .LabelledNumbers: LabelledNumbers
include("GradedUnitRanges/GradedUnitRanges.jl")
# This makes the following names accessible
# as `GradedArrays.x`.
using .GradedUnitRanges:
  GradedUnitRanges,
  GradedOneTo,
  GradedUnitRange,
  GradedUnitRangeDual,
  LabelledUnitRangeDual,
  blocklabels,
  blockmergesortperm,
  blocksortperm,
  dag,
  dual,
  dual_type,
  flip,
  gradedrange,
  isdual,
  nondual,
  nondual_type,
  sector_type,
  space_isequal,
  unmerged_tensor_product
include("SymmetrySectors/SymmetrySectors.jl")
using .SymmetrySectors: SymmetrySectors
include("gradedarray.jl")

end
