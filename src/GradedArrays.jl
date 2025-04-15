module GradedArrays

include("LabelledNumbers/LabelledNumbers.jl")
using .LabelledNumbers: LabelledNumbers
include("GradedUnitRanges/GradedUnitRanges.jl")
include("SymmetrySectors/SymmetrySectors.jl")
include("GradedUnitRanges/fusion.jl")

# This makes the following names accessible
# as `GradedArrays.x`.
using .GradedUnitRanges:
  GradedUnitRanges,
  GradedOneTo,
  GradedUnitRange,
  blocklabels,
  dag,
  dual,
  flip,
  gradedrange,
  isdual,
  map_blocklabels,
  nondual,
  sector_type,
  sectorunitrange,
  space_isequal
using .SymmetrySectors: SymmetrySectors
include("gradedarray.jl")

end
