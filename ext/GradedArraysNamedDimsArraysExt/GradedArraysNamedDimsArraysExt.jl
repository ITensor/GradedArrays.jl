module GradedArraysNamedDimsArraysExt

using GradedArrays: GradedArrays, dual, isdual
using NamedDimsArrays: AbstractNamedUnitRange, denamed, name, named

GradedArrays.dual(r::AbstractNamedUnitRange) = named(dual(denamed(r)), dual(name(r)))
GradedArrays.isdual(r::AbstractNamedUnitRange) = isdual(denamed(r))

end
