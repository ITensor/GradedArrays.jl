module GradedArraysNamedDimsArraysExt

using GradedArrays: GradedArrays, dual, isdual
using NamedDimsArrays: AbstractNamedUnitRange, dename, name, named

function GradedArrays.dual(r::AbstractNamedUnitRange)
    return named(dual(dename(r)), dual(name(r)))
end

function GradedArrays.isdual(r::AbstractNamedUnitRange)
    return isdual(dename(r))
end

end
