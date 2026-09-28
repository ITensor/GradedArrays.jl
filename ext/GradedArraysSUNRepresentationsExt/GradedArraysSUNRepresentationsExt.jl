module GradedArraysSUNRepresentationsExt

using GradedArrays: GradedArrays, TensorKitSector, tensorkitsector
using SUNRepresentations: SUNIrrep

# GradedArrays has no `SU(N)` sector of its own, so one arrives as a `TensorKitSector`. This
# constructor is the shorthand letting a caller give the `N-1` Dynkin labels.
function GradedArrays.TensorKitSector{SUNIrrep{N}}(λ::NTuple{M, Int}) where {N, M}
    M + 1 == N || throw(ArgumentError("Length of λ must be N-1 for SU(N) irreps"))
    return TensorKitSector(SUNIrrep{N}((λ..., 0)))
end
GradedArrays.label(s::TensorKitSector{<:SUNIrrep}) = Base.front(tensorkitsector(s).I)

end
