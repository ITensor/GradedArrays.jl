module GradedArraysSUNRepresentationsExt

using GradedArrays: GradedArrays, TensorKitSector
using SUNRepresentations: SUNIrrep
using TensorKitSectors: TensorKitSectors as TKS

# GradedArrays has no `SU(N)` sector of its own, so one arrives as a `TensorKitSector`. This
# constructor is the shorthand letting a caller give the `N-1` Dynkin labels.
function GradedArrays.TensorKitSector{SUNIrrep{N}}(λ::NTuple{M, Int}) where {N, M}
    M + 1 == N || throw(ArgumentError("Length of λ must be N-1 for SU(N) irreps"))
    return TensorKitSector(SUNIrrep{N}((λ..., 0)))
end
GradedArrays.label(s::TensorKitSector{<:SUNIrrep}) = Base.front(TKS.Sector(s).I)

end
