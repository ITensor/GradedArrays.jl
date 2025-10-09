module GradedArraysSUNRepresentationsExt

using SUNRepresentations: SUNIrrep
using GradedArrays: GradedArrays, SectorRange

function GradedArrays.SectorRange{SUNIrrep{N}}(λ::NTuple{M, Int}) where {N, M}
    M + 1 == N || throw(ArgumentError("Length of λ must be N-1 for SU(N) irreps"))
    return SectorRange(SUNIrrep((λ..., 0)))
end
GradedArrays.sector_label(c::SUNIrrep) = Base.front(c.I)

end
