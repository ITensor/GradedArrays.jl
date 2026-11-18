module GradedArraysSUNRepresentationsExt

using GradedArrays: GradedArrays, SU
using SUNRepresentations: SUNIrrep, dynkin_label

# Both parameters, so that this is a concrete type.
GradedArrays.tensorkit_sectortype(::Type{<:SU{N}}) where {N} = SUNIrrep{N, N - 1}
GradedArrays.Sector(c::SUNIrrep{N, M}) where {N, M} = SU{N, M}(dynkin_label(c))

end
