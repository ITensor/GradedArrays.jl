module GradedArraysSUNRepresentationsExt

using GradedArrays: GradedArrays, SU
using SUNRepresentations: SUNIrrep, dynkin_label

# `SU` itself lives in GradedArrays, so it can be named, constructed and compared without this
# extension. Everything that needs SU(N) representation theory needs `SUNRepresentations` and so
# lives here: naming the counterpart is what the generic sector machinery reaches through for a
# sector's dimension, its ordering, its fusion style and its fusion rule.
# Spelled with both parameters so that this is a concrete type, as it is for every other sector.
GradedArrays.tensorkit_sectortype(::Type{<:SU{N}}) where {N} = SUNIrrep{N, N - 1}
# Both store the same Dynkin labels in the same layout, so this hands the tuple straight over.
GradedArrays.Sector(c::SUNIrrep{N, M}) where {N, M} = SU{N, M}(dynkin_label(c))

end
