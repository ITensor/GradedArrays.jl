module GradedArraysSUNRepresentationsExt

using GradedArrays: GradedArrays, SU
using SUNRepresentations: SUNIrrep, dynkin_label

# `SU` itself lives in GradedArrays, so it can be named, constructed and compared without this
# extension. Everything that needs SU(N) representation theory needs `SUNRepresentations` and so
# lives here: naming the counterpart is what the generic sector machinery reaches through for a
# sector's dimension, its ordering, its fusion style and its fusion rule.
GradedArrays.tensorkit_sectortype(::Type{SU{N}}) where {N} = SUNIrrep{N}
# Both label by Dynkin labels, so this is the same representation spelled the same way.
GradedArrays.Sector(c::SUNIrrep{N}) where {N} = SU{N}(dynkin_label(c)...)

end
