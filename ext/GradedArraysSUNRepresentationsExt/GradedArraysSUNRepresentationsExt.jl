module GradedArraysSUNRepresentationsExt

using GradedArrays: GradedArrays, SUN, label
using SUNRepresentations: SUNIrrep, weight
using TensorKitSectors: TensorKitSectors as TKS

# `SUN` itself lives in GradedArrays, so it can be named, constructed and compared without this
# extension. Everything that needs SU(N) representation theory needs `SUNRepresentations` and so
# lives here: these two conversions are what the generic sector machinery reaches through for a
# sector's dimension, its ordering, its fusion style and its fusion rule.
TKS.Sector(s::SUN{N}) where {N} = SUNIrrep{N}(label(s))
GradedArrays.tensorkit_sectortype(::Type{SUN{N}}) where {N} = SUNIrrep{N}
# `SUN` shifts the weight to end in zero, so this also canonicalizes.
GradedArrays.Sector(c::SUNIrrep{N}) where {N} = SUN{N}(weight(c))

end
