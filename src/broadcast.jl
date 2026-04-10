using Base.Broadcast: Broadcast as BC

# ========================  SectorStyle broadcasting  ========================
#
# Decomposes broadcasts through the Kronecker structure (sector ⊗ data):
# - broadcasted_data(bc) strips sector wrappers, rebuilds a plain broadcast on raw data
# - broadcasted_sector(bc) extracts and validates the common sector factor
# - similar recombines via sector ⊗ similar(broadcasted_data, elt)

struct SectorStyle{N} <: BC.AbstractArrayStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

BC.BroadcastStyle(::Type{<:AbelianSectorDelta{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:AbelianSectorArray{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:SectorMatrix{<:Any}}) = SectorStyle{2}()
BC.BroadcastStyle(::Type{<:SectorIdentity{<:Any}}) = SectorStyle{2}()
BC.BroadcastStyle(style::SectorStyle{N}, ::BC.DefaultArrayStyle{0}) where {N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::SectorStyle{N}) where {N} = style
BC.BroadcastStyle(s1::SectorStyle{N}, ::SectorStyle{N}) where {N} = s1

# ---- Kronecker decomposition of broadcasts ----

# Strip sector wrappers and rebuild a plain broadcast on raw data arrays.
function broadcasted_data(bc::BC.Broadcasted{<:SectorStyle})
    return BC.broadcasted(bc.f, broadcasted_data.(bc.args)...)
end
broadcasted_data(a::AbstractSectorArray) = data(a)
broadcasted_data(a::AbstractSectorDelta) = a
broadcasted_data(x) = x

# Extract and validate the common sector factor across all sector array arguments.
function broadcasted_sector(bc::BC.Broadcasted{<:SectorStyle})
    bc′ = BC.flatten(bc)
    sects = [sector(arg) for arg in bc′.args if arg isa AbstractSectorArray]
    isempty(sects) && throw(ArgumentError("No AbstractSectorArray found in broadcast"))
    s = first(sects)
    all(==(s), sects) || throw(DimensionMismatch("sector mismatch in broadcast"))
    return s
end

function _find_data(bc::BC.Broadcasted{<:SectorStyle})
    bc′ = BC.flatten(bc)
    idx = findfirst(arg -> arg isa AbstractSectorArray, bc′.args)
    return data(bc′.args[idx])
end

function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type)
    s = broadcasted_sector(bc)
    return s ⊗ similar(_find_data(bc), elt)
end

function Base.copyto!(dest::AbstractSectorArray, bc::BC.Broadcasted{<:SectorStyle})
    isnothing(tryflattenlinear(bc)) &&
        throw(ArgumentError("SectorArray broadcasting requires linear operations"))
    data_bc = broadcasted_data(bc)
    copyto!(data(dest), data_bc)
    return dest
end

# ========================  AbelianGradedArray broadcasting  ========================

struct AbelianGradedStyle{N} <: BC.AbstractArrayStyle{N} end
AbelianGradedStyle{N}(::Val{M}) where {N, M} = AbelianGradedStyle{M}()

function BC.BroadcastStyle(::Type{<:AbelianGradedArray{<:Any, N}}) where {N}
    return AbelianGradedStyle{N}()
end
function BC.BroadcastStyle(
        style::AbelianGradedStyle{N},
        ::BC.DefaultArrayStyle{0}
    ) where {N}
    return style
end
function BC.BroadcastStyle(
        ::BC.DefaultArrayStyle{0},
        style::AbelianGradedStyle{N}
    ) where {N}
    return style
end
BC.BroadcastStyle(s1::AbelianGradedStyle{N}, ::AbelianGradedStyle{N}) where {N} = s1

# TODO: Ideally this would incorporate information from all broadcast arguments
# (or their blocktypes) when computing similar, rather than picking one argument.
function Base.similar(bc::BC.Broadcasted{<:AbelianGradedStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbelianGradedArray, bc′.args)]
    return similar(arg, elt)
end

function Base.copyto!(dest::AbelianGradedArray, bc::BC.Broadcasted{<:AbelianGradedStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("AbelianGradedArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end
