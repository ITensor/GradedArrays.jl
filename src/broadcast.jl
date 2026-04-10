using Base.Broadcast: Broadcast as BC

# ========================  AbelianSectorArray / AbelianSectorDelta broadcasting  ========================

struct SectorStyle{N} <: BC.AbstractArrayStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

BC.BroadcastStyle(::Type{<:AbelianSectorDelta{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:AbelianSectorArray{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:SectorMatrix{<:Any}}) = SectorStyle{2}()
BC.BroadcastStyle(::Type{<:SectorIdentity{<:Any}}) = SectorStyle{2}()
BC.BroadcastStyle(style::SectorStyle{N}, ::BC.DefaultArrayStyle{0}) where {N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::SectorStyle{N}) where {N} = style
BC.BroadcastStyle(s1::SectorStyle{N}, ::SectorStyle{N}) where {N} = s1

function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type, ax)
    bc′ = BC.flatten(bc)
    idx = findfirst(arg -> arg isa AbstractSectorArray, bc′.args)
    arg = bc′.args[idx]
    return similar(arg, elt, axes(arg))
end

function Base.copyto!(dest::AbelianSectorArray, bc::BC.Broadcasted{<:SectorStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("AbelianSectorArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
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

function Base.similar(bc::BC.Broadcasted{<:AbelianGradedStyle}, elt::Type, ax)
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
