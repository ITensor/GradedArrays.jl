using Base.Broadcast: Broadcast as BC

# ========================  SectorArray / SectorDelta broadcasting  ========================

struct SectorStyle{N} <: BC.AbstractArrayStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

BC.BroadcastStyle(::Type{<:SectorDelta{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:SectorArray{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(style::SectorStyle{N}, ::BC.DefaultArrayStyle{0}) where {N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::SectorStyle{N}) where {N} = style
BC.BroadcastStyle(s1::SectorStyle{N}, ::SectorStyle{N}) where {N} = s1

function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type, ax)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa SectorArray, bc′.args)]
    return similar(arg, elt, axes(arg))
end

function Base.copyto!(dest::SectorArray, bc::BC.Broadcasted{<:SectorStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("SectorArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end

# ========================  AbelianArray broadcasting  ========================

struct AbelianBroadcastStyle{N} <: BC.AbstractArrayStyle{N} end
AbelianBroadcastStyle{N}(::Val{M}) where {N, M} = AbelianBroadcastStyle{M}()

BC.BroadcastStyle(::Type{<:AbelianArray{<:Any, N}}) where {N} = AbelianBroadcastStyle{N}()
function BC.BroadcastStyle(
        style::AbelianBroadcastStyle{N},
        ::BC.DefaultArrayStyle{0}
    ) where {N}
    return style
end
function BC.BroadcastStyle(
        ::BC.DefaultArrayStyle{0},
        style::AbelianBroadcastStyle{N}
    ) where {N}
    return style
end
BC.BroadcastStyle(s1::AbelianBroadcastStyle{N}, ::AbelianBroadcastStyle{N}) where {N} = s1

function Base.similar(bc::BC.Broadcasted{<:AbelianBroadcastStyle}, elt::Type, ax)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbelianArray, bc′.args)]
    return similar(arg, elt)
end

function Base.copyto!(dest::AbelianArray, bc::BC.Broadcasted{<:AbelianBroadcastStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("AbelianArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end
