using Base.Broadcast: Broadcast as BC

# ========================  SectorStyle broadcasting  ========================

struct SectorStyle{N} <: BC.AbstractArrayStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

function BC.BroadcastStyle(::Type{<:AbelianSectorDelta{<:Any, <:Any, N}}) where {N}
    return SectorStyle{N}()
end
function BC.BroadcastStyle(::Type{<:AbelianSectorArray{<:Any, <:Any, N}}) where {N}
    return SectorStyle{N}()
end
BC.BroadcastStyle(style::SectorStyle{N}, ::BC.DefaultArrayStyle{0}) where {N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::SectorStyle{N}) where {N} = style
BC.BroadcastStyle(s1::SectorStyle{N}, ::SectorStyle{N}) where {N} = s1

# `SectorMatrix` and `SectorIdentity` are fused single-sector blocks whose row (codomain) and
# column (domain) axes are structurally constrained, so the broadcasting implementation is
# more subtle and not supported yet.
function BC.BroadcastStyle(::Type{<:SectorMatrix})
    return throw(ArgumentError("broadcasting on `SectorMatrix` is not supported yet"))
end
function BC.BroadcastStyle(::Type{<:SectorIdentity})
    return throw(ArgumentError("broadcasting on `SectorIdentity` is not supported yet"))
end

# Only linear broadcasts are supported, so allocate from the flattened linear expression's
# axes: a `conj` operand lowers to a `ConjArray` whose axes are already dualized, so the
# result axes (and the rejection of a half-conjugated broadcast like `conj.(s) .- t`) fall
# out of the standard machinery rather than needing the broadcast function threaded through
# by hand.
function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractSectorArray, bc′.args)]
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("SectorArray broadcasting requires linear operations"))
    return similar(arg, elt, axes(lb))
end

function Base.copyto!(dest::AbstractSectorArray, bc::BC.Broadcasted{<:SectorStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("SectorArray broadcasting requires linear operations"))
    copyto!(dest, lb)
    return dest
end

# ========================  GradedArray broadcasting  ========================

struct GradedStyle{N} <: BC.AbstractArrayStyle{N} end
GradedStyle{N}(::Val{M}) where {N, M} = GradedStyle{M}()

function BC.BroadcastStyle(::Type{<:AbstractGradedArray{<:Any, <:Any, N}}) where {N}
    return GradedStyle{N}()
end
function BC.BroadcastStyle(
        style::GradedStyle{N},
        ::BC.DefaultArrayStyle{0}
    ) where {N}
    return style
end
function BC.BroadcastStyle(
        ::BC.DefaultArrayStyle{0},
        style::GradedStyle{N}
    ) where {N}
    return style
end
BC.BroadcastStyle(s1::GradedStyle{N}, ::GradedStyle{N}) where {N} = s1

# TODO: Ideally this would incorporate information from all broadcast arguments
# (or their blocktypes) when computing similar, rather than picking one argument.
# The axes come from the flattened linear expression so a `conj` operand (lowered to a
# `ConjArray` with dualized axes) gives the result dualized axes for free.
function Base.similar(bc::BC.Broadcasted{<:GradedStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractGradedArray, bc′.args)]
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("AbstractGradedArray broadcasting requires linear operations"))
    return similar(arg, elt, axes(lb))
end

function Base.copyto!(dest::AbstractGradedArray, bc::BC.Broadcasted{<:GradedStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("AbstractGradedArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end

# ========================  FusedGradedArray broadcasting  ========================
#
# `FusedGradedMatrix` and `FusedGradedVector` store their blocks keyed by coupled sector, a
# different layout from `AbelianGradedArray`'s cartesian blocks. They broadcast through a
# dedicated `FusedGradedStyle` so allocating the result rebuilds the fused block structure,
# rather than routing through the shared axes-based `similar` that `unmatricize` relies on
# to build an *unfused* `AbelianGradedArray` destination. Only linear broadcasts are
# supported; the block arithmetic is the `bipermutedimsopadd!` overload in `tensoralgebra.jl`.

struct FusedGradedStyle{N} <: BC.AbstractArrayStyle{N} end
FusedGradedStyle{N}(::Val{M}) where {N, M} = FusedGradedStyle{M}()

BC.BroadcastStyle(::Type{<:FusedGradedVector}) = FusedGradedStyle{1}()
BC.BroadcastStyle(::Type{<:FusedGradedMatrix}) = FusedGradedStyle{2}()
BC.BroadcastStyle(style::FusedGradedStyle{N}, ::BC.DefaultArrayStyle{0}) where {N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::FusedGradedStyle{N}) where {N} = style
BC.BroadcastStyle(s1::FusedGradedStyle{N}, ::FusedGradedStyle{N}) where {N} = s1

# Rebuild an undef fused array from the linear expression's axes (a `conj` operand dualizes those
# axes, so the result lands in the dual sectors). Isolated from `Base.similar` on purpose: the
# fused-vs-unfused destination cannot be told apart by axis type alone, so the broadcast path
# allocates here rather than through the shared axes-based `similar`.
function _similar_fused(::Type{T}, ax::Tuple{<:GradedOneTo}) where {T}
    return FusedGradedVector{T}(undef, _fusedcodomain(only(ax)))
end
function _similar_fused(::Type{T}, ax::Tuple{<:GradedOneTo, <:GradedOneTo}) where {T}
    return FusedGradedMatrix{T}(undef, _fusedcodomain(ax[1]), _fuseddomain(ax[2]))
end

function Base.similar(bc::BC.Broadcasted{<:FusedGradedStyle}, elt::Type)
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("FusedGradedArray broadcasting requires linear operations"))
    return _similar_fused(elt, axes(lb))
end

function Base.copyto!(dest::FusedGradedVecOrMat, bc::BC.Broadcasted{<:FusedGradedStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("FusedGradedArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end
