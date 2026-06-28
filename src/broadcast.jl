using Base.Broadcast: Broadcast as BC

# Allocate the broadcast destination from a representative operand `arg`. The result axes
# come from the flattened linear expression, where a `conj` operand is a `ConjArray` with
# dualized axes. Only `conj` changes the axes; when they are unchanged, use the plain
# `similar(arg, elt)` (every sector/graded type supports it), reserving the dualized 3-arg
# form for the `conj` case, which only the data-carrying array types need to support.
function similar_broadcast(bc::BC.Broadcasted, arg, elt::Type)
    lb = tryflattenlinear(bc)
    ax = isnothing(lb) ? axes(arg) : axes(lb)
    return ax == axes(arg) ? similar(arg, elt) : similar(arg, elt, ax)
end

# ========================  SectorStyle broadcasting  ========================

struct SectorStyle{N} <: BC.AbstractArrayStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

BC.BroadcastStyle(::Type{<:AbelianSectorDelta{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:AbelianSectorArray{<:Any, N}}) where {N} = SectorStyle{N}()
BC.BroadcastStyle(::Type{<:SectorMatrix{<:Any}}) = SectorStyle{2}()
BC.BroadcastStyle(::Type{<:SectorIdentity{<:Any}}) = SectorStyle{2}()
BC.BroadcastStyle(style::SectorStyle{N}, ::BC.DefaultArrayStyle{0}) where {N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::SectorStyle{N}) where {N} = style
BC.BroadcastStyle(s1::SectorStyle{N}, ::SectorStyle{N}) where {N} = s1

# Allocate from the flattened linear expression's axes: a `conj` operand lowers to a
# `ConjArray` whose axes are already dualized, so the result axes (and the rejection of a
# half-conjugated broadcast like `conj.(s) .- t`) fall out of the standard machinery rather
# than needing the broadcast function threaded through by hand.
function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractSectorArray, bc′.args)]
    return similar_broadcast(bc, arg, elt)
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

function BC.BroadcastStyle(::Type{<:AbstractGradedArray{<:Any, N}}) where {N}
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
    return similar_broadcast(bc, arg, elt)
end

function Base.copyto!(dest::AbstractGradedArray, bc::BC.Broadcasted{<:GradedStyle})
    lb = tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("AbstractGradedArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end

# ====================  FusedGradedMatrix broadcasting  ====================
#
# `FusedGradedMatrix` stores blocks keyed by coupled sector rather than by
# cartesian block index, so the `GradedStyle` path (which lowers to
# `bipermutedimsopadd!` over cartesian block storage) silently produces wrong
# results. Until a proper sector-keyed broadcast path lands, FGM broadcasting
# is an explicit error; use `Base.:(+)` / `Base.:(-)` or block-wise operations
# instead.

struct FusedGradedStyle <: BC.AbstractArrayStyle{2} end
FusedGradedStyle(::Val{N}) where {N} = FusedGradedStyle()

BC.BroadcastStyle(::Type{<:FusedGradedMatrix}) = FusedGradedStyle()
BC.BroadcastStyle(s::FusedGradedStyle, ::BC.DefaultArrayStyle{0}) = s
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, s::FusedGradedStyle) = s
BC.BroadcastStyle(s::FusedGradedStyle, ::FusedGradedStyle) = s

function Base.copy(::BC.Broadcasted{FusedGradedStyle})
    return throw(
        ArgumentError(
            "Broadcasting on `FusedGradedMatrix` is not supported; use `+`/`-` " *
                "or operate block-wise via `blocks(A)` instead."
        )
    )
end
function Base.copyto!(::AbstractArray, ::BC.Broadcasted{FusedGradedStyle})
    return throw(
        ArgumentError(
            "Broadcasting on `FusedGradedMatrix` is not supported; use `+`/`-` " *
                "or operate block-wise via `blocks(A)` instead."
        )
    )
end
