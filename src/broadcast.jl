using Base.Broadcast: Broadcast as BC

# ========================  Sector-array broadcasting  ========================
#
# Every sector array/delta broadcasts data-wise (operate on the reduced data, keep the sector), so
# the linear machinery — the combination rules and `copyto!` — is shared across all of them via
# `AbstractSectorStyle`. Only allocation (`similar`) differs, because reconstructing the result from
# the broadcast axes depends on the concrete type: an `AbelianSectorArray` has one sector per axis,
# so `Base.similar(arg, elt, axes)` is well-defined; a fused block (`SectorMatrix`, `SectorVector`)
# has a single *coupled* sector that axis type alone cannot distinguish from an unfused array, so the
# block style reconstructs from its rank instead.

abstract type AbstractSectorStyle{N} <: BC.AbstractArrayStyle{N} end

BC.BroadcastStyle(style::AbstractSectorStyle, ::BC.DefaultArrayStyle{0}) = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::AbstractSectorStyle) = style
BC.BroadcastStyle(style::S, ::S) where {S <: AbstractSectorStyle} = style

function Base.copyto!(dest::AbstractSectorArray, bc::BC.Broadcasted{<:AbstractSectorStyle})
    copyto!(dest, flattenlinear(bc))
    return dest
end

# ---- abelian sector arrays and structural deltas ----

struct AbelianSectorStyle{N} <: AbstractSectorStyle{N} end
AbelianSectorStyle{N}(::Val{M}) where {N, M} = AbelianSectorStyle{M}()

function BC.BroadcastStyle(::Type{<:AbstractSectorDelta{<:Any, <:Any, N}}) where {N}
    return AbelianSectorStyle{N}()
end
function BC.BroadcastStyle(::Type{<:AbelianSectorArray{<:Any, <:Any, N}}) where {N}
    return AbelianSectorStyle{N}()
end

# Allocate from the flattened linear expression's axes. Each `AbelianSectorArray` axis carries its
# own sector, so `similar(arg, elt, axes)` fully determines the result. A `conj` operand lowers to a
# `ConjArray` whose axes are already dualized, so the result axes — and the rejection of a
# half-conjugated broadcast like `conj.(s) .- t` — fall out of the standard machinery.
function Base.similar(bc::BC.Broadcasted{<:AbelianSectorStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractSectorArray, bc′.args)]
    return similar(arg, elt, axes(flattenlinear(bc)))
end

# ---- fused blocks (SectorVector, SectorMatrix) ----

struct SectorStyle{N} <: AbstractSectorStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

BC.BroadcastStyle(::Type{<:SectorVector}) = SectorStyle{1}()
BC.BroadcastStyle(::Type{<:SectorMatrix}) = SectorStyle{2}()

# Rebuild the block from the linear expression's `SectorOneTo` axes, which carry the coupled sector
# (dualized when the broadcast conjugated an operand). Keyed on the block style's rank rather than
# `Base.similar` on axis type, which cannot tell a fused block from an unfused `AbelianSectorArray`.
function Base.similar(bc::BC.Broadcasted{SectorStyle{1}}, elt::Type)
    ax = axes(flattenlinear(bc))
    return SectorVector{elt}(undef, sector(ax[1]), datalength(ax[1]))
end
function Base.similar(bc::BC.Broadcasted{SectorStyle{2}}, elt::Type)
    ax = axes(flattenlinear(bc))
    return SectorMatrix{elt}(undef, sector(ax[1]), datalength(ax[1]), datalength(ax[2]))
end

# ========================  Graded-array broadcasting  ========================
#
# The graded layer mirrors the sector layer: linear machinery shared via `AbstractGradedStyle`, and
# allocation split by whether the result reconstructs from axes alone. An `AbelianGradedArray` has
# one graded range per axis, so `Base.similar(arg, elt, axes)` (the same path `unmatricize` uses)
# works; a fused graded array keys its blocks by coupled sector, a layout axis type cannot tell apart
# from the unfused cartesian one, so it reconstructs from its rank instead.

abstract type AbstractGradedStyle{N} <: BC.AbstractArrayStyle{N} end

BC.BroadcastStyle(style::AbstractGradedStyle, ::BC.DefaultArrayStyle{0}) = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::AbstractGradedStyle) = style
BC.BroadcastStyle(style::S, ::S) where {S <: AbstractGradedStyle} = style

# Broadcasting a graded array together with a non-graded array (a plain dense array, `N ≥ 1`) has
# no meaning: the block/symmetry structure has no counterpart in the dense operand, and the generic
# fallback recurses instead of erroring. Reject it. Scalars (`DefaultArrayStyle{0}`) stay allowed
# via the more specific methods above.
function BC.BroadcastStyle(::AbstractGradedStyle, ::BC.DefaultArrayStyle)
    return error("cannot broadcast a graded array together with a non-graded array")
end
function BC.BroadcastStyle(::BC.DefaultArrayStyle, ::AbstractGradedStyle)
    return error("cannot broadcast a graded array together with a non-graded array")
end

function Base.copyto!(dest::AbstractGradedArray, bc::BC.Broadcasted{<:AbstractGradedStyle})
    return copyto!(dest, flattenlinear(bc))
end

# ---- unfused (cartesian-block) graded arrays ----

struct AbelianGradedStyle{N} <: AbstractGradedStyle{N} end
AbelianGradedStyle{N}(::Val{M}) where {N, M} = AbelianGradedStyle{M}()

# Default for a generic graded array; the fused subtypes override this below.
function BC.BroadcastStyle(::Type{<:AbstractGradedArray{<:Any, <:Any, N}}) where {N}
    return AbelianGradedStyle{N}()
end

# TODO: Ideally this would incorporate information from all broadcast arguments
# (or their blocktypes) when computing similar, rather than picking one argument.
# The axes come from the flattened linear expression so a `conj` operand (lowered to a
# `ConjArray` with dualized axes) gives the result dualized axes for free.
function Base.similar(bc::BC.Broadcasted{<:AbelianGradedStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractGradedArray, bc′.args)]
    return similar(arg, elt, axes(flattenlinear(bc)))
end

# ---- fused (coupled-sector-block) graded arrays ----
#
# `FusedGradedMatrix` and `FusedGradedVector` store their blocks keyed by coupled sector, a
# different layout from `AbelianGradedArray`'s cartesian blocks. Allocating the result rebuilds the
# fused block structure rather than routing through the shared axes-based `similar` that
# `unmatricize` relies on to build an *unfused* `AbelianGradedArray` destination. Only linear
# broadcasts are supported; the block arithmetic is the `bipermutedimsopadd!` overload in
# `tensoralgebra.jl`.

struct GradedStyle{N} <: AbstractGradedStyle{N} end
GradedStyle{N}(::Val{M}) where {N, M} = GradedStyle{M}()

BC.BroadcastStyle(::Type{<:FusedGradedVector}) = GradedStyle{1}()
BC.BroadcastStyle(::Type{<:FusedGradedMatrix}) = GradedStyle{2}()

# Rebuild the fused array from the linear expression's axes: the undef constructors invert `axes`,
# undoing the domain dualization a `conj` operand introduces so the result lands in the right sectors.
function Base.similar(bc::BC.Broadcasted{GradedStyle{1}}, elt::Type)
    return FusedGradedVector{elt}(undef, axes(flattenlinear(bc)))
end
function Base.similar(bc::BC.Broadcasted{GradedStyle{2}}, elt::Type)
    return FusedGradedMatrix{elt}(undef, axes(flattenlinear(bc)))
end
