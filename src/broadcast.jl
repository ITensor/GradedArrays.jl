using Base.Broadcast: Broadcast as BC

# ========================  Sector-array broadcasting  ========================

abstract type AbstractSectorStyle{N} <: BC.AbstractArrayStyle{N} end

BC.BroadcastStyle(style::AbstractSectorStyle, ::BC.DefaultArrayStyle{0}) = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::AbstractSectorStyle) = style
BC.BroadcastStyle(style::S, ::S) where {S <: AbstractSectorStyle} = style

function Base.copyto!(dest::AbstractSectorArray, bc::BC.Broadcasted{<:AbstractSectorStyle})
    copyto!(dest, flattenlinear(bc))
    return dest
end

# Skip Base's eager axis combining. `similar` and `copyto!` work from `flattenlinear(bc)`, so the
# combined axes are never needed.
BC.instantiate(bc::BC.Broadcasted{<:AbstractSectorStyle}) = bc

# `dest .= <dense expression>` writes the reduced data, which equals the full array only under
# unique fusion.
function BC.materialize!(
        dest::AbstractSectorArray,
        bc::BC.Broadcasted{<:BC.DefaultArrayStyle}
    )
    require_unique_fusion(dest)
    BC.materialize!(data(dest), bc)
    return dest
end

Base.:+(a::AbstractSectorArray, b::AbstractSectorArray) = a .+ b
Base.:-(a::AbstractSectorArray, b::AbstractSectorArray) = a .- b
Base.:*(a::AbstractSectorArray, x::Number) = a .* x
Base.:*(x::Number, a::AbstractSectorArray) = x .* a
Base.:/(a::AbstractSectorArray, x::Number) = a ./ x

# ---- abelian sector arrays and structural deltas ----

struct UniqueSectorStyle{N} <: AbstractSectorStyle{N} end
UniqueSectorStyle{N}(::Val{M}) where {N, M} = UniqueSectorStyle{M}()

function BC.BroadcastStyle(::Type{<:AbstractSectorDelta{<:Any, <:Any, N}}) where {N}
    return UniqueSectorStyle{N}()
end
function BC.BroadcastStyle(::Type{<:UniqueSectorArray{<:Any, <:Any, N}}) where {N}
    return UniqueSectorStyle{N}()
end

# The axes of a `conj` operand are already dualized, so a half-conjugated broadcast like
# `conj.(s) .- t` is rejected by the axis check.
function Base.similar(bc::BC.Broadcasted{<:UniqueSectorStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractSectorArray, bc′.args)]
    return similar(arg, elt, axes(flattenlinear(bc)))
end

# ---- fused blocks (FusedSectorVector, FusedSectorMatrix) ----

struct SectorStyle{N} <: AbstractSectorStyle{N} end
SectorStyle{N}(::Val{M}) where {N, M} = SectorStyle{M}()

BC.BroadcastStyle(::Type{<:FusedSectorVector}) = SectorStyle{1}()
BC.BroadcastStyle(::Type{<:FusedSectorMatrix}) = SectorStyle{2}()

# Dispatches on the style rather than the axis type, which cannot tell a fused block from a
# `UniqueSectorArray`.
# TODO: these allocate default CPU `Array` block storage; the style should carry a data/storage type (as
# Base's `ArrayStyle` carries the array type) so non-`Array` (e.g. GPU) block storage is preserved.
function Base.similar(bc::BC.Broadcasted{SectorStyle{1}}, elt::Type)
    ax = axes(flattenlinear(bc))
    return FusedSectorVector{elt}(undef, sector(ax[1]), datalength(ax[1]))
end
function Base.similar(bc::BC.Broadcasted{SectorStyle{2}}, elt::Type)
    ax = axes(flattenlinear(bc))
    return FusedSectorMatrix{elt}(
        undef,
        sector(ax[1]),
        datalength(ax[1]),
        datalength(ax[2])
    )
end

# ========================  Graded-array broadcasting  ========================

abstract type AbstractGradedStyle{N} <: BC.AbstractArrayStyle{N} end

BC.BroadcastStyle(style::AbstractGradedStyle, ::BC.DefaultArrayStyle{0}) = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::AbstractGradedStyle) = style
BC.BroadcastStyle(style::S, ::S) where {S <: AbstractGradedStyle} = style

# The generic fallback recurses instead of erroring.
function BC.BroadcastStyle(::AbstractGradedStyle, ::BC.DefaultArrayStyle)
    return error("cannot broadcast a graded array together with a non-graded array")
end
function BC.BroadcastStyle(::BC.DefaultArrayStyle, ::AbstractGradedStyle)
    return error("cannot broadcast a graded array together with a non-graded array")
end

# See the `AbstractSectorStyle` override above.
BC.instantiate(bc::BC.Broadcasted{<:AbstractGradedStyle}) = bc

# The array a linear broadcast reproduces.
broadcast_array(a::TA.ScaledBroadcasted) = broadcast_array(TA.unscaled(a))
broadcast_array(a::TA.ConjBroadcasted) = broadcast_array(parent(a))
broadcast_array(a::TA.AddBroadcasted) = broadcast_array(first(TA.addends(a)))
broadcast_array(a::TA.PermutedDims) = parent(a)
broadcast_array(a::AbstractArray) = a

# ---- fused graded arrays ----
abstract type AbstractFusedGradedStyle{N} <: AbstractGradedStyle{N} end

struct FusedGradedStyle{N} <: AbstractFusedGradedStyle{N} end
FusedGradedStyle{N}(::Val{M}) where {N, M} = FusedGradedStyle{M}()

BC.BroadcastStyle(::Type{<:FusedGradedVector}) = FusedGradedStyle{1}()
BC.BroadcastStyle(::Type{<:FusedGradedMatrix}) = FusedGradedStyle{2}()

# Base's broadcast `axes` loses the graded axes.
Base.axes(bc::BC.Broadcasted{<:AbstractGradedStyle}) = axes(flattenlinear(bc))

# TODO: these allocate default CPU `Array` block storage; the style should carry a data/storage type (as
# Base's `ArrayStyle` carries the array type) so non-`Array` (e.g. GPU) block storage is preserved.
function Base.similar(bc::BC.Broadcasted{FusedGradedStyle{1}}, elt::Type)
    return FusedGradedVector{elt}(undef, only(axes(bc)))
end
function Base.similar(bc::BC.Broadcasted{FusedGradedStyle{2}}, elt::Type)
    # The constructor takes the non-dual domain.
    return FusedGradedMatrix{elt}(undef, axes(bc)[1], dual(axes(bc)[2]))
end

function Base.copyto!(
        dest::AbstractFusedGradedArray,
        bc::BC.Broadcasted{<:AbstractFusedGradedStyle}
    )
    return copyto!(dest, flattenlinear(bc))
end
