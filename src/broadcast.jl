using Base.Broadcast: Broadcast as BC

# ========================  Sector-array broadcasting  ========================
#
# Every sector array/delta broadcasts data-wise (operate on the reduced data, keep the sector), so
# the linear machinery — the combination rules and `copyto!` — is shared across all of them via
# `AbstractSectorStyle`. Only allocation (`similar`) differs, because reconstructing the result from
# the broadcast axes depends on the concrete type: an `UniqueSectorArray` has one sector per axis,
# so `Base.similar(arg, elt, axes)` is well-defined; a fused block (`FusedSectorMatrix`, `FusedSectorVector`)
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

# Skip Base's eager axis-combining. A sector array stores only the reduced data, not a dense array
# over its full (structural sector factor times degeneracy) axes, so `similar` and `copyto!` rebuild
# the result from `flattenlinear(bc)` and the combined axes are never needed.
BC.instantiate(bc::BC.Broadcasted{<:AbstractSectorStyle}) = bc

# `dest .= <dense expression>`: materialize a dense RHS into the reduced data (well-defined only
# under unique/abelian fusion, where the reduced data is the full array) rather than letting Base
# combine it against the destination's full axes. Sector-style RHS expressions are not dense-styled
# and keep flowing through the linear-broadcast `copyto!` above.
function BC.materialize!(
        dest::AbstractSectorArray,
        bc::BC.Broadcasted{<:BC.DefaultArrayStyle}
    )
    require_unique_fusion(dest)
    BC.materialize!(data(dest), bc)
    return dest
end

# Route array arithmetic through the broadcast fold rather than Base's array arithmetic. These are
# graded linear combinations, so broadcasting (which rebuilds from the reduced data) rather than
# element-wise arraymath over the full axes is the right primitive.
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

# Allocate from the flattened linear expression's axes. Each `UniqueSectorArray` axis carries its
# own sector, so `similar(arg, elt, axes)` fully determines the result. A `conj` operand lowers to a
# `ConjArray` whose axes are already dualized, so the result axes — and the rejection of a
# half-conjugated broadcast like `conj.(s) .- t` — fall out of the standard machinery.
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

# Rebuild the block from the linear expression's `SectorOneTo` axes, which carry the coupled sector
# (dualized when the broadcast conjugated an operand). Keyed on the block style's rank rather than
# `Base.similar` on axis type, which cannot tell a fused block from an unfused `UniqueSectorArray`.
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
#
# The graded layer mirrors the sector layer: linear machinery shared via `AbstractGradedStyle`, and
# each concrete graded array reconstructs the broadcast result from its own style (`GradedArray` and
# the fused `FusedGraded*` matrices/vectors each define their own `similar`).

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

# See the `AbstractSectorStyle` override above.
BC.instantiate(bc::BC.Broadcasted{<:AbstractGradedStyle}) = bc

# The array a linear broadcast reproduces (its output sector type, `similar_map` dispatch), through the
# named-tensor `PermutedDims` alignment wrapper.
broadcast_array(a::TA.ScaledBroadcasted) = broadcast_array(TA.unscaled(a))
broadcast_array(a::TA.ConjBroadcasted) = broadcast_array(parent(a))
broadcast_array(a::TA.AddBroadcasted) = broadcast_array(first(TA.addends(a)))
broadcast_array(a::TA.PermutedDims) = parent(a)
broadcast_array(a::AbstractArray) = a

# ---- fused (coupled-sector-block) graded arrays ----
#
# The fused family: `FusedGradedStyle` (`FusedGradedMatrix`/`FusedGradedVector`) and
# `FusedGradedDiagonalStyle` (`FusedGradedDiagonal`, defined in `fusedgradeddiagonal.jl`). Grouping
# them under `AbstractFusedGradedStyle` keys the mixing rules off the two-family split: mixing a
# `GradedArray` (`GradedStyle`) with any fused array errors (in `gradedarray.jl`), while within the
# fused family a diagonal promotes to the dense fused matrix (in `fusedgradeddiagonal.jl`).
abstract type AbstractFusedGradedStyle{N} <: AbstractGradedStyle{N} end

# `FusedGradedMatrix` and `FusedGradedVector` store their blocks keyed by coupled sector. Allocating
# the result rebuilds the fused block structure from the linear expression's axes. Only linear
# broadcasts are supported; the block arithmetic is the `bipermutedimsopadd!` overload in
# `tensoralgebra.jl`.

struct FusedGradedStyle{N} <: AbstractFusedGradedStyle{N} end
FusedGradedStyle{N}(::Val{M}) where {N, M} = FusedGradedStyle{M}()

BC.BroadcastStyle(::Type{<:FusedGradedVector}) = FusedGradedStyle{1}()
BC.BroadcastStyle(::Type{<:FusedGradedMatrix}) = FusedGradedStyle{2}()

# Base's broadcast `axes` (via `BlockArrays.combine_axes`) loses the graded axes; the flattened linear
# expression keeps them.
Base.axes(bc::BC.Broadcasted{<:AbstractGradedStyle}) = axes(flattenlinear(bc))

function Base.similar(bc::BC.Broadcasted{FusedGradedStyle{1}}, elt::Type)
    return FusedGradedVector{elt}(undef, axes(bc))
end
function Base.similar(bc::BC.Broadcasted{FusedGradedStyle{2}}, elt::Type)
    return FusedGradedMatrix{elt}(undef, axes(bc))
end

# Fused-array linear broadcasts fold to the leaf via `flattenlinear` and apply block-wise. Covers the
# whole fused family (dense matrix/vector and diagonal) via `AbstractFusedGradedStyle`.
function Base.copyto!(
        dest::AbstractFusedGradedArray,
        bc::BC.Broadcasted{<:AbstractFusedGradedStyle}
    )
    return copyto!(dest, flattenlinear(bc))
end
