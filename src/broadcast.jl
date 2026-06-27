using Base.Broadcast: Broadcast as BC

# ========================  SectorStyle broadcasting  ========================
#
# Decomposes broadcasts through the Kronecker structure (sector ⊗ data):
# - broadcasted_data(bc) strips sector wrappers, rebuilds a plain broadcast on raw data
# - broadcasted_sector(bc) extracts and validates the common sector factor
# - similar recombines via sector_kron(broadcasted_sector, similar(broadcasted_data, elt))

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
broadcasted_data(x) = x

# Map a broadcast function to its action on the sector: `conj` dualizes the sector axes,
# every other function leaves them intact. Only `conj` changes the selection rule, so the
# arithmetic nodes (`+`, `-`, `*`, `/`) and `identity` all map to `identity` here.
sector_op(::typeof(conj)) = sector_conj
sector_op(::Any) = identity

# Conjugate the sector (structural) part of a factor by dualizing its per-axis sectors. This
# is the sign-free part of `conj`: a structural factor's stored value is always `one(T)`, so
# it cannot represent the fermionic reversal sign that a full `conj` carries. That sign (and
# the data conjugation) ride the data side via `op = conj` through `bipermutedimsopadd!`, so
# here we only flip the axis dualities.
function sector_conj(s::AbelianSectorDelta{T}) where {T}
    return AbelianSectorDelta{T}(map(conj, s.sectors))
end
sector_conj(s::SectorIdentity{T}) where {T} = SectorIdentity{T}(conj(s.sector))

# Extract and validate the common sector factor, threading the one axis-changing op
# through the broadcast tree. Walking the un-flattened tree is what makes this op-aware —
# `conj.(x)` dualizes, so a broadcast that conjugates some operands but not others has
# mismatched axes and is rejected here as a sector mismatch (e.g. `conj.(s) .- t` is
# ill-formed; `conj.(s) .- conj.(t)` is fine).
broadcasted_sector(::Any) = nothing
broadcasted_sector(a::AbstractSectorArray) = sector(a)
function broadcasted_sector(bc::BC.Broadcasted)
    sects = filter(!isnothing, map(broadcasted_sector, bc.args))
    isempty(sects) &&
        throw(ArgumentError("No AbstractSectorArray found in broadcast"))
    s = first(sects)
    all(==(s), sects) || throw(DimensionMismatch("sector mismatch in broadcast"))
    return sector_op(bc.f)(s)
end

function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type)
    s = broadcasted_sector(bc)
    return sector_kron(s, similar(broadcasted_data(bc), elt))
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

# Determine the axes the broadcast result should carry, threading `conj` (which dualizes the
# graded axes) through the tree exactly as `broadcasted_sector` does for `SectorStyle`. This
# makes `similar` op-aware: `conj.(a)` gets dualized axes, and a broadcast that conjugates only
# some operands such as `conj.(a) .- b` is rejected as an axes mismatch.
axes_op(::typeof(conj)) = Base.Fix1(map, conj)
axes_op(::Any) = identity

broadcasted_gradedaxes(::Any) = nothing
broadcasted_gradedaxes(a::AbstractGradedArray) = axes(a)
function broadcasted_gradedaxes(bc::BC.Broadcasted)
    axs = filter(!isnothing, map(broadcasted_gradedaxes, bc.args))
    isempty(axs) && throw(ArgumentError("No AbstractGradedArray found in broadcast"))
    ax = first(axs)
    all(==(ax), axs) || throw(DimensionMismatch("axes mismatch in broadcast"))
    return axes_op(bc.f)(ax)
end

# TODO: Ideally this would incorporate information from all broadcast arguments
# (or their blocktypes) when computing similar, rather than picking one argument.
function Base.similar(bc::BC.Broadcasted{<:GradedStyle}, elt::Type)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractGradedArray, bc′.args)]
    return similar(arg, elt, broadcasted_gradedaxes(bc))
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
