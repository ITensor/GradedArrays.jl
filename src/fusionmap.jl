# =============================================================================
#  FusionMap — a `FusionArray`/`FusedGradedMatrix` reinterpreted as an
#  `AbstractTensorMap`, so TensorKit machinery operates on the owned blocks
#  directly (no scratch `TensorMap` round-trip).
# =============================================================================

using Dictionaries: gettoken, gettokenvalue
using StridedViews: StridedView
using TensorKit: TensorKit as TK, AbstractTensorMap, ←

"""
    FusionMap{T,S,N₁,N₂} <: TK.AbstractTensorMap{T,S,N₁,N₂}

An `AbstractTensorMap` view of a matricized `FusedGradedMatrix` with a given codomain/domain
`HomSpace`. It shares the underlying blocks with the `FusionArray` it comes from, so TensorKit
operations (permute, twist, …) mutate those blocks in place instead of round-tripping through a
scratch `TensorMap`. `block`/`subblock` present the stored blocks as the reduced coupled-sector
matrices and their per-fusion-tree strided views.
"""
struct FusionMap{T, S, N₁, N₂, M <: FusedGradedMatrix{T}, SBS, BS} <:
    AbstractTensorMap{T, S, N₁, N₂}
    matricized::M
    space::TK.TensorMapSpace{S, N₁, N₂}
    # `subblock` is called once per fusion tree by TensorKit's generic transform kernel; caching
    # these space-derived tables at wrap time keeps it O(1) instead of rebuilding them per call.
    subblockstructure::SBS
    blockstructure::BS

    function FusionMap(
            matricized::FusedGradedMatrix{T}, space::TK.TensorMapSpace{S, N₁, N₂}
        ) where {T, S, N₁, N₂}
        subblockstructure = TK.subblockstructure(space)
        blockstructure = TK.blockstructure(space)
        return new{
            T,
            S,
            N₁,
            N₂,
            typeof(matricized),
            typeof(subblockstructure),
            typeof(blockstructure),
        }(
            matricized, space, subblockstructure, blockstructure
        )
    end
end

# ============================  interface  ============================

TK.space(fm::FusionMap) = fm.space
TensorAlgebra.matricize(fm::FusionMap) = fm.matricized

# The stored block for a coupled sector is exactly TensorKit's reduced block (same shape and
# basis), so hand it over directly, zero-copy.
TK.block(fm::FusionMap, c::TKS.Sector) = fm.matricized.blocks[SectorRange(c)]

# A fusion tree pair occupies a strided sub-region of its coupled block. TensorKit's
# `subblockstructure` gives the `(size, strides, offset)` into the whole flat data vector; the
# block is contiguous, so its own base offset (from `blockstructure`) turns that into a
# block-relative offset. The block is column-major, so the strides carry over unchanged.
function TK.subblock(
        fm::FusionMap{T, S, N₁, N₂}, (
            f₁,
            f₂,
        )::Tuple{TK.FusionTree{I, N₁}, TK.FusionTree{I, N₂}}
    ) where {T, S, N₁, N₂, I <: TKS.Sector}
    found, token = gettoken(fm.subblockstructure, (f₁, f₂))
    @boundscheck found ||
        throw(TK.SectorMismatch("fusion tree pair $((f₁, f₂)) is not present"))
    sz, str, offset = gettokenvalue(fm.subblockstructure, token)
    block = TK.block(fm, f₁.coupled)
    _, range = fm.blockstructure[f₁.coupled]
    block_offset = offset - (first(range) - 1)
    # A non-dense block (e.g. a `Diagonal` factorization factor) has no strided sub-region: only the
    # whole block is representable, which is the single fusion tree of a `1←1` sector. Hand it back
    # as-is; a proper sub-region would require densifying, so error rather than copy silently. This
    # mirrors TensorKit's `DiagonalTensorMap`, whose `subblock` only covers the diagonal tree.
    if !(block isa DenseArray)
        (block_offset == 0 && size(block) == sz) ||
            error("cannot take a strided sub-region of a non-dense $(typeof(block)) block")
        return block
    end
    return StridedView(vec(block), sz, str, block_offset)
end

# Allocate a fresh `FusionMap` (backed by its own `FusedGradedMatrix`) for the requested space,
# via the `FusionArray` undef constructor.
function Base.similar(::FusionMap, ::Type{T}, V::TK.TensorMapSpace) where {T <: Number}
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(V)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(V)))
    return FusionMap(matricize(FusionArray{T}(undef, axes_codomain, axes_domain)), V)
end

# ============================  conversions (share storage)  ============================

"""
    FusionMap(fa::FusionArray)

View a `FusionArray` as an `AbstractTensorMap`, sharing its matricized blocks (zero-copy).
"""
function FusionMap(fa::FusionArray)
    # Derive the space type from the sector type (not a leg) so the rank-0 case, with no legs, still
    # resolves the trivial `one(Sp)` codomain/domain.
    Sp = typeof(ElementarySpace(trivial_gradedrange(sectortype(fa))))
    codomain = mapreduce(ElementarySpace, TK.:⊗, axes_codomain(fa); init = one(Sp))
    domain = mapreduce(ElementarySpace, TK.:⊗, axes_domain(fa); init = one(Sp))
    return FusionMap(matricize(fa), codomain ← domain)
end

"""
    FusionArray(fm::FusionMap)

View a `FusionMap` back as a `FusionArray`, sharing its matricized blocks (zero-copy).
"""
function FusionArray(fm::FusionMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(fm)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(fm)))
    return FusionArray(matricize(fm), axes_codomain, axes_domain)
end

# ============================  functional conversion seam  ============================
# `tensormap` / `fusionarray` are the type-agnostic seam callers use in place of the concrete
# `FusionMap` / `FusionArray` constructors, so the TensorKit-view type can change without touching
# every call site. `tensormap` is the zero-copy `FusionMap` view; `fusionarray` reconstructs a
# `FusionArray` from any `AbstractTensorMap` (zero-copy for a `FusionMap`, copying otherwise).
tensormap(a::FusionArray) = FusionMap(a)
fusionarray(t::TK.AbstractTensorMap) = FusionArray(t)
