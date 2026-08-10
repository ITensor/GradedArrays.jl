# =============================================================================
#  FusionArray — always-fused symmetric array backed by a matricized
#  `FusedGradedMatrix`.
# =============================================================================

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using TensorKit: TensorKit as TK

"""
    FusionArray{T,S,N,NC,ND,M} <: AbstractArray{T,N}

Always-fused symmetric array: an `N`-dimensional graded array split into `NC` codomain and `ND`
domain legs (`NC + ND == N`), backed by a matricized [`FusedGradedMatrix`](@ref). The external axes
are `GradedOneTo` and may be unfused or unsorted (a sector repeated, or out of `SectorRange` order);
the `matricized` backing is always over the fused-sorted coupled space, and the per-leg sort
permutation relates the two.
"""
struct FusionArray{
        T, S, N, NC, ND, M <: FusedGradedMatrix{T, S},
    } <: AbstractArray{T, N}
    matricized::M
    axes_codomain::NTuple{NC, GradedOneTo{S}}
    axes_domain::NTuple{ND, GradedOneTo{S}}

    function FusionArray(
            matricized::FusedGradedMatrix{T, S},
            axes_codomain::NTuple{NC, GradedOneTo{S}},
            axes_domain::NTuple{ND, GradedOneTo{S}}
        ) where {T, S, NC, ND}
        return new{T, S, NC + ND, NC, ND, typeof(matricized)}(
            matricized, axes_codomain, axes_domain
        )
    end
end

# ============================  Accessors  ============================

axes_codomain(fa::FusionArray) = fa.axes_codomain
axes_domain(fa::FusionArray) = fa.axes_domain

# Domain axes are stored codomain-facing (TensorKit's `domain` convention); `biaxes` dualizes the
# domain half, so a domain leg reads as a dual axis (matching TensorKit's `space(t, i)`) and the
# codomain/domain split rides along. `axes` is the flat form; `codomain`/`domain` recover the halves.
biaxes(fa::FusionArray) = BiTuple(axes_codomain(fa), map(conj, axes_domain(fa)))
Base.axes(fa::FusionArray) = Tuple(biaxes(fa))
Base.size(fa::FusionArray) = map(length, axes(fa))

# Recover the split halves of a `biaxes` `BiTuple`: `codomain` is the first half as-is; `domain`
# re-dualizes the second half (stored dualized as the external view) back to codomain-facing form.
codomain(bt::BiTuple) = bt.t1
domain(bt::BiTuple) = map(conj, bt.t2)

ndims_codomain(fa::FusionArray) = length(axes_codomain(fa))
ndims_domain(fa::FusionArray) = length(axes_domain(fa))

# One-argument `matricize` uses the array's own codomain/domain split, so it is the stored
# matrix directly (see `matricize(::FusionArrayMatricizeStyle, …)` for re-splitting to another).
TensorAlgebra.matricize(fa::FusionArray) = fa.matricized

# ============================  block indexing (unique fusion)  ============================
# Unique fusion only: for non-abelian symmetry a `Block`'s external leg sectors don't pin down the
# block. The returned `UniqueSectorArray` is a view into the block's strided data, so get/set writes
# back in place.

function viewblock(
        a::FusionArray{T, S, N, NC, ND},
        I::Block{N}
    ) where {T, S, N, NC, ND}
    require_unique_fusion(a)
    bk = Int.(Tuple(I))
    sects = ntuple(d -> eachsectoraxis(axes(a, d))[bk[d]], Val(N))
    # The block carries the array's own codomain/domain split. The codomain legs are stored as-is;
    # the domain legs are stored codomain-facing (un-dualed), taken from `axes_domain(a)` rather than
    # the dualized `axes(a)`, matching the block type's storage convention (its `axes` re-dualizes).
    cod = ntuple(i -> sects[i], Val(NC))
    dom = ntuple(j -> eachsectoraxis(axes_domain(a)[j])[bk[NC + j]], Val(ND))
    # Dualize each leg's sector on dual axes to match TensorKit's external-sector indexing.
    blockdata = tensormap(a)[map(r -> isdual(r) ? TKS.dual(label(r)) : label(r), sects)]
    # Fused-sorted axes have one block per sector, so the merged block is the whole block. Only an
    # unfused axis (a repeated sector) stores its positional blocks as one merged block, so then slice
    # each leg to this block's subrange within its merged sector: `invblockmergeperm` maps a fine block
    # to its `Block[subrange]` in the fused-sorted merged axis.
    all(is_fused_sorted, axes(a)) && return UniqueSectorArray(blockdata, cod, dom)
    ranges = ntuple(Val(N)) do d
        g = axes(a, d)
        return only(
            invblockmergeperm(g, sectorsortperm(g), sectormergesort(g))[bk[d]].indices
        )
    end
    return UniqueSectorArray(view(blockdata, ranges...), cod, dom)
end

Base.view(a::FusionArray{T, <:Any, N}, I::Block{N}) where {T, N} = viewblock(a, I)
# Disambiguate the N=1 case against the `Vararg{Block{1}, N}` method.
Base.view(a::FusionArray{T, <:Any, 1}, I::Block{1}) where {T} = viewblock(a, I)

# Rank-0 (scalar) access: a rank-0 `FusionArray` (e.g. a full contraction to a scalar) holds one
# trivial-sector value in its 1×1 matricized block. Read it as a TensorKit scalar and write it into
# that block; the generic block path indexes `FusionMap` by external sectors, of which a rank-0 array
# has none. Defined on the concrete type to take precedence over the `Vararg` block methods, which
# also match a no-argument call at N=0.
Base.getindex(a::FusionArray{<:Any, <:Any, 0}) = TK.scalar(tensormap(a))
function Base.setindex!(a::FusionArray{<:Any, <:Any, 0}, value)
    only(values(matricize(a).blocks))[begin] = value
    return a
end

# A `FusionArray` block is the same unique-fusion `UniqueSectorArray` the abelian backend returns.
# TODO: derive the block data type from the `FusedGradedMatrix` block type (its `D`) rather than
# hardcoding `Array{T, N}`, so non-`Array` storage (GPU, etc.) is preserved — e.g. via
# `Base.promote_op` on `view(A, ::Block)` (the actual returned block type). Also only well-defined
# for unique fusion (blocks are `UniqueSectorArray` only then); tie it to that guard.
function blocktype(::Type{<:FusionArray{T, S, N, NC, ND}}) where {T, S, N, NC, ND}
    return UniqueSectorArray{T, S, N, NC, ND, Array{T, N}}
end
blocktype(a::FusionArray) = blocktype(typeof(a))

# The block storage type is the datatype of the blocks, so only `blocktype` is type-specific.
datatype(::Type{T}) where {T <: FusionArray} = datatype(blocktype(T))
datatype(a::FusionArray) = datatype(typeof(a))
sectortype(::Type{<:FusionArray{T, S}}) where {T, S} = S

# ============================  block storage interface (unique fusion)  ============================
# Unique fusion only: the symmetry-allowed blocks, computed from the axes since a `FusionArray` holds
# the coupled matrix, not per-block data.
function eachblockstoredindex(a::FusionArray)
    require_unique_fusion(a)
    return allowedblocks(axes(a))
end

# Sparse view of the stored (symmetry-allowed) external blocks, the N-dim analog of
# `FusedGradedMatrixBlocks` / `AbelianBlocks`. `blockstoredlength`, `isstored(a, ::Block)`, and scalar
# `getindex`/`setindex!` all derive from this generically. A stored entry shares data via `viewblock`;
# an unstored entry is a symmetry-forbidden block and errors.
struct FusionArrayBlocks{T, S, N, A <: FusionArray{T, S, N}} <:
    AbstractSparseArray{UniqueSectorArray{T, S, N}, N}
    parent::A
end
BlockArrays.blocks(a::FusionArray) = FusionArrayBlocks(a)
Base.size(b::FusionArrayBlocks) = blocklength.(axes(b.parent))

function SparseArraysBase.eachstoredindex(::IndexCartesian, b::FusionArrayBlocks)
    return [CartesianIndex(Int.(Tuple(bI))) for bI in eachblockstoredindex(b.parent)]
end
function SparseArraysBase.storedvalues(b::FusionArrayBlocks)
    return [view(b.parent, bI) for bI in eachblockstoredindex(b.parent)]
end
function SparseArraysBase.isstored(
        b::FusionArrayBlocks{<:Any, <:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return Block(I...) in eachblockstoredindex(b.parent)
end
function SparseArraysBase.getstoredindex(
        b::FusionArrayBlocks{<:Any, <:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return view(b.parent, Block(I...))
end
function SparseArraysBase.setstoredindex!(
        b::FusionArrayBlocks{<:Any, <:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    copy_sector!(view(b.parent, Block(I...)), value)
    return b
end
function SparseArraysBase.getunstoredindex(
        b::FusionArrayBlocks{<:Any, <:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return error("Block $(I) is not stored.")
end
function SparseArraysBase.setunstoredindex!(
        b::FusionArrayBlocks{<:Any, <:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    return error("Block $(I) is not stored.")
end

# ============================  similar  ============================
# `similar` must build a `FusionArray`. Without explicit axes, preserve the prototype's own
# codomain/domain split (like `copy`); with explicit flat axes the split is unrecoverable (see the
# `copy` note), so put all axes in the codomain, matching the broadcast `similar`.
function Base.similar(a::FusionArray, ::Type{T}) where {T}
    return FusionArray(similar(matricize(a), T), axes_codomain(a), axes_domain(a))
end
function Base.similar(
        a::FusionArray, ::Type{T}, axes::Tuple{GradedOneTo{S}, Vararg{GradedOneTo{S}}}
    ) where {T, S}
    return TensorAlgebra.similar_map(a, T, axes, ())
end
# Empty axes build a rank-0 `FusionArray`; the sector type is read from the prototype (it cannot
# be inferred from the empty axes). Without this, `similar(fa, T, ())` falls through to Base and
# returns a plain rank-0 `Array`.
function Base.similar(a::FusionArray, ::Type{T}, ::Tuple{}) where {T}
    return FusionArray{T, sectortype(a)}(undef, (), ())
end

# ============================  copyto!  ============================
# Copy `src` into `dest` as an identity leg permutation into `dest`'s own codomain/domain split, so
# `bipermutedims!` bends `src` when the two splits differ and validates the external axes. The generic
# `AbstractArray` `copyto!` scalar-indexes, which errors on forbidden (symmetry-disallowed) blocks.
# `Base.copy!` rides on this: it checks `axes` equality and forwards here.
function Base.copyto!(dest::FusionArray, src::FusionArray)
    bipermutedims!(
        dest, src,
        ntuple(identity, Val(ndims_codomain(dest))),
        ntuple(i -> ndims_codomain(dest) + i, Val(ndims_domain(dest)))
    )
    return dest
end

# ============================  ==  ============================
# Compare the shared array contents, not the internal matricization: a `FusionArray` is an array
# whose `axes` are the flat external legs, so two with equal `axes` are equal iff their data matches.
# Rematricize `b` to `a`'s codomain/domain split, then compare coupled blocks (unlike a `TensorMap`,
# whose split is part of its identity). This also avoids Base's element-wise fallback, which would
# index forbidden blocks.
function Base.:(==)(a::FusionArray, b::FusionArray)
    axes(a) == axes(b) || return false
    return matricize(a) == matricize(b, Val(ndims_codomain(a)))
end

# ============================  dot  ============================
# Sum the coupled-block inner products of the (norm-preserving) matricized forms, rather than the
# block-walk that iterates `eachblockstoredindex` (unique-fusion-only for `FusionArray`)
# and scalar-indexes. `b` is rematricized to `a`'s split so the coupled blocks line up.
function LinearAlgebra.dot(a::FusionArray, b::FusionArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    ma = matricize(a)
    mb = matricize(b, Val(ndims_codomain(a)))
    init = zero(LinearAlgebra.dot(zero(eltype(a)), zero(eltype(b))))
    return sum(keys(ma.blocks); init) do c
        return LinearAlgebra.dot(ma.blocks[c], mb.blocks[c])
    end
end

# `LinearAlgebra.normalize` infers its result eltype via `typeof(first(a)/nrm)`, which scalar-indexes
# opaque block storage; route through the graded `/` instead.
function LinearAlgebra.normalize(a::FusionArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# ============================  permutedims  ============================
# Route through `permutedimsopadd!` (which forwards to the `FusionArray` `bipermutedimsopadd!`), so
# the fermion braiding/bend signs are applied by TensorKit. Without this, Base's generic
# `permutedims` allocates via `similar` and scalar-permutes the data, dropping the sign.
function Base.permutedims(a::FusionArray{<:Any, <:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> axes(a)[perm[i]], Val(N))
    return permutedims!(similar(a, dest_axes), a, perm)
end
function Base.permutedims!(
        y::FusionArray{<:Any, <:Any, N}, x::FusionArray{<:Any, <:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ============================  conj  ============================
# Conjugate while keeping the codomain/domain split, unlike a plain `conj.(a)`
# broadcast, which materializes into a fresh array and so lands at the all-codomain split. Fill a
# same-split destination (per-leg axes dualized) with a single `op = conj` permute-add over the
# identity biperm, so the TensorKit-backed transform folds in the leg-reversal fermion sign and the
# non-abelian recoupling that a bare block conjugation would drop.
function Base.conj(fa::FusionArray{<:Any, <:Any, <:Any, NC, ND}) where {NC, ND}
    dest = TensorAlgebra.similar_map(
        fa, map(dual, axes_codomain(fa)), map(dual, axes_domain(fa))
    )
    TensorAlgebra.bipermutedimsopadd!(
        dest, conj, fa, ntuple(identity, Val(NC)), ntuple(i -> NC + i, Val(ND)), true, false
    )
    return dest
end

# ============================  matrix operations (guarded)  ============================
# Matrix / linear-algebra operations live only on the matrix storage type `FusedGradedMatrix`. On a
# `FusionArray` they would otherwise fall through to the generic `AbstractArray` methods, which
# scalar-index into a dense, non-graded result. Error instead; matricize to a `FusedGradedMatrix`
# first (`matricize(a)`).
Base.adjoint(A::FusionArray) = _matrix_op_error(adjoint, A)
Base.transpose(A::FusionArray) = _matrix_op_error(transpose, A)
Base.:*(A::FusionArray, B::FusionArray) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::FusionArray) = _matrix_op_error(LinearAlgebra.tr, A)
MAK.one!(A::FusionArray) = _matrix_op_error(MAK.one!, A)
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::FusionArray) = _matrix_op_error($f, A)
end

# ============================  TensorMap conversion  ============================

"""
    TK.TensorMap(fa::FusionArray)

Convert a `FusionArray` to a `TK.TensorMap`, building the codomain/domain product
spaces from the per-leg axes and copying each coupled-sector block.
"""
function TK.TensorMap(fa::FusionArray)
    # Derive the space type from the sector type (not a leg) so the rank-0 case, with no legs, still
    # resolves the trivial `one(Sp)` codomain/domain. The `matricized` backing is over the fused-sorted
    # coupled space, so build each leg's space from `sectormergesort` of the (possibly unsorted) stored
    # axis; the `TensorMap` is the fused-sorted TensorKit view, and the stored-axis order is reapplied
    # only when going back to a dense array (`Array`).
    Sp = typeof(ElementarySpace(trivial_gradedrange(sectortype(fa))))
    codsp = mapreduce(
        ElementarySpace ∘ sectormergesort,
        TK.:⊗,
        axes_codomain(fa);
        init = one(Sp)
    )
    domsp =
        mapreduce(ElementarySpace ∘ sectormergesort, TK.:⊗, axes_domain(fa); init = one(Sp))
    return copy!(TK.TensorMap{eltype(fa)}(undef, codsp, domsp), fa)
end

"""
    FusionArray(t::TK.AbstractTensorMap)

Build a `FusionArray` from a `TensorMap`, taking the per-leg external axes from its codomain
and domain spaces.
"""
function FusionArray(t::TK.AbstractTensorMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(t)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(t)))
    return copy!(FusionArray{eltype(t)}(undef, axes_codomain, axes_domain), t)
end

# Copy a matrix `TensorMap` (one codomain and one domain leg) block-wise into a `FusedGradedMatrix`.
function Base.copy!(m::FusedGradedMatrix, t::TK.AbstractTensorMap{<:Any, <:Any, 1, 1})
    for c in TK.blocksectors(t)
        copy!(m.blocks[SectorRange(c)], TK.block(t, c))
    end
    return m
end

# Copy a `FusedGradedMatrix` block-wise into a matrix `TensorMap` (one codomain and one domain leg).
function Base.copy!(t::TK.AbstractTensorMap{<:Any, <:Any, 1, 1}, m::FusedGradedMatrix)
    for (c, b) in pairs(m.blocks)
        copy!(TK.block(t, label(c)), b)
    end
    return t
end

# Copy a `TensorMap` into a `FusionArray` with matching codomain and domain (same coupled blocks).
function Base.copy!(a::FusionArray, t::TK.AbstractTensorMap)
    (TK.numout(t) == ndims_codomain(a) && TK.numin(t) == ndims_domain(a)) ||
        throw(DimensionMismatch("TensorMap codomain/domain does not match the FusionArray"))
    for c in TK.blocksectors(t)
        copy!(matricize(a).blocks[SectorRange(c)], TK.block(t, c))
    end
    return a
end

# Copy a `FusionArray` into a `TensorMap` with matching codomain and domain (same coupled blocks).
function Base.copy!(t::TK.AbstractTensorMap, a::FusionArray)
    (TK.numout(t) == ndims_codomain(a) && TK.numin(t) == ndims_domain(a)) ||
        throw(DimensionMismatch("TensorMap codomain/domain does not match the FusionArray"))
    for (c, b) in pairs(matricize(a).blocks)
        copy!(TK.block(t, label(c)), b)
    end
    return t
end

# ============================  construction from axes  ============================

# Axes are given codomain-facing (un-dualized), the same convention they are stored in,
# matching `similar_map`/`unmatricize`. The sector type is read from the axes, so at least one leg
# must be present; the rank-0 (both-empty) case takes `S` explicitly through the method below.
function FusionArray{T}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    S = sectortype(first((axes_codomain..., axes_domain...)))
    return FusionArray{T, S}(undef, axes_codomain, axes_domain)
end

# Fuse each side's per-leg axes into its coupled `GradedOneTo` (through the GradedArrays interface,
# which uses the TensorKitSectors fusion rules), seeding empty groups with the trivial sector.
# `FusedGradedMatrix{T}(undef, …)` then allocates the reduced blocks. With `S` given as a type
# parameter this also covers rank-0: both groups fuse to the trivial sector, giving a 1×1 scalar.
function FusionArray{T, S}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T, S}
    init = trivial_gradedrange(S)
    coupled_codomain = reduce(tensor_product, axes_codomain; init)
    coupled_domain = reduce(tensor_product, axes_domain; init)
    m = FusedGradedMatrix{T}(undef, coupled_codomain, coupled_domain)
    return FusionArray(m, axes_codomain, axes_domain)
end

# A `FusionArray` source reproduces a `FusionArray`, routing straight to its `undef` constructor.
function TensorAlgebra.similar_map(
        ::FusionArray, ::Type{T},
        axes_codomain::Tuple{GradedOneTo, Vararg{GradedOneTo}},
        axes_domain::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        ::FusionArray, ::Type{T},
        axes_codomain::Tuple{}, axes_domain::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end
# Rank-0: the empty axes carry no sector type, so read it from the `FusionArray` prototype.
function TensorAlgebra.similar_map(
        fa::FusionArray, ::Type{T}, axes_codomain::Tuple{}, axes_domain::Tuple{}
    ) where {T}
    return FusionArray{T, sectortype(fa)}(undef, axes_codomain, axes_domain)
end

# Fill the reduced coupled-sector blocks in place, forwarding to the matricized `FusedGradedMatrix`
# (which fills via `eachblockstoredindex`). Construct with `FusionArray{T}(undef, …)` first.
for f! in (:rand!, :randn!)
    @eval begin
        Random.$f!(rng::AbstractRNG, fa::FusionArray) = (Random.$f!(rng, matricize(fa)); fa)
        Random.$f!(fa::FusionArray) = Random.$f!(Random.default_rng(), fa)
    end
end

# ============================  in-place primitives / algebra  ============================
# `zero!`/`scale!`/`norm`/`real`/`imag` are a block-walk over `eachblockstoredindex` on the fused
# arrays; `FusionArray`'s own `eachblockstoredindex` is unique-fusion-only, so forward to the
# matricized `FusedGradedMatrix` instead. `+`/`-` are left to the `AbstractArray` broadcast machinery.

TensorAlgebra.zero!(fa::FusionArray) = (zero!(matricize(fa)); fa)
TensorAlgebra.scale!(fa::FusionArray, α::Number) = (scale!(matricize(fa), α); fa)
LinearAlgebra.norm(fa::FusionArray, p::Real = 2) = LinearAlgebra.norm(matricize(fa), p)
Base.fill!(fa::FusionArray, v) = (fill!(matricize(fa), v); fa)
Base.iszero(fa::FusionArray) = iszero(matricize(fa))

# Copy the matricized matrix and reuse the axes. Defined directly (rather than through `similar`)
# because a generic `similar` over flat axes cannot recover the
# codomain/domain split a `FusionArray` needs; `copy` must preserve it (e.g. `one!!(copy(A), …)`
# relies on the copy keeping `A`'s split so the identity fill lands in the stored matrix).
function Base.copy(fa::FusionArray)
    return FusionArray(copy(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end

# Sort each external axis by sector and merge repeated sectors. The internal fused storage is already
# canonical (blocks are keyed by coupled sector), so the data is unchanged: only the external axes are
# re-labeled to their merged-sorted form, which re-slices the same coupled blocks into the merged
# external blocks. Copies the matricized matrix so the result is an independent array.
function sectormergesort(a::FusionArray)
    return FusionArray(
        copy(matricize(a)),
        map(sectormergesort, axes_codomain(a)),
        map(sectormergesort, axes_domain(a))
    )
end

function Base.real(fa::FusionArray)
    return FusionArray(real(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end
function Base.imag(fa::FusionArray)
    return FusionArray(imag(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end

function Base.:*(a::FusionArray, x::Number)
    return FusionArray(matricize(a) * x, axes_codomain(a), axes_domain(a))
end
Base.:*(x::Number, a::FusionArray) = a * x
Base.:-(a::FusionArray) = (-one(eltype(a))) * a
function Base.:/(a::FusionArray, x::Number)
    return FusionArray(matricize(a) / x, axes_codomain(a), axes_domain(a))
end
Base.:+(a::FusionArray, b::FusionArray) = a .+ b
Base.:-(a::FusionArray, b::FusionArray) = a .- b

# ============================  broadcasting  ============================
# Opt in to the graded linear-broadcast machinery so linear combinations (`a + b`, `2a - b`, …)
# materialize through `bipermutedimsopadd!`. The shared graded `copyto!` and the
# `LinearBroadcasted` fold do the work; only allocation is `FusionArray`-specific.

struct FusionArrayStyle{N} <: AbstractGradedStyle{N} end
FusionArrayStyle{N}(::Val{M}) where {N, M} = FusionArrayStyle{M}()

function BC.BroadcastStyle(::Type{<:FusionArray{<:Any, <:Any, N}}) where {N}
    return FusionArrayStyle{N}()
end

# Build the result with all axes in the codomain (matching TensorKit's move-to-codomain convention
# for `+`/`-`), so operands with any codomain/domain split are bent into it by the fold.
# TODO: This picks the default block data type and so does not preserve non-`Array` block types
# (e.g. GPU arrays). Carry the block data type on `FusionArrayStyle` and use it here, as
# BlockSparseArrays does for its broadcast style.
function Base.similar(bc::BC.Broadcasted{<:FusionArrayStyle}, elt::Type)
    return FusionArray{elt}(undef, axes(flattenlinear(bc)), ())
end

function Base.copyto!(dest::FusionArray, bc::BC.Broadcasted{<:FusionArrayStyle})
    return copyto!(dest, flattenlinear(bc))
end

# ============================  bipermutedimsopadd! (permute primitive)  ============================
# `y = α * op.(permute(x, …)) + β * y`, delegated to the `AbstractTensorMap` `bipermutedimsopadd!`
# (fusion-tree recombination plus braiding/fermion signs). Wrapping `y`/`x` as `FusionMap`s shares
# their blocks, so TensorKit writes the result into `y` in place.

function TensorAlgebra.bipermutedimsopadd!(
        y::FusionArray, op, x::FusionArray,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    TensorAlgebra.bipermutedimsopadd!(
        tensormap(y), op, tensormap(x), perm_codomain, perm_domain, α, β
    )
    return y
end

# A `FusedGradedMatrix` source (e.g. a matricized factorization output permuted into a `FusionArray`
# destination) is wrapped zero-copy as a one-codomain/one-domain `FusionArray` — its stored matrix is
# already the `FusionArray` matricized form — and forwarded to the method above. Without this the
# generic block-wise permute-add would block-index the `FusionArray` destination, which it
# does not support.
function TensorAlgebra.bipermutedimsopadd!(
        y::FusionArray, op, x::FusedGradedMatrix,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    x_fa = FusionArray(x, (axes(x, 1),), (dual(axes(x, 2)),))
    return TensorAlgebra.bipermutedimsopadd!(y, op, x_fa, perm_codomain, perm_domain, α, β)
end

# ============================  fermionic twist  ============================
# The contraction twist scales blocks by a per-fusion-tree fermion phase. Wrapping `a` as a
# `FusionMap` shares its blocks, so `TK.twist!` scales them in place.
function twist!(a::FusionArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    TK.twist!(tensormap(a), dims)
    return a
end

# ============================  TensorAlgebra primitive interface  ============================
# `matricize` returns the stored `FusedGradedMatrix`, `unmatricize` puts axes back on. Because
# `FusedGradedMatrix` already has `mul!` and block-wise factorizations, contraction and
# factorizations ride the generic `TensorAlgebra` machinery with no high-level overloads.

struct FusionArrayMatricizeStyle <: MatricizeStyle end

TensorAlgebra.MatricizeStyle(::Type{<:FusionArray}) = FusionArrayMatricizeStyle()

# When the requested split matches the stored split this is the stored matrix. Otherwise it is a
# leg bend (the `matricizeopperm` fast path only reaches here with an identity permutation, so
# legs stay in order and only the codomain/domain boundary moves), which for a `FusionArray` is
# not a free reshape. Re-split with the array's own `bipermutedims`, then take its 1-arg
# `matricize`. This is what lets a contraction over a subset of legs matricize a factor whose
# stored split differs.
function TensorAlgebra.matricize(
        ::FusionArrayMatricizeStyle,
        fa::FusionArray,
        ::Val{K}
    ) where {K}
    K == ndims_codomain(fa) && return matricize(fa)
    N = ndims(fa)
    # TODO: Once `permutedims` on a `FusionArray` routes to `bipermutedimsopadd!`, bend with the
    # identity-permutation `permutedims` directly (ideally a `[bi]permutedims(fa, Val(K))` split-only
    # spelling). See the "Unified `permutedims` surface" follow-up.
    fa_bent = TensorAlgebra.bipermutedims(
        fa, ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K))
    )
    return matricize(fa_bent)
end

function TensorAlgebra.unmatricize(
        ::FusionArrayMatricizeStyle, m::FusedGradedMatrix, axes_codomain::Tuple,
        axes_domain::Tuple
    )
    return FusionArray(m, axes_codomain, axes_domain)
end

# ============================  contraction (SectorMatricize path)  ============================
# Contraction dispatches through `default_contract_algorithm(::FusionArray, …)`, which
# fixes the `SectorMatricize`/`TwistedSectorMatricize` styles, so a `FusionArray` contraction rides
# that path rather than `FusionArrayMatricizeStyle`.

# Twist only the right factor; the left factor and the output use plain `SectorMatricize`.
function TensorAlgebra.default_contract_algorithm(
        ::Type{<:FusionArray},
        ::Type{<:FusionArray}
    )
    return TensorAlgebra.Matricize(
        SectorMatricize(), TwistedSectorMatricize(), SectorMatricize()
    )
end

function TensorAlgebra.matricize(::SectorMatricize, fa::FusionArray, ndims_cod::Val)
    return matricize(FusionArrayMatricizeStyle(), fa, ndims_cod)
end

function TensorAlgebra.unmatricizeperm!(
        ::SectorMatricize, a_dest::FusionArray{<:Any, <:Any, N}, m::FusedGradedMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    ) where {N}
    # Permute `a_dest` into the matricized leg order to get the matricized-order axes with correct
    # per-leg duality; the permuted data is discarded. Wrap `m` in those axes, then permute back.
    # TODO: Switch to `permutedims` once it routes through `bipermutedimsopadd!` (see the "Unified
    # `permutedims` surface" follow-up), as in the `matricize` leg-bend above.
    template = TensorAlgebra.bipermutedims(a_dest, invperm_codomain, invperm_domain)
    tmp = FusionArray(m, axes_codomain(template), axes_domain(template))
    perm_dest = invperm((invperm_codomain..., invperm_domain...))
    ndims_cod_dest = ndims_codomain(a_dest)
    perm_codomain = ntuple(i -> perm_dest[i], Val(ndims_cod_dest))
    perm_domain = ntuple(i -> perm_dest[ndims_cod_dest + i], Val(N - ndims_cod_dest))
    # TODO: Switch to `Base.permutedims!` once it is defined to route through `bipermutedimsopadd!`.
    bipermutedims!(a_dest, tmp, perm_codomain, perm_domain)
    return a_dest
end

# ============================  concatenation  ============================
# Place whole blocks (no scalar indexing) with the inner `concatenate!` on the block containers. That
# works because `FusionArrayBlocks` is an `AbstractSparseArray`, so the placement visits only the
# stored (symmetry-allowed) blocks, whereas a dense path would touch forbidden positions.
function TensorAlgebra.concatenate!(dest::FusionArray, dims, args...)
    TensorAlgebra.concatenate!(blocks(dest), dims, blocks.(args)...)
    return dest
end
# Route `Base.cat` through the same machinery so it uses the graded destination and placement.
Base._cat(dims, as::FusionArray...) = TensorAlgebra.concatenate(dims, as...)

# ============================  project (dense -> symmetric)  ============================
# The dense-to-symmetric projection for the `FusionArray` backend is delegated to TensorKit in the
# shared `unchecked_project_graded` worker (`gradedconstructors.jl`): it projects over the equivalent
# `ElementarySpace`s and wraps the resulting `TensorMap` with `FusionArray(t)`. `unproject` below is
# the dense inverse used by `TA.project`'s verification.

# In-place projection into a preallocated destination: view `dest` as a `FusionMap` (sharing its
# blocks) and project the dense `src` straight into those blocks through TensorKit, which drops the
# forbidden regions and handles a lower-rank `src` reshaped into a flux-canceling aux leg. The generic
# `AbstractArray` `projectto!` would scalar-`copyto!` and error mid-write on a forbidden block.
function TensorAlgebra.projectto!(dest::FusionArray, src::AbstractArray)
    TensorAlgebra.projectto!(tensormap(dest), src)
    return dest
end

# Dense form. The matricized backing has no `eachblockstoredindex` (the generic dense path), so
# materialize through the zero-copy `FusionMap` view. The view is fused-sorted per leg, so for an
# unfused/unsorted stored axis, move each leg's sector blocks back to the stored order (whole-block
# moves preserve the array type, e.g. GPU).
function Base.Array(fa::FusionArray{<:Any, <:Any, N}) where {N}
    dense = convert(Array, tensormap(fa))
    all(is_fused_sorted, axes(fa)) && return dense
    sortedlengths = map(g -> Vector(blocklengths(g))[sortperm(sectors(g))], axes(fa))
    invperms = ntuple(d -> Block.(invperm(sortperm(sectors(axes(fa)[d])))), Val(N))
    return parent(BlockedArray(dense, sortedlengths...)[invperms...])
end
# Rank-0: TensorKit's `convert(Array, ::rank-0 TensorMap)` hits a VectorInterface `add!` gap for some
# eltypes (e.g. `Float32`), so build the 0-dim array from the scalar directly.
Base.Array(fa::FusionArray{<:Any, <:Any, 0}) = fill(fa[])

# `unproject` is the dense inverse of `projectto!` used by `TA.project`'s verification, and the dense
# form at an arbitrary codomain split `Val{K}` used to compare tensors across backends (which agree
# only at a common split). When `K` is the array's own split the `TensorMap` conversion undoes the
# domain-leg bend directly; otherwise bend the legs to a `K`-codomain split first. The bend carries
# the fermion signs.
function TensorAlgebra.unproject(fa::FusionArray, ::Val{K}) where {K}
    N = ndims(fa)
    0 <= K <= N ||
        throw(
        ArgumentError(
            "`unproject` codomain split $K is out of range for a $N-index array"
        )
    )
    K == ndims_codomain(fa) && return Array(fa)
    fa_bent = TensorAlgebra.bipermutedims(
        fa, ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K))
    )
    return Array(fa_bent)
end

# ============================  show  ============================

function Base.summary(io::IO, fa::FusionArray)
    print(
        io, Base.dims2string(size(fa)), " ", nameof(typeof(fa)),
        " (codomain ", ndims_codomain(fa), ", domain ", ndims_domain(fa), ")"
    )
    return nothing
end

Base.show(io::IO, fa::FusionArray) = summary(io, fa)

function Base.show(io::IO, ::MIME"text/plain", fa::FusionArray)
    summary(io, fa)
    println(io, ":")
    # Show the per-leg axes as stored (domain axes codomain-facing), so the printed duality reflects
    # storage rather than the on-the-fly dualization `axes(fa)` applies to domain legs.
    for (d, g) in enumerate(axes_codomain(fa))
        print(io, "  Codomain Dim $d: ")
        show(io, g)
        println(io)
    end
    for (d, g) in enumerate(axes_domain(fa))
        print(io, "  Domain Dim $d: ")
        show(io, g)
        println(io)
    end
    show(io, MIME"text/plain"(), matricize(fa))
    return nothing
end
