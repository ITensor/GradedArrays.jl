# =============================================================================
#  GradedArray — always-fused symmetric array backed by a matricized
#  `FusedGradedMatrix`.
# =============================================================================

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using TensorKit: TensorKit as TK, ←

const GA = GradedArrays

"""
    GradedArray{T,S,N,NC,ND,M} <: AbstractArray{T,N}

Always-fused symmetric array: an `N`-dimensional graded array split into `NC` codomain and `ND`
domain legs (`NC + ND == N`), backed by a matricized [`FusedGradedMatrix`](@ref). The external axes
are `GradedOneTo` and may be unfused or unsorted (a sector repeated, or out of `SectorRange` order);
the `matricized` backing is always over the fused-sorted coupled space, and the per-leg sort
permutation relates the two.
"""
struct GradedArray{
        T, S, N, NC, ND, M <: AbstractFusedGradedMatrix{T, S},
    } <: AbstractArray{T, N}
    matricized::M
    axes_codomain::NTuple{NC, GradedOneTo{S}}
    axes_domain::NTuple{ND, GradedOneTo{S}}

    function GradedArray(
            matricized::AbstractFusedGradedMatrix{T, S},
            axes_codomain::NTuple{NC, AbstractGradedOneTo{S}},
            axes_domain::NTuple{ND, AbstractGradedOneTo{S}}
        ) where {T, S, NC, ND}
        return new{T, S, NC + ND, NC, ND, typeof(matricized)}(
            matricized, map(GradedOneTo, axes_codomain), map(GradedOneTo, axes_domain)
        )
    end
end

# ============================  Accessors  ============================

# `axes_codomain`/`axes_domain` are the stored primitives (field reads); `biaxes` derives from them
# generically (in `tensoralgebra.jl`). Domain axes are stored codomain-facing (TensorKit's `domain`
# convention); the derived `biaxes` dualizes the domain half, so a domain leg reads as a dual axis
# (matching TensorKit's `space(t, i)`), and `axes` is the flat form.
axes_codomain(fa::GradedArray) = fa.axes_codomain
axes_domain(fa::GradedArray) = fa.axes_domain

Base.axes(fa::GradedArray) = Tuple(biaxes(fa))
Base.size(fa::GradedArray) = map(length, axes(fa))

# Build a codomain/domain `BiTuple` from the two halves in codomain-facing (un-dualized) form,
# dualizing the domain half for storage; `codomain`/`domain` recover them. Internal, analogous to
# `TensorKit.HomSpace` (also accessed with `codomain`/`domain`), but over `gradedrange` axes or
# sectors rather than vector spaces.
bispace(codomain, domain) = BiTuple(codomain, map(conj, domain))
codomain(bt::BiTuple) = bt.t1
domain(bt::BiTuple) = map(conj, bt.t2)

ndims_codomain(fa::GradedArray) = length(axes_codomain(fa))
ndims_domain(fa::GradedArray) = length(axes_domain(fa))

# One-argument `matricize` uses the array's own codomain/domain split, so it is the stored
# matrix directly (see `matricize(::GradedMatricize, …)` for re-splitting to another).
TensorAlgebra.matricize(fa::GradedArray) = fa.matricized

# ============================  block indexing (unique fusion)  ============================
# Unique fusion only: for non-abelian symmetry a `Block`'s external leg sectors don't pin down the
# block. The returned `UniqueSectorArray` is a view into the block's strided data, so get/set writes
# back in place.

function viewblock(
        a::GradedArray{T, S, N, NC, ND},
        I::Block{N}
    ) where {T, S, N, NC, ND}
    assert_block_indexing()
    require_unique_fusion(a)
    bk = Int.(Tuple(I))
    sects = ntuple(d -> eachsectoraxis(axes(a, d))[bk[d]], Val(N))
    # The block carries the array's own codomain/domain split. The codomain legs are stored as-is;
    # the domain legs are stored codomain-facing (un-dualed), taken from `axes_domain(a)` rather than
    # the dualized `axes(a)`, matching the block type's storage convention (its `axes` re-dualizes).
    cod = ntuple(i -> sects[i], Val(NC))
    dom = ntuple(j -> eachsectoraxis(axes_domain(a)[j])[bk[NC + j]], Val(ND))
    # Dualize each leg's sector on dual axes to match TensorKit's external-sector indexing.
    blockdata = to_tensormap(a)[map(r -> isdual(r) ? TKS.dual(label(r)) : label(r), sects)]
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

Base.view(a::GradedArray{T, <:Any, N}, I::Block{N}) where {T, N} = viewblock(a, I)
# Disambiguate the N=1 case against the `Vararg{Block{1}, N}` method.
Base.view(a::GradedArray{T, <:Any, 1}, I::Block{1}) where {T} = viewblock(a, I)

# Rank-0 (scalar) access: a rank-0 `GradedArray` (e.g. a full contraction to a scalar) holds one
# trivial-sector value in its 1×1 matricized block. Read it as a TensorKit scalar and write it into
# that block; the generic block path indexes the `TensorMap` by external sectors, of which a rank-0 array
# has none. Defined on the concrete type to take precedence over the `Vararg` block methods, which
# also match a no-argument call at N=0.
Base.getindex(a::GradedArray{<:Any, <:Any, 0}) = TK.scalar(to_tensormap(a))
function Base.setindex!(a::GradedArray{<:Any, <:Any, 0}, value)
    only(values(sectordata(matricize(a))))[begin] = value
    return a
end

# A `GradedArray` block is the same unique-fusion `UniqueSectorArray` the abelian backend returns.
# TODO: derive the block data type from the `FusedGradedMatrix` block type (its `D`) rather than
# hardcoding `Array{T, N}`, so non-`Array` storage (GPU, etc.) is preserved — e.g. via
# `Base.promote_op` on `view(A, ::Block)` (the actual returned block type). Also only well-defined
# for unique fusion (blocks are `UniqueSectorArray` only then); tie it to that guard.
function blocktype(::Type{<:GradedArray{T, S, N, NC, ND}}) where {T, S, N, NC, ND}
    return UniqueSectorArray{T, S, N, NC, ND, Array{T, N}}
end
blocktype(a::GradedArray) = blocktype(typeof(a))

# The block storage type is the datatype of the blocks, so only `blocktype` is type-specific.
datatype(::Type{T}) where {T <: GradedArray} = datatype(blocktype(T))
datatype(a::GradedArray) = datatype(typeof(a))
sectortype(::Type{<:GradedArray{T, S}}) where {T, S} = S

# ============================  block storage interface (unique fusion)  ============================
# Unique fusion only: the symmetry-allowed blocks, computed from the axes since a `GradedArray` holds
# the coupled matrix, not per-block data.
function eachblockstoredindex(a::GradedArray)
    require_unique_fusion(a)
    return allowedblocks(axes(a))
end

# View of the stored (symmetry-allowed) external blocks, the N-dim analog of `FusedGradedMatrixBlocks`.
# Implements GradedArrays' stored-entry interface; `blockstoredlength` and the array's scalar
# `getindex`/`setindex!` read through `isstored(blocks(a), …)`. A stored entry shares data via
# `viewblock`; an unstored entry is a symmetry-forbidden block and errors.
struct GradedArrayBlocks{T, S, N, A <: GradedArray{T, S, N}} <:
    AbstractArray{UniqueSectorArray{T, S, N}, N}
    parent::A
end
BlockArrays.blocks(a::GradedArray) = GradedArrayBlocks(a)
Base.size(b::GradedArrayBlocks) = blocklength.(axes(b.parent))
Base.getindex(b::GradedArrayBlocks, I::Int...) = getindex_sparse(b, I...)
Base.setindex!(b::GradedArrayBlocks, value, I::Int...) = setindex!_sparse(b, value, I...)

function eachstoredindex(::IndexCartesian, b::GradedArrayBlocks)
    return [CartesianIndex(Int.(Tuple(bI))) for bI in eachblockstoredindex(b.parent)]
end
function storedvalues(b::GradedArrayBlocks)
    return [view(b.parent, bI) for bI in eachblockstoredindex(b.parent)]
end
function isstored(
        b::GradedArrayBlocks{<:Any, <:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return Block(I...) in eachblockstoredindex(b.parent)
end
function getstoredindex(
        b::GradedArrayBlocks{<:Any, <:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return view(b.parent, Block(I...))
end
function setstoredindex!(
        b::GradedArrayBlocks{<:Any, <:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    copy_sector!(view(b.parent, Block(I...)), value)
    return b
end
function getunstoredindex(
        b::GradedArrayBlocks{<:Any, <:Any, N}, I::Vararg{Int, N}
    ) where {N}
    return error("Block $(I) is not stored.")
end
function setunstoredindex!(
        b::GradedArrayBlocks{<:Any, <:Any, N}, value, I::Vararg{Int, N}
    ) where {N}
    return error("Block $(I) is not stored.")
end

# ============================  similar  ============================
# `similar` must build a `GradedArray`. Without explicit axes, preserve the prototype's own
# codomain/domain split (like `copy`); with explicit flat axes the split is unrecoverable (see the
# `copy` note), so put all axes in the codomain, matching the broadcast `similar`.
function Base.similar(a::GradedArray, ::Type{T}) where {T}
    return GradedArray(similar(matricize(a), T), axes_codomain(a), axes_domain(a))
end
function Base.similar(
        a::GradedArray, ::Type{T},
        axes::Tuple{AbstractGradedOneTo{S}, Vararg{AbstractGradedOneTo{S}}}
    ) where {T, S}
    return TensorAlgebra.similar_map(a, T, axes, ())
end
# Empty axes build a rank-0 `GradedArray`; the sector type is read from the prototype (it cannot
# be inferred from the empty axes). Without this, `similar(fa, T, ())` falls through to Base and
# returns a plain rank-0 `Array`.
function Base.similar(a::GradedArray, ::Type{T}, ::Tuple{}) where {T}
    return GradedArray{T, sectortype(a)}(undef, (), ())
end

# ============================  copyto!  ============================
# Copy `src` into `dest` as an identity leg permutation into `dest`'s own codomain/domain split, so
# `bipermutedims!` bends `src` when the two splits differ and validates the external axes. The generic
# `AbstractArray` `copyto!` scalar-indexes, which errors on forbidden (symmetry-disallowed) blocks.
# `Base.copy!` rides on this: it checks `axes` equality and forwards here.
function Base.copyto!(dest::GradedArray, src::GradedArray)
    bipermutedims!(
        dest, src,
        ntuple(identity, Val(ndims_codomain(dest))),
        ntuple(i -> ndims_codomain(dest) + i, Val(ndims_domain(dest)))
    )
    return dest
end

# ============================  ==  ============================
# Compare the shared array contents, not the internal matricization: a `GradedArray` is an array
# whose `axes` are the flat external legs, so two with equal `axes` are equal iff their data matches.
# Rematricize `b` to `a`'s codomain/domain split, then compare coupled blocks (unlike a `TensorMap`,
# whose split is part of its identity). This also avoids Base's element-wise fallback, which would
# index forbidden blocks.
function Base.:(==)(a::GradedArray, b::GradedArray)
    axes(a) == axes(b) || return false
    return matricize(a) == matricize(b, Val(ndims_codomain(a)))
end

# ============================  dot  ============================
# Sum the coupled-block inner products of the (norm-preserving) matricized forms, rather than the
# block-walk that iterates `eachblockstoredindex` (unique-fusion-only for `GradedArray`)
# and scalar-indexes. `b` is rematricized to `a`'s split so the coupled blocks line up.
function LinearAlgebra.dot(a::GradedArray, b::GradedArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    ma = matricize(a)
    mb = matricize(b, Val(ndims_codomain(a)))
    init = zero(LinearAlgebra.dot(zero(eltype(a)), zero(eltype(b))))
    sda, sdb = sectordata(ma), sectordata(mb)
    return sum(keys(sda); init) do c
        return LinearAlgebra.dot(sda[c], sdb[c])
    end
end

# `LinearAlgebra.normalize` infers its result eltype via `typeof(first(a)/nrm)`, which scalar-indexes
# opaque block storage; route through the graded `/` instead.
function LinearAlgebra.normalize(a::GradedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# ============================  permutedims  ============================
# Route through `permutedimsopadd!` (which forwards to the `GradedArray` `bipermutedimsopadd!`), so
# the fermion braiding/bend signs are applied by TensorKit. Without this, Base's generic
# `permutedims` allocates via `similar` and scalar-permutes the data, dropping the sign.
function Base.permutedims(a::GradedArray{<:Any, <:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> axes(a)[perm[i]], Val(N))
    return permutedims!(similar(a, dest_axes), a, perm)
end
function Base.permutedims!(
        y::GradedArray{<:Any, <:Any, N}, x::GradedArray{<:Any, <:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ============================  conj  ============================
# `conj` on a graded array is charge conjugation (dualizes the axes, applies the fermion phase), so route
# through the broadcast path even for a real eltype, where Base's `conj` would short-circuit to the
# identity and drop it.
Base.conj(a::GradedArray) = Base.Broadcast.broadcast_preserving_zero_d(conj, a)

# ============================  matrix operations (guarded)  ============================
# Matrix / linear-algebra operations live only on the matrix storage type `FusedGradedMatrix`. On a
# `GradedArray` they would otherwise fall through to the generic `AbstractArray` methods, which
# scalar-index into a dense, non-graded result. Error instead; matricize to a `FusedGradedMatrix`
# first (`matricize(a)`).
Base.adjoint(A::GradedArray) = _matrix_op_error(adjoint, A)
Base.transpose(A::GradedArray) = _matrix_op_error(transpose, A)
Base.:*(A::GradedArray, B::GradedArray) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::GradedArray) = _matrix_op_error(LinearAlgebra.tr, A)
MAK.one!(A::GradedArray) = _matrix_op_error(MAK.one!, A)
# The matrix predicates (`FusedGradedMatrix` defines these) are matrix concepts too: on a `GradedArray`
# they would otherwise fall through to a dense elementwise scan.
LinearAlgebra.isdiag(A::GradedArray) = _matrix_op_error(LinearAlgebra.isdiag, A)
LinearAlgebra.istriu(A::GradedArray) = _matrix_op_error(LinearAlgebra.istriu, A)
LinearAlgebra.istril(A::GradedArray) = _matrix_op_error(LinearAlgebra.istril, A)
LinearAlgebra.isposdef(A::GradedArray) = _matrix_op_error(LinearAlgebra.isposdef, A)
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::GradedArray) = _matrix_op_error($f, A)
end

# ============================  TensorMap conversion  ============================

"""
    TK.TensorMap(fa::GradedArray)

Convert a `GradedArray` to a `TK.TensorMap`, building the codomain/domain product
spaces from the per-leg axes and copying each coupled-sector block.
"""
function TK.TensorMap(fa::GradedArray)
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
    GradedArray(t::TK.AbstractTensorMap)

Build a `GradedArray` from a `TensorMap`, taking the per-leg external axes from its codomain
and domain spaces. This copies the data; `to_gradedarray` is the zero-copy view counterpart.
"""
function GradedArray(t::TK.AbstractTensorMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(t)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(t)))
    return copy!(GradedArray{eltype(t)}(undef, axes_codomain, axes_domain), t)
end
# A plain `TensorMap` has a zero-copy view, so its copying constructor is `copy` of that view.
GradedArray(t::TK.TensorMap) = copy(to_gradedarray(t))

# Copy a matrix `TensorMap` (one codomain and one domain leg) block-wise into a `FusedGradedMatrix`.
function Base.copy!(m::FusedGradedMatrix, t::TK.AbstractTensorMap{<:Any, <:Any, 1, 1})
    msd = sectordata(m)
    for c in TK.blocksectors(t)
        copy!(msd[SectorRange(c)], TK.block(t, c))
    end
    return m
end

# Copy a `FusedGradedMatrix` block-wise into a matrix `TensorMap` (one codomain and one domain leg).
function Base.copy!(t::TK.AbstractTensorMap{<:Any, <:Any, 1, 1}, m::FusedGradedMatrix)
    for (c, b) in pairs(sectordata(m))
        copy!(TK.block(t, label(c)), b)
    end
    return t
end

# Copy a `TensorMap` into a `GradedArray` with matching codomain and domain (same coupled blocks).
function Base.copy!(a::GradedArray, t::TK.AbstractTensorMap)
    (TK.numout(t) == ndims_codomain(a) && TK.numin(t) == ndims_domain(a)) ||
        throw(DimensionMismatch("TensorMap codomain/domain does not match the GradedArray"))
    asd = sectordata(matricize(a))
    for c in TK.blocksectors(t)
        copy!(asd[SectorRange(c)], TK.block(t, c))
    end
    return a
end

# Copy a `GradedArray` into a `TensorMap` with matching codomain and domain (same coupled blocks).
function Base.copy!(t::TK.AbstractTensorMap, a::GradedArray)
    (TK.numout(t) == ndims_codomain(a) && TK.numin(t) == ndims_domain(a)) ||
        throw(DimensionMismatch("TensorMap codomain/domain does not match the GradedArray"))
    for (c, b) in pairs(sectordata(matricize(a)))
        copy!(TK.block(t, label(c)), b)
    end
    return t
end

# ============================  construction from axes  ============================

# Axes are given codomain-facing (un-dualized), the same convention they are stored in,
# matching `similar_map`/`unmatricize`. The sector type is read from the axes, so at least one leg
# must be present; the rank-0 (both-empty) case takes `S` explicitly through the method below.
function GradedArray{T}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    S = sectortype(first((axes_codomain..., axes_domain...)))
    return GradedArray{T, S}(undef, axes_codomain, axes_domain)
end

# Fuse each side's per-leg axes into its coupled `GradedOneTo` (through the GradedArrays interface,
# which uses the TensorKitSectors fusion rules), seeding empty groups with the trivial sector.
# `FusedGradedMatrix{T}(undef, …)` then allocates the reduced blocks. With `S` given as a type
# parameter this also covers rank-0: both groups fuse to the trivial sector, giving a 1×1 scalar.
function GradedArray{T, S}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T, S}
    init = trivial_gradedrange(S)
    coupled_codomain = reduce(tensor_product, axes_codomain; init)
    coupled_domain = reduce(tensor_product, axes_domain; init)
    m = FusedGradedMatrix{T}(undef, coupled_codomain, coupled_domain)
    return GradedArray(m, axes_codomain, axes_domain)
end

# A `GradedArray` source reproduces a `GradedArray`, routing straight to its `undef` constructor.
function TensorAlgebra.similar_map(
        ::GradedArray, ::Type{T},
        axes_codomain::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}},
        axes_domain::Tuple{Vararg{AbstractGradedOneTo}}
    ) where {T}
    return GradedArray{T}(undef, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        ::GradedArray, ::Type{T},
        axes_codomain::Tuple{},
        axes_domain::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
    ) where {T}
    return GradedArray{T}(undef, axes_codomain, axes_domain)
end
# Rank-0: the empty axes carry no sector type, so read it from the `GradedArray` prototype.
function TensorAlgebra.similar_map(
        fa::GradedArray, ::Type{T}, axes_codomain::Tuple{}, axes_domain::Tuple{}
    ) where {T}
    return GradedArray{T, sectortype(fa)}(undef, axes_codomain, axes_domain)
end

# Fill the reduced coupled-sector blocks in place, forwarding to the matricized `FusedGradedMatrix`
# (which fills via `eachblockstoredindex`). Construct with `GradedArray{T}(undef, …)` first.
for f! in (:rand!, :randn!)
    @eval begin
        Random.$f!(rng::AbstractRNG, fa::GradedArray) = (Random.$f!(rng, matricize(fa)); fa)
        Random.$f!(fa::GradedArray) = Random.$f!(Random.default_rng(), fa)
    end
end

# ============================  in-place primitives / algebra  ============================
# `zero!`/`scale!`/`norm`/`real`/`imag` are a block-walk over `eachblockstoredindex` on the fused
# arrays; `GradedArray`'s own `eachblockstoredindex` is unique-fusion-only, so forward to the
# matricized `FusedGradedMatrix` instead. `+`/`-` are left to the `AbstractArray` broadcast machinery.

TensorAlgebra.zero!(fa::GradedArray) = (zero!(matricize(fa)); fa)
TensorAlgebra.scale!(fa::GradedArray, α::Number) = (scale!(matricize(fa), α); fa)
function LinearAlgebra.rmul!(fa::GradedArray, α::Number)
    LinearAlgebra.rmul!(matricize(fa), α)
    return fa
end
function LinearAlgebra.lmul!(α::Number, fa::GradedArray)
    LinearAlgebra.lmul!(α, matricize(fa))
    return fa
end
LinearAlgebra.norm(fa::GradedArray, p::Real = 2) = LinearAlgebra.norm(matricize(fa), p)
Base.fill!(fa::GradedArray, v) = (fill!(matricize(fa), v); fa)
Base.iszero(fa::GradedArray) = iszero(matricize(fa))

# ============================  reductions  ============================
# `sum`/`maximum`/`minimum`/`extrema` are not defined on a `GradedArray`: the intended reduction is
# ambiguous. The reduction over the dense array `Array(a)` and the reduction over the fused matrix
# `matricize(a)` differ for non-abelian fusion (the Clebsch-Gordan recoupling mixes the dense entries),
# so require the caller to pick one explicitly rather than silently committing to a split-dependent
# densification. `FusedGradedMatrix` defines the fused-matrix reductions, and `Array` gives the dense
# ones.
@noinline function _reduction_error(op, ::GradedArray)
    return error(
        "`$op` is not defined for a `GradedArray`: reduce explicitly over `Array(a)` for the " *
            "dense array, or over `matricize(a)` for the fused matrix (these differ for non-abelian " *
            "fusion)."
    )
end
for op in (:sum, :maximum, :minimum, :extrema)
    @eval begin
        Base.$op(a::GradedArray; kwargs...) = _reduction_error($op, a)
        Base.$op(f, a::GradedArray; kwargs...) = _reduction_error($op, a)
    end
end

# Copy the matricized matrix and reuse the axes. Defined directly (rather than through `similar`)
# because a generic `similar` over flat axes cannot recover the
# codomain/domain split a `GradedArray` needs; `copy` must preserve it (e.g. `one!!(copy(A), …)`
# relies on the copy keeping `A`'s split so the identity fill lands in the stored matrix).
function Base.copy(fa::GradedArray)
    return GradedArray(copy(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end

# Sort each external axis by sector and merge repeated sectors. The internal fused storage is already
# canonical (blocks are keyed by coupled sector), so the data is unchanged: only the external axes are
# re-labeled to their merged-sorted form, which re-slices the same coupled blocks into the merged
# external blocks. Copies the matricized matrix so the result is an independent array.
function sectormergesort(a::GradedArray)
    return GradedArray(
        copy(matricize(a)),
        map(sectormergesort, axes_codomain(a)),
        map(sectormergesort, axes_domain(a))
    )
end

function Base.real(fa::GradedArray)
    return GradedArray(real(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end
function Base.imag(fa::GradedArray)
    return GradedArray(imag(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end

Base.:+(a::GradedArray, b::GradedArray) = a .+ b
Base.:-(a::GradedArray, b::GradedArray) = a .- b

# ============================  broadcasting  ============================
# Opt in to the graded linear-broadcast machinery so linear combinations (`a + b`, `2a - b`, …)
# materialize through `bipermutedimsopadd!`. The shared graded `copyto!` and the
# `LinearBroadcasted` fold do the work; only allocation is `GradedArray`-specific.

struct GradedStyle{N} <: AbstractGradedStyle{N} end
GradedStyle{N}(::Val{M}) where {N, M} = GradedStyle{M}()

function BC.BroadcastStyle(::Type{<:GradedArray{<:Any, <:Any, N}}) where {N}
    return GradedStyle{N}()
end

# A `GradedArray` and a bare fused graded array (`FusedGradedMatrix`/`Vector`/`Diagonal`) do not mix
# in a single broadcast expression: the coupled-block layout has no aligned counterpart in the
# external-axis array. Error deliberately rather than densify. Only one direction is needed:
# `result_style` evaluates `BroadcastStyle` in both operand orders, so this fires either way.
function BC.BroadcastStyle(::GradedStyle, ::AbstractFusedGradedStyle)
    return error("cannot broadcast a `GradedArray` together with a fused graded array")
end

# TODO: This picks the default block data type and so does not preserve non-`Array` block types
# (e.g. GPU arrays). Carry the block data type on `GradedStyle` and use it here, as BlockSparseArrays
# does for its broadcast style.
function Base.similar(bc::BC.Broadcasted{<:GradedStyle}, elt::Type)
    lb = flattenlinear(bc)
    bi = biaxes(lb)
    return TensorAlgebra.similar_map(broadcast_array(lb), elt, codomain(bi), domain(bi))
end

function Base.copyto!(dest::GradedArray, bc::BC.Broadcasted{<:GradedStyle})
    return copyto!(dest, flattenlinear(bc))
end

# ============================  TensorKit conversion (real, zero-copy `TensorMap`)  ============================
# `to_tensormap` views a `GradedArray` as a genuine `TensorMap` sharing the matricized contiguous buffer
# (the buffer is laid out in TensorKit's `.data` order), so TensorKit's concrete-type kernels (permute,
# `twist!`, projection) run natively and write back in place. `to_gradedarray` is the reverse view: a
# `GradedArray` sharing a `TensorMap`'s `.data`. The `GradedArray(t)` / `TK.TensorMap(a)` constructors
# are the copying counterparts.

# The fused-sorted coupled `HomSpace` for the given external axes. Each leg's space is the
# `sectormergesort` of its (possibly unfused/unsorted) axis; the rank-0 case resolves the trivial
# space from the sector type since it has no legs.
function tensormapspace(::Type{S}, axes_codomain::Tuple, axes_domain::Tuple) where {S}
    Sp = typeof(ElementarySpace(trivial_gradedrange(S)))
    codomain =
        mapreduce(ElementarySpace ∘ sectormergesort, TK.:⊗, axes_codomain; init = one(Sp))
    domain =
        mapreduce(ElementarySpace ∘ sectormergesort, TK.:⊗, axes_domain; init = one(Sp))
    return codomain ← domain
end

# `to_tensormap(m, axes_codomain, axes_domain)` is the storage-level conversion: a general matricized
# backing wraps as a `TensorMap` over its buffer; a diagonal factor wraps as a `DiagonalTensorMap` over
# its contiguous diagonal buffer, keeping the bond diagonal through the conversion (TensorKit's own
# `svd`/`eig` produce a `DiagonalTensorMap`). Both share storage, no copy.
to_tensormap(a::GradedArray) = to_tensormap(matricize(a), axes_codomain(a), axes_domain(a))
function to_tensormap(m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple)
    return TK.TensorMap(m.buffer, tensormapspace(sectortype(m), axes_codomain, axes_domain))
end
function to_tensormap(d::FusedGradedDiagonal, axes_codomain::Tuple, axes_domain::Tuple)
    return TK.DiagonalTensorMap(
        MAK.diagview(d).buffer, ElementarySpace(sectormergesort(only(axes_codomain)))
    )
end

# Zero-copy reverse of `to_tensormap`: share the `TensorMap`'s `.data` (already in the matricized
# buffer's contiguous layout) as a `FusedGradedMatrix`, then wrap it with the per-leg external axes.
# Mirrors the `undef` constructor but over borrowed storage; `GradedArray(t)` is the copying form.
function to_gradedarray(t::TK.TensorMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(t)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(t)))
    S = sectortype(first((axes_codomain..., axes_domain...)))
    init = trivial_gradedrange(S)
    coupled_codomain = reduce(tensor_product, axes_codomain; init)
    coupled_domain = reduce(tensor_product, axes_domain; init)
    m = FusedGradedMatrix(t.data, coupled_codomain, coupled_domain)
    return GradedArray(m, axes_codomain, axes_domain)
end

# ============================  bipermutedimsopadd! (permute primitive)  ============================
# `y = α * op.(permute(x, …)) + β * y`, delegated to the `AbstractTensorMap` `bipermutedimsopadd!`
# (fusion-tree recombination plus braiding/fermion signs). Wrapping `y`/`x` as `TensorMap`s shares
# their buffers, so TensorKit writes the result into `y` in place.

function TensorAlgebra.bipermutedimsopadd!(
        y::GradedArray, op, x::GradedArray,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    TensorAlgebra.bipermutedimsopadd!(
        to_tensormap(y), op, to_tensormap(x), perm_codomain, perm_domain, α, β
    )
    return y
end

# ============================  fermionic twist  ============================
# The contraction twist scales blocks by a per-fusion-tree fermion phase. Wrapping `a` as a
# The `TensorMap` shares its buffer, so `TK.twist!` scales it in place.
function twist!(a::GradedArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    TK.twist!(to_tensormap(a), dims)
    return a
end

# ============================  TensorAlgebra primitive interface  ============================
# `matricize` returns the stored `FusedGradedMatrix`, `unmatricize` puts axes back on. Because
# `FusedGradedMatrix` already has `mul!` and block-wise factorizations, contraction and
# factorizations ride the generic `TensorAlgebra` machinery with no high-level overloads.

TensorAlgebra.MatricizeStyle(::Type{<:GradedArray}) = GradedMatricize()

# When the requested split matches the stored split this is the stored matrix. Otherwise it is a
# leg bend (the `matricizeopperm` fast path only reaches here with an identity permutation, so
# legs stay in order and only the codomain/domain boundary moves), which for a `GradedArray` is
# not a free reshape. Re-split with the array's own `bipermutedims`, then take its 1-arg
# `matricize`. This is what lets a contraction over a subset of legs matricize a factor whose
# stored split differs.
function TensorAlgebra.matricize(
        ::GradedMatricize,
        fa::GradedArray,
        ::Val{K}
    ) where {K}
    K == ndims_codomain(fa) && return matricize(fa)
    N = ndims(fa)
    # TODO: Once `permutedims` on a `GradedArray` routes to `bipermutedimsopadd!`, bend with the
    # identity-permutation `permutedims` directly (ideally a `[bi]permutedims(fa, Val(K))` split-only
    # spelling). See the "Unified `permutedims` surface" follow-up.
    fa_bent = TensorAlgebra.bipermutedims(
        fa, ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K))
    )
    return matricize(fa_bent)
end

function TensorAlgebra.check_input(
        ::typeof(unmatricize), m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple
    )
    init = trivial_gradedrange(sectortype(m))
    (
        axis_codomain(m) == reduce(tensor_product, axes_codomain; init) &&
            axis_domain(m) == reduce(tensor_product, axes_domain; init)
    ) || throw(ArgumentError("axes do not fuse to the matrix's coupled axes"))
    return nothing
end

function TensorAlgebra.unmatricize(
        ::GradedMatricize, m::FusedGradedMatrix, axes_codomain::Tuple,
        axes_domain::Tuple
    )
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return GradedArray(m, axes_codomain, axes_domain)
end

# A `{1,1}` unmatricize (one codomain axis, one domain axis) reproduces the diagonal's own square
# bond and is the endomorphism identity: the result stays diagonal, wrapped up to the tensor-level
# `GradedArray`. `check_input` rejects a mismatched single-axis split rather than densifying it.
function TensorAlgebra.check_input(
        ::typeof(unmatricize), d::FusedGradedDiagonal, axes_codomain::Tuple, axes_domain::Tuple
    )
    (GA.axes_codomain(d) == axes_codomain && GA.axes_domain(d) == axes_domain) ||
        throw(
        ArgumentError(
            "a `FusedGradedDiagonal` unmatricizes only to its own square bond axis"
        )
    )
    return nothing
end

function TensorAlgebra.unmatricize(
        ::GradedMatricize, d::FusedGradedDiagonal,
        axes_codomain::Tuple{<:AbstractGradedOneTo}, axes_domain::Tuple{<:AbstractGradedOneTo}
    )
    check_input(unmatricize, d, axes_codomain, axes_domain)
    return GradedArray(d, axes_codomain, axes_domain)
end

# Any other split is a genuine bond-split (for example a rank-4 generalized-diagonal tensor), not
# representable as a `FusedGradedDiagonal`, so densify to a `FusedGradedMatrix` and reconstruct
# through its own `unmatricize`.
function TensorAlgebra.unmatricize(
        style::GradedMatricize, d::FusedGradedDiagonal, axes_codomain::Tuple, axes_domain::Tuple
    )
    return unmatricize(style, FusedGradedMatrix(d), axes_codomain, axes_domain)
end

# ============================  contraction  ============================

# A general graded right factor is twisted; the per-position `TwistedGradedMatricize` can only come
# from an explicit override. A matrix-level right factor needs no twist and falls out of the default
# `default_contract_algorithm`: both operands share `GradedMatricize`, which the default combinator
# maps to itself.
for A in (:GradedArray, :AbstractFusedGradedArray)
    @eval function TensorAlgebra.default_contract_algorithm(
            ::Type{<:$A},
            ::Type{<:GradedArray}
        )
        return TensorAlgebra.Matricize(
            GradedMatricize(), TwistedGradedMatricize(), GradedMatricize()
        )
    end
end

function TensorAlgebra.matricize(
        ::GradedMatricize, m::AbstractFusedGradedMatrix, ::Val{K}
    ) where {K}
    K == 1 || throw(
        ArgumentError(
            "a matrix-level fused array matricizes only with a single codomain leg"
        )
    )
    return m
end

function TensorAlgebra.unmatricizeperm!(
        ::GradedMatricize, a_dest::GradedArray{<:Any, <:Any, N},
        m::AbstractFusedGradedMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    ) where {N}
    # Permute `a_dest` into the matricized leg order to get the matricized-order axes with correct
    # per-leg duality; the permuted data is discarded. Wrap `m` in those axes, then permute back.
    # TODO: Switch to `permutedims` once it routes through `bipermutedimsopadd!` (see the "Unified
    # `permutedims` surface" follow-up), as in the `matricize` leg-bend above.
    template = TensorAlgebra.bipermutedims(a_dest, invperm_codomain, invperm_domain)
    tmp = GradedArray(m, axes_codomain(template), axes_domain(template))
    perm_dest = invperm((invperm_codomain..., invperm_domain...))
    ndims_cod_dest = ndims_codomain(a_dest)
    perm_codomain = ntuple(i -> perm_dest[i], Val(ndims_cod_dest))
    perm_domain = ntuple(i -> perm_dest[ndims_cod_dest + i], Val(N - ndims_cod_dest))
    # TODO: Switch to `Base.permutedims!` once it is defined to route through `bipermutedimsopadd!`.
    bipermutedims!(a_dest, tmp, perm_codomain, perm_domain)
    return a_dest
end

# ============================  concatenation  ============================
# Place whole symmetry-allowed blocks (no scalar indexing) via `concatenate_sparse!` on the block
# containers. When the containers subtyped `AbstractSparseArray` this went through the generic
# `TensorAlgebra.concatenate!` (which slices the destination); `concatenate_sparse!` is the
# whole-block stand-in that needs only the stored-entry interface. The block views are guarded, so
# opt into block indexing for the placement.
#
# Abelian-only stand-in: `cat` / `directsum` will be reimplemented for non-abelian fusion, superseding
# this path.
function TensorAlgebra.concatenate!(dest::GradedArray, dims, args...)
    with_block_indexing() do
        return concatenate_sparse!(blocks(dest), dims, blocks.(args)...)
    end
    return dest
end
# Route `Base.cat` through the same machinery so it uses the graded destination and placement.
Base._cat(dims, as::GradedArray...) = TensorAlgebra.concatenate(dims, as...)

# ============================  project (dense -> symmetric)  ============================
# The dense-to-symmetric projection for the `GradedArray` backend is delegated to TensorKit in the
# shared `unchecked_project_graded` worker (`gradedconstructors.jl`): it projects over the equivalent
# `ElementarySpace`s and wraps the resulting `TensorMap` with `GradedArray(t)`. `unproject` below is
# the dense inverse used by `TA.project`'s verification.

# In-place projection into a preallocated destination: view `dest` as a `TensorMap` (sharing its
# buffer) and project the dense `src` straight into those blocks through TensorKit, which drops the
# forbidden regions and handles a lower-rank `src` reshaped into a flux-canceling aux leg. The generic
# `AbstractArray` `projectto!` would scalar-`copyto!` and error mid-write on a forbidden block.
function TensorAlgebra.projectto!(dest::GradedArray, src::AbstractArray)
    TensorAlgebra.projectto!(to_tensormap(dest), src)
    return dest
end

# Dense form. The matricized backing has no `eachblockstoredindex` (the generic dense path), so
# materialize through the zero-copy `TensorMap` view. The view is fused-sorted per leg, so for an
# unfused/unsorted stored axis, move each leg's sector blocks back to the stored order (whole-block
# moves preserve the array type, e.g. GPU).
# TensorKit defines only the untyped `convert(::Type{Array}, t)` (by its own account a checking
# path, not tuned for speed), and it widens the element type by the sector scalar type, so a
# fusion-coefficient-carrying result converts back to `T` in a second pass (an `InexactError` for
# an integer `T` over a non-abelian sector, an exotic combination we accept losing).
function Base.Array{T, N}(fa::GradedArray{<:Any, <:Any, N}) where {T, N}
    dense = convert(Array{T, N}, convert(Array, to_tensormap(fa)))
    all(is_fused_sorted, axes(fa)) && return dense
    sortedlengths = map(g -> Vector(blocklengths(g))[sortperm(sectors(g))], axes(fa))
    invperms = ntuple(d -> Block.(invperm(sortperm(sectors(axes(fa)[d])))), Val(N))
    return parent(BlockedArray(dense, sortedlengths...)[invperms...])
end
# Rank-0: TensorKit's `convert(Array, ::rank-0 TensorMap)` hits a VectorInterface `add!` gap for some
# eltypes (e.g. `Float32`), so build the 0-dim array from the scalar directly.
Base.Array{T, 0}(fa::GradedArray{<:Any, <:Any, 0}) where {T} = fill(convert(T, fa[]))

# `unproject` is the dense inverse of `projectto!` used by `TA.project`'s verification, and the dense
# form at an arbitrary codomain split `Val{K}` used to compare tensors across backends (which agree
# only at a common split). When `K` is the array's own split the `TensorMap` conversion undoes the
# domain-leg bend directly; otherwise bend the legs to a `K`-codomain split first. The bend carries
# the fermion signs.
function TensorAlgebra.unproject(fa::GradedArray, ::Val{K}) where {K}
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

function Base.summary(io::IO, fa::GradedArray)
    print(
        io, Base.dims2string(size(fa)), " ", nameof(typeof(fa)),
        " (codomain ", ndims_codomain(fa), ", domain ", ndims_domain(fa), ")"
    )
    return nothing
end

Base.show(io::IO, fa::GradedArray) = summary(io, fa)

function Base.show(io::IO, ::MIME"text/plain", fa::GradedArray)
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
