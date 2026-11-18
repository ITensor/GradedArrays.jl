# =============================================================================
#  GradedArray
# =============================================================================

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using TensorKit: TensorKit as TK, ←

const GA = GradedArrays

"""
    GradedArray

An array over graded axes, storing only the symmetry-allowed blocks. Its legs are split into a
codomain and a domain group, so it can also be read as a map between spaces.
"""
struct GradedArray{
        T, S, N, NC, ND, M <: AbstractFusedGradedMatrix{T, S}, I <: TKS.Sector,
    } <: AbstractArray{T, N}
    matricized::M
    axes_codomain::NTuple{NC, GradedOneTo{S, I}}
    axes_domain::NTuple{ND, GradedOneTo{S, I}}

    function GradedArray(
            matricized::AbstractFusedGradedMatrix{T, S},
            axes_codomain::NTuple{NC, AbstractGradedOneTo{S}},
            axes_domain::NTuple{ND, AbstractGradedOneTo{S}}
        ) where {T, S, NC, ND}
        return new{T, S, NC + ND, NC, ND, typeof(matricized), tensorkit_sectortype(S)}(
            matricized, map(GradedOneTo, axes_codomain), map(GradedOneTo, axes_domain)
        )
    end
end

# Wrap a fused graded matrix as a rank-2 `GradedArray`, sharing storage.
GradedArray(m::AbstractFusedGradedMatrix) = GradedArray(m, axes_codomain(m), axes_domain(m))

# ============================  Accessors  ============================

# Domain axes are stored un-dualized, like TensorKit's `domain`. `axes` dualizes them, like
# TensorKit's `space(t, i)`.
axes_codomain(fa::GradedArray) = fa.axes_codomain
axes_domain(fa::GradedArray) = fa.axes_domain

# Kept out of line so it is not compiled into every caller.
@noinline Base.axes(fa::GradedArray) = Tuple(biaxes(fa))
# Base's fallback returns either the axis or a trailing `OneTo(1)`, a union that does not infer.
function Base.axes(fa::GradedArray, d::Integer)
    nc = ndims_codomain(fa)
    return d ≤ nc ? axes_codomain(fa)[d] : conj(axes_domain(fa)[d - nc])
end
Base.size(fa::GradedArray) = map(length, axes(fa))

# Like `TensorKit.HomSpace`, but over graded axes or sectors. The domain is given un-dualized and
# stored dualized.
bispace(codomain, domain) = BiTuple(codomain, map(conj, domain))
codomain(bt::BiTuple) = bt.t1
domain(bt::BiTuple) = map(conj, bt.t2)

TensorAlgebra.ndims_codomain(fa::GradedArray) = length(axes_codomain(fa))
TensorAlgebra.ndims_domain(fa::GradedArray) = length(axes_domain(fa))
TensorAlgebra.hasbipartition(::GradedArray) = true

TensorAlgebra.matricize(fa::GradedArray) = fa.matricized

Base.dataids(fa::GradedArray) = Base.dataids(matricize(fa))
# Unwrapping `parent` repeatedly reaches the flat storage buffer, which `TensorAlgebra.data` uses.
Base.parent(fa::GradedArray) = matricize(fa)

# ============================  block indexing (unique fusion)  ============================
# Unique fusion only: for a non-abelian symmetry the leg sectors of a `Block` don't determine the
# block.

function viewblock(
        a::GradedArray{T, S, N, NC, ND},
        I::Block{N}
    ) where {T, S, N, NC, ND}
    assert_block_indexing()
    require_unique_fusion(a)
    bk = Int.(Tuple(I))
    sects = ntuple(d -> eachstructureaxis(axes(a, d))[bk[d]], Val(N))
    # The block type stores its domain legs un-dualized, as `axes_domain(a)` does.
    cod = ntuple(i -> sects[i], Val(NC))
    dom = ntuple(j -> eachstructureaxis(axes_domain(a)[j])[bk[NC + j]], Val(ND))
    # Conjugate each dual leg's sector to match TensorKit's external-sector indexing.
    blockdata = to_tensormap(a)[map(r -> TKS.Sector(flip_dual(r)), sects)]
    # An axis with a repeated sector stores its blocks for that sector merged into one, so slice
    # each leg to this block's subrange of the merged block.
    all(isfusedsorted, axes(a)) && return UniqueSectorArray(blockdata, cod, dom)
    ranges = ntuple(Val(N)) do d
        g = axes(a, d)
        return only(
            invblockmergeperm(g, sectorsortperm(g), fusesectors(g))[bk[d]].indices
        )
    end
    return UniqueSectorArray(view(blockdata, ranges...), cod, dom)
end

Base.view(a::GradedArray{T, <:Any, N}, I::Block{N}) where {T, N} = viewblock(a, I)
# Disambiguate the N=1 case against the `Vararg{Block{1}, N}` method.
Base.view(a::GradedArray{T, <:Any, 1}, I::Block{1}) where {T} = viewblock(a, I)

# Defined on the concrete type to take precedence over the `Vararg` block methods, which also match
# a no-argument call at N=0.
Base.getindex(a::GradedArray{<:Any, <:Any, 0}) = TK.scalar(to_tensormap(a))
function Base.setindex!(a::GradedArray{<:Any, <:Any, 0}, value)
    only(values(sectordata(matricize(a))))[begin] = value
    return a
end

# TODO: derive the block data type from the `FusedGradedMatrix` block type (its `D`) rather than
# hardcoding `Array{T, N}`, so non-`Array` storage (GPU, etc.) is preserved — e.g. via
# `Base.promote_op` on `view(A, ::Block)` (the actual returned block type). Also only well-defined
# for unique fusion (blocks are `UniqueSectorArray` only then); tie it to that guard.
function blocktype(::Type{<:GradedArray{T, S, N, NC, ND}}) where {T, S, N, NC, ND}
    return UniqueSectorArray{T, S, N, NC, ND, Array{T, N}}
end
blocktype(a::GradedArray) = blocktype(typeof(a))

datatype(::Type{T}) where {T <: GradedArray} = datatype(blocktype(T))
datatype(a::GradedArray) = datatype(typeof(a))
sectortype(::Type{<:GradedArray{T, S}}) where {T, S} = S

# ============================  block storage interface (unique fusion)  ============================
# Unique fusion only. Computed from the axes, since the storage is per coupled sector.
function eachblockstoredindex(a::GradedArray)
    require_unique_fusion(a)
    return allowedblocks(axes(a))
end

# A sparse array over the symmetry-allowed blocks. Accessing a symmetry-forbidden block errors.
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
# With explicit flat axes the codomain/domain split is unknown, so all axes go in the codomain.
function Base.similar(a::GradedArray, ::Type{T}) where {T}
    return GradedArray(similar(matricize(a), T), axes_codomain(a), axes_domain(a))
end
function Base.similar(
        a::GradedArray, ::Type{T},
        axes::Tuple{AbstractGradedOneTo{S}, Vararg{AbstractGradedOneTo{S}}}
    ) where {T, S}
    return TensorAlgebra.similar_map(a, T, axes, ())
end
# Otherwise Base returns a rank-0 `Array`.
function Base.similar(a::GradedArray, ::Type{T}, ::Tuple{}) where {T}
    return GradedArray{T, sectortype(a)}(undef, (), ())
end

# ============================  copyto!  ============================
# `bipermutedims!` bends `src` when its codomain/domain split differs from that of `dest`.
function Base.copyto!(dest::GradedArray, src::GradedArray)
    bipermutedims!(
        dest, src,
        ntuple(identity, Val(ndims_codomain(dest))),
        ntuple(i -> ndims_codomain(dest) + i, Val(ndims_domain(dest)))
    )
    return dest
end

# ============================  ==  ============================
# Unlike for a `TensorMap`, the codomain/domain split is not part of the identity, so `b` is
# rematricized to the split of `a`.
function Base.:(==)(a::GradedArray, b::GradedArray)
    axes(a) == axes(b) || return false
    return matricize(a) == matricize(b, Val(ndims_codomain(a)))
end

# ============================  dot  ============================
# `b` is rematricized to `a`'s split so the coupled blocks line up.
function LinearAlgebra.dot(a::GradedArray, b::GradedArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    return LinearAlgebra.dot(matricize(a), matricize(b, Val(ndims_codomain(a))))
end

# `LinearAlgebra.normalize` scalar-indexes to infer its result eltype.
function LinearAlgebra.normalize(a::GradedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# ============================  permutedims  ============================
# Base's generic `permutedims` would drop the fermion signs.
function Base.permutedims(a::GradedArray{<:Any, <:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> axes(a)[perm[i]], Val(N))
    return permutedims!(similar(a, dest_axes), a, perm)
end
function Base.permutedims!(
        y::GradedArray{<:Any, <:Any, N}, x::GradedArray{<:Any, <:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsconjadd!(y, false, x, perm, true, false)
    return y
end

# ============================  conj  ============================
# `conj` dualizes the axes and applies the fermion phase, so it is not the identity for a real
# eltype as Base's `conj` would have it.
Base.conj(a::GradedArray) = Base.Broadcast.broadcast_preserving_zero_d(conj, a)

# ============================  matrix operations (guarded)  ============================
# These would fall through to the generic `AbstractArray` methods, which scalar-index into a dense
# result.
Base.adjoint(A::GradedArray) = _matrix_op_error(adjoint, A)
Base.transpose(A::GradedArray) = _matrix_op_error(transpose, A)
Base.:*(A::GradedArray, B::GradedArray) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::GradedArray) = _matrix_op_error(LinearAlgebra.tr, A)
MAK.one!(A::GradedArray) = _matrix_op_error(MAK.one!, A)
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

Convert a `GradedArray` to a `TK.TensorMap` with the same codomain and domain. This copies the
data.
"""
function TK.TensorMap(fa::GradedArray)
    # From the sector type rather than a leg, so that a rank-0 array with no legs works too.
    Sp = typeof(ElementarySpace(trivial_gradedrange(sectortype(fa))))
    codsp = mapreduce(
        ElementarySpace ∘ fusesectors,
        TK.:⊗,
        axes_codomain(fa);
        init = one(Sp)
    )
    domsp =
        mapreduce(ElementarySpace ∘ fusesectors, TK.:⊗, axes_domain(fa); init = one(Sp))
    return copy!(TK.TensorMap{eltype(fa)}(undef, codsp, domsp), fa)
end

"""
    GradedArray(t::TK.AbstractTensorMap)

Build a `GradedArray` from a `TensorMap`, taking the per-leg external axes from its codomain
and domain spaces. This copies the data.
"""
function GradedArray(t::TK.AbstractTensorMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(t)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(t)))
    return copy!(GradedArray{eltype(t)}(undef, axes_codomain, axes_domain), t)
end
GradedArray(t::TK.TensorMap) = copy(to_gradedarray(t))

function Base.copy!(m::FusedGradedMatrix, t::TK.AbstractTensorMap{<:Any, <:Any, 1, 1})
    msd = sectordata(m)
    for c in TK.blocksectors(t)
        copy!(msd[Sector(c)], TK.block(t, c))
    end
    return m
end

function Base.copy!(t::TK.AbstractTensorMap{<:Any, <:Any, 1, 1}, m::FusedGradedMatrix)
    for (c, b) in pairs(sectordata(m))
        copy!(TK.block(t, TKS.Sector(c)), b)
    end
    return t
end

function Base.copy!(a::GradedArray, t::TK.AbstractTensorMap)
    (TK.numout(t) == ndims_codomain(a) && TK.numin(t) == ndims_domain(a)) ||
        throw(DimensionMismatch("TensorMap codomain/domain does not match the GradedArray"))
    asd = sectordata(matricize(a))
    for c in TK.blocksectors(t)
        copy!(asd[Sector(c)], TK.block(t, c))
    end
    return a
end

function Base.copy!(t::TK.AbstractTensorMap, a::GradedArray)
    (TK.numout(t) == ndims_codomain(a) && TK.numin(t) == ndims_domain(a)) ||
        throw(DimensionMismatch("TensorMap codomain/domain does not match the GradedArray"))
    for (c, b) in pairs(sectordata(matricize(a)))
        copy!(TK.block(t, TKS.Sector(c)), b)
    end
    return t
end

# ============================  construction from axes  ============================

# The domain axes are given un-dualized. A rank-0 array needs `S` given explicitly.
function GradedArray{T}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    S = sectortype(first((axes_codomain..., axes_domain...)))
    return GradedArray{T, S}(undef, axes_codomain, axes_domain)
end

function GradedArray{T, S}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T, S}
    m = FusedGradedMatrix{T}(undef, fuseaxes(S, axes_codomain), fuseaxes(S, axes_domain))
    return GradedArray(m, axes_codomain, axes_domain)
end

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
# The empty axes carry no sector type.
function TensorAlgebra.similar_map(
        fa::GradedArray, ::Type{T}, axes_codomain::Tuple{}, axes_domain::Tuple{}
    ) where {T}
    return GradedArray{T, sectortype(fa)}(undef, axes_codomain, axes_domain)
end

for f! in (:rand!, :randn!)
    @eval begin
        Random.$f!(rng::AbstractRNG, fa::GradedArray) = (Random.$f!(rng, matricize(fa)); fa)
        Random.$f!(fa::GradedArray) = Random.$f!(Random.default_rng(), fa)
    end
end

# ============================  in-place primitives / algebra  ============================

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
# Ambiguous, since reducing over `Array(a)` and over `matricize(a)` differ for non-abelian fusion.
function Base.mapreduce(f, op, a::GradedArray; kwargs...)
    return error(
        "Elementwise reductions are not defined for a `GradedArray`: reduce explicitly over " *
            "`Array(a)` for the dense array, or over `matricize(a)` for the fused matrix (these " *
            "differ for non-abelian fusion)."
    )
end

# Not through `similar` with flat axes, which would lose the codomain/domain split.
function Base.copy(fa::GradedArray)
    return GradedArray(copy(matricize(fa)), axes_codomain(fa), axes_domain(fa))
end

# Sort each axis by sector and merge repeated sectors. The stored data is unchanged, since it is
# keyed by coupled sector.
function fusesectors(a::GradedArray)
    return GradedArray(
        copy(matricize(a)),
        map(fusesectors, axes_codomain(a)),
        map(fusesectors, axes_domain(a))
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

struct GradedStyle{N} <: AbstractGradedStyle{N} end
GradedStyle{N}(::Val{M}) where {N, M} = GradedStyle{M}()

function BC.BroadcastStyle(::Type{<:GradedArray{<:Any, <:Any, N}}) where {N}
    return GradedStyle{N}()
end

# Only one operand order is needed, since `result_style` tries both.
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

# ============================  TensorKit conversion (zero-copy)  ============================
# `to_tensormap` and `to_gradedarray` convert between a `GradedArray` and a `TensorMap` that share
# storage.

function tensormapspace(::Type{S}, axes_codomain::Tuple, axes_domain::Tuple) where {S}
    Sp = Vect[tensorkit_sectortype(S)]
    codomain =
        mapreduce(ElementarySpace ∘ fusesectors, TK.:⊗, axes_codomain; init = one(Sp))
    domain =
        mapreduce(ElementarySpace ∘ fusesectors, TK.:⊗, axes_domain; init = one(Sp))
    return codomain ← domain
end

to_tensormap(a::GradedArray) = to_tensormap(matricize(a), axes_codomain(a), axes_domain(a))
function to_tensormap(m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple)
    # The inner constructor, since inferring the outer ones' dense-array branch is slow to compile.
    V = tensormapspace(sectortype(m), axes_codomain, axes_domain)
    TT = TK.tensormaptype(
        TK.spacetype(V), length(axes_codomain), length(axes_domain), typeof(m.buffer)
    )
    return TT(m.buffer, V)
end
function to_tensormap(d::FusedGradedDiagonal, axes_codomain::Tuple, axes_domain::Tuple)
    return TK.DiagonalTensorMap(
        MAK.diagview(d).buffer, ElementarySpace(fusesectors(only(axes_codomain)))
    )
end
# `adjoint` swaps codomain and domain back.
function to_tensormap(m::AdjointFusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple)
    return adjoint(to_tensormap(parent(m), axes_domain, axes_codomain))
end

function to_gradedarray(t::TK.TensorMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(t)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(t)))
    S = sectortype(first((axes_codomain..., axes_domain...)))
    m = FusedGradedMatrix(t.data, fuseaxes(S, axes_codomain), fuseaxes(S, axes_domain))
    return GradedArray(m, axes_codomain, axes_domain)
end

# ===========================  bipermutedimsconjadd! (permute primitive)  ===========================

function TensorAlgebra.bipermutedimsconjadd!(
        y::GradedArray, isconj::Bool, x::GradedArray,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    # Dispatched at run time, so TensorKit's permutation only compiles when a call actually
    # permutes, and not for every caller that has a permuting branch.
    Base.inferencebarrier(TensorAlgebra.bipermutedimsconjadd!)(
        to_tensormap(y), isconj, to_tensormap(x), perm_codomain, perm_domain, α, β
    )
    return y
end

# ============================  fermionic twist  ============================
function TA.twist!(a::GradedArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    _twist!(to_tensormap(a), dims)
    return a
end

# `TK.twist!` without its argument checks, and with the subblocks read by position: `TK.twist!`
# indexes `t[f₁, f₂]` once per fusion tree pair, which rebuilds the subblock structure of the whole
# space each time.
function _twist!(t::TK.TensorMap, dims)
    TK.has_shared_twist(t, dims) && return t
    N₁ = TK.numout(t)
    # By position: `StridedSubblocks` is in the order of `fusiontrees`. Iterating `TK.subblocks`
    # instead returns each element boxed, since its `iterate` is not inlined.
    blocks = TK.StridedSubblocks(t)
    for (i, (f₁, f₂)) in enumerate(TK.fusiontrees(t))
        θ = prod(
            k -> k <= N₁ ? TKS.twist(f₁.uncoupled[k]) : TKS.twist(f₂.uncoupled[k - N₁]),
            dims
        )
        isone(θ) || TK.scale!(@inbounds(blocks[i]), θ)
    end
    return t
end

# ============================  TensorAlgebra primitive interface  ============================

# The view is the stored matrix at the stored split, or its adjoint at the swapped split when
# conjugating. A transpose would reverse the arrows, so it is not a view.
function TensorAlgebra.isoutputview(
        ::typeof(TensorAlgebra.matricizeconj), isconj::Bool, a::GradedArray,
        perm_codomain, perm_domain
    )
    return !isconj && length(perm_codomain) == ndims_codomain(a) &&
        TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain) ||
        isconj && length(perm_domain) == ndims_codomain(a) &&
        TensorAlgebra.isidentitybiperm(perm_domain, perm_codomain)
end
function TensorAlgebra.matricizeconjview(
        isconj::Bool, fa::GradedArray, perm_codomain, perm_domain
    )
    return isconj ? adjoint(matricize(fa)) : matricize(fa)
end
function TensorAlgebra.matricizeconjcopy(
        isconj::Bool, fa::GradedArray, perm_codomain, perm_domain
    )
    fa_perm = TensorAlgebra.permutedimsconj(isconj, fa, perm_codomain, perm_domain)
    return matricize(fa_perm)
end

function TensorAlgebra.check_input(
        ::typeof(unmatricize), m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple
    )
    S = sectortype(m)
    (
        axis_codomain(m) == fuseaxes(S, axes_codomain) &&
            axis_domain(m) == fuseaxes(S, axes_domain)
    ) || throw(ArgumentError("axes do not fuse to the matrix's coupled axes"))
    return nothing
end

function TensorAlgebra.unmatricize(
        m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple
    )
    check_input(unmatricize, m, axes_codomain, axes_domain)
    return GradedArray(m, axes_codomain, axes_domain)
end

# Unmatricizing to the diagonal's own axes keeps it diagonal.
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
        d::FusedGradedDiagonal,
        axes_codomain::Tuple{<:AbstractGradedOneTo}, axes_domain::Tuple{<:AbstractGradedOneTo}
    )
    check_input(unmatricize, d, axes_codomain, axes_domain)
    return GradedArray(d, axes_codomain, axes_domain)
end

# Any other split is not representable as a `FusedGradedDiagonal`.
function TensorAlgebra.unmatricize(
        d::FusedGradedDiagonal, axes_codomain::Tuple, axes_domain::Tuple
    )
    return unmatricize(FusedGradedMatrix(d), axes_codomain, axes_domain)
end

# ============================  contraction  ============================

to_gradedarray(a::GradedArray) = a
to_gradedarray(m::AbstractFusedGradedMatrix) = GradedArray(m)

# ========================  graded contraction  ========================
# Chooses among the same four layouts as TensorAlgebra's matricized contraction. For fermionic
# sectors, the contracted legs of the right factor are twisted before matricizing, so the result
# does not depend on the contraction order.
function TensorAlgebra.contractpermconjadd!(
        ::TensorAlgebra.MatricizeContract,
        a_dest::GradedArray, biperm_dest_codomain, biperm_dest_domain,
        isconj1::Bool, a1::Union{GradedArray, AbstractFusedGradedMatrix}, biperm1_codomain,
        biperm1_domain,
        isconj2::Bool, a2::Union{GradedArray, AbstractFusedGradedMatrix}, biperm2_codomain,
        biperm2_domain,
        α::Number, β::Number
    )
    a1, a2 = to_gradedarray(a1), to_gradedarray(a2)
    biperm_dest = (biperm_dest_codomain..., biperm_dest_domain...)
    TupleTools.isperm(biperm_dest) ||
        throw(ArgumentError("destination permutation is not a permutation"))
    invperm_codomain, invperm_domain = TensorAlgebra.bipartition(
        TupleTools.invperm(biperm_dest), Val(length(biperm1_codomain))
    )
    check_input(
        TensorAlgebra.contractpermconjadd!,
        a_dest, invperm_codomain, invperm_domain,
        isconj1, a1, biperm1_codomain, biperm1_domain,
        isconj2, a2, biperm2_codomain, biperm2_domain
    )
    # The contracted legs reordered to `a2`'s storage order.
    p = TupleTools.sortperm(biperm2_codomain)
    biperm1_domain_alt = map(i -> biperm1_domain[i], p)
    biperm2_codomain_alt = map(i -> biperm2_codomain[i], p)
    l1 = matricize_copy_lengths(
        isconj1, a1, biperm1_codomain, biperm1_domain, biperm1_domain_alt
    )
    l2 = matricize_copy_lengths(
        isconj2, a2, biperm2_domain, biperm2_codomain, biperm2_codomain_alt
    )
    l_dest, l_dest_swapped = dest_copy_lengths(a_dest, invperm_codomain, invperm_domain)
    costs = (
        l_dest + l1[1] + l2[3],
        l_dest + l1[2] + l2[4],
        l_dest_swapped + l2[1] + l1[3],
        l_dest_swapped + l2[2] + l1[4],
    )
    i = argmin(costs)
    biperm1_contracted, biperm2_contracted = if isodd(i)
        (biperm1_domain, biperm2_codomain)
    else
        (biperm1_domain_alt, biperm2_codomain_alt)
    end
    if i ≤ 2
        _contractpermconjadd_graded!(
            a_dest, invperm_codomain, invperm_domain,
            isconj1, a1, biperm1_codomain, biperm1_contracted,
            isconj2, a2, biperm2_contracted, biperm2_domain,
            α, β
        )
    else
        _contractpermconjadd_graded!(
            a_dest, invperm_domain, invperm_codomain,
            isconj2, a2, biperm2_domain, biperm2_contracted,
            isconj1, a1, biperm1_contracted, biperm1_codomain,
            α, β
        )
    end
    return a_dest
end

# The number of elements copied to matricize `a` as the left factor `(open | contracted)` and as
# the right factor `(contracted | open)`, for both orders of the contracted legs.
@noinline function matricize_copy_lengths(
        isconj::Bool, a, perm_open, perm_contracted, perm_contracted_alt
    )
    n = length(a)
    copy_length(perm_codomain, perm_domain) =
    if TensorAlgebra.isoutputview(
            TensorAlgebra.matricizeconj, isconj, a, perm_codomain, perm_domain
        )
        0
    else
        n
    end
    return (
        copy_length(perm_open, perm_contracted),
        copy_length(perm_open, perm_contracted_alt),
        copy_length(perm_contracted, perm_open),
        copy_length(perm_contracted_alt, perm_open),
    )
end

# The number of elements copied to write the destination through its direct and its swapped split.
@noinline function dest_copy_lengths(a_dest, perm_codomain, perm_domain)
    n = length(a_dest)
    function copy_length(perm_codomain, perm_domain)
        TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain) &&
            TensorAlgebra.isoutputview(
            TensorAlgebra.matricizeconj, false, a_dest, perm_codomain, perm_domain
        ) &&
            return 0
        return n
    end
    return copy_length(perm_codomain, perm_domain), copy_length(perm_domain, perm_codomain)
end

# One storage layout of the graded contraction, with `a1` and `a2` the left and right factor.
function _contractpermconjadd_graded!(
        a_dest, perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1, perm1_codomain, perm1_domain,
        isconj2::Bool, a2, perm2_codomain, perm2_domain,
        α, β
    )
    # The twist goes on the factor TensorKit's `blas_contract!` would pick: the one that has to be
    # copied anyway if only one does, otherwise the smaller, so it never forces an extra copy.
    twist1 = twist2 = false
    if needs_contraction_twist(isconj2, a2, perm2_codomain)
        copy1 =
            !TensorAlgebra.isoutputview(
            TensorAlgebra.matricizeconj, isconj1, a1, perm1_codomain, perm1_domain
        )
        copy2 =
            !TensorAlgebra.isoutputview(
            TensorAlgebra.matricizeconj, isconj2, a2, perm2_codomain, perm2_domain
        )
        twist1 = copy1 ⊻ copy2 ? copy1 : length(a1) < length(a2)
        twist2 = !twist1
    end
    a1_mat = if twist1
        twisted_matricizeconj(isconj1, a1, perm1_codomain, perm1_domain, :domain)
    else
        @noinline TensorAlgebra.matricizeconj(isconj1, a1, perm1_codomain, perm1_domain)
    end
    a2_mat = if twist2
        twisted_matricizeconj(isconj2, a2, perm2_codomain, perm2_domain, :codomain)
    else
        @noinline TensorAlgebra.matricizeconj(isconj2, a2, perm2_codomain, perm2_domain)
    end
    # Dispatched at run time, since each matricization is the stored matrix or its adjoint
    # depending on the run-time conjugation flags.
    return Base.inferencebarrier(_unmatricizemuladd!)(
        a_dest, perm_dest_codomain, perm_dest_domain, a1_mat, a2_mat, α, β
    )
end

@noinline function _unmatricizemuladd!(
        a_dest, perm_dest_codomain, perm_dest_domain, a1_mat, a2_mat, α, β
    )
    if TensorAlgebra.isoutputview(
            TensorAlgebra.matricizeconj, false, a_dest, perm_dest_codomain, perm_dest_domain
        )
        a_dest_mat = TensorAlgebra.matricizeconjview(
            false, a_dest, perm_dest_codomain, perm_dest_domain
        )
        LinearAlgebra.mul!(a_dest_mat, a1_mat, a2_mat, α, β)
    else
        a_dest_mat = a1_mat * a2_mat
        TensorAlgebra.unmatricizeadd!(
            a_dest, a_dest_mat, perm_dest_codomain, perm_dest_domain, α, β
        )
    end
    return a_dest
end

# Whether the two axis groups hold the same axes with the same repeats, in any order. Reordering
# the factors changes the fusion trees but not the coupled sectors or their multiplicities
# (`Nsymbol` is symmetric in the two fusing sectors), so the groups fuse to an equal coupled axis
# for abelian and non-abelian sectors alike. The ordered pass comes first because in the common
# case the axes are the operand's own stored objects.
function ismultisetequal(axs1::Tuple, axs2::Tuple)
    length(axs1) == length(axs2) || return false
    all(((ax1, ax2),) -> ax1 === ax2, zip(axs1, axs2)) && return true
    return all(ax -> count(isequal(ax), axs1) == count(isequal(ax), axs2), axs1)
end

# Allocate a graded destination over the given external axes, with the domain axes un-dualized,
# reusing an operand's stored coupled axis instead of re-fusing whenever the destination side
# holds a reordering of that operand's stored codomain/domain group.
function allocate_graded(
        ::Type{T}, a1::GradedArray, a2::GradedArray,
        axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    S = sectortype(a1)
    coupled_codomain = coupled_axis(
        S, axes_codomain, GA.axes_codomain(a1), axis_codomain(matricize(a1))
    )
    coupled_domain = coupled_axis(
        S, axes_domain, GA.axes_domain(a2), axis_domain(matricize(a2))
    )
    m = FusedGradedMatrix{T}(undef, coupled_codomain, coupled_domain)
    return GradedArray(m, axes_codomain, axes_domain)
end
# Given one side's axes rather than the operands, so it compiles per leg count, not per operand pair.
@noinline function coupled_axis(
        ::Type{S},
        axs::Tuple,
        axs_stored::Tuple,
        coupled_stored
    ) where {S}
    ismultisetequal(axs, axs_stored) && return coupled_stored
    return fuseaxes(S, axs)
end

# Unlike the generic method this skips the `zero!`, since every consumer overwrites the
# destination in full.
function TensorAlgebra.allocate_output(
        ::typeof(TensorAlgebra.contractpermconjalign),
        perm_dest_codomain, perm_dest_domain,
        isconj1::Bool, a1::Union{GradedArray, AbstractFusedGradedMatrix}, perm1_codomain,
        perm1_domain,
        isconj2::Bool, a2::Union{GradedArray, AbstractFusedGradedMatrix}, perm2_codomain,
        perm2_domain
    )
    a1, a2 = to_gradedarray(a1), to_gradedarray(a2)
    axes_codomain, axes_domain = TensorAlgebra.output_axes(
        TensorAlgebra.contractpermconjalign,
        perm_dest_codomain, perm_dest_domain,
        isconj1, axes(a1), perm1_codomain, perm1_domain,
        isconj2, axes(a2), perm2_codomain, perm2_domain
    )
    T = Base.promote_op(LinearAlgebra.matprod, eltype(a1), eltype(a2))
    return allocate_graded(T, a1, a2, axes_codomain, axes_domain)
end

function TensorAlgebra.allocate_output(
        ::typeof(TensorAlgebra.bipermutedimsconj), isconj::Bool, src::GradedArray,
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    )
    axes_codomain = map(i -> isconj ? conj(axes(src, i)) : axes(src, i), perm_codomain)
    axes_domain = map(i -> isconj ? axes(src, i) : conj(axes(src, i)), perm_domain)
    return allocate_graded(eltype(src), src, src, axes_codomain, axes_domain)
end

# A fused matrix is already its own `{1,1}` matricization.
function TensorAlgebra.isoutputview(
        ::typeof(TensorAlgebra.matricizeconj), isconj::Bool, a::AbstractFusedGradedMatrix,
        perm_codomain, perm_domain
    )
    return !isconj && length(perm_codomain) == ndims_codomain(a) &&
        TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain)
end
function TensorAlgebra.matricizeconjview(
        isconj::Bool, m::AbstractFusedGradedMatrix, perm_codomain, perm_domain
    )
    return m
end
function TensorAlgebra.matricizeconjcopy(
        isconj::Bool, m::AbstractFusedGradedMatrix, perm_codomain, perm_domain
    )
    length(perm_codomain) == 1 || throw(
        ArgumentError(
            "a matrix-level fused array matricizes only with a single codomain leg"
        )
    )
    !isconj && TensorAlgebra.isidentitybiperm(perm_codomain, perm_domain) &&
        return copy(m)
    return TensorAlgebra.permutedimsconj(isconj, m, perm_codomain, perm_domain)
end

function TensorAlgebra.unmatricizeadd!(
        a_dest::GradedArray{<:Any, <:Any, N}, m::AbstractFusedGradedMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}},
        α::Number, β::Number
    ) where {N}
    # With the identity permutation and equal coupled axes the buffers have the same layout.
    md = matricize(a_dest)
    if m isa FusedGradedMatrix && md isa FusedGradedMatrix &&
            ndims_codomain(a_dest) == length(invperm_codomain) &&
            (invperm_codomain..., invperm_domain...) == ntuple(identity, Val(N)) &&
            axis_codomain(m) == axis_codomain(md) && axis_domain(m) == axis_domain(md)
        TensorAlgebra.scaleadd!(md.buffer, m.buffer, α, β)
        return a_dest
    end
    # The permutation back follows the codomain/domain split of `a_dest`, which the graded
    # `bipermutedimsconjadd!` requires.
    axes_codomain_m, axes_domain_m = TensorAlgebra.bipartition_axes(
        axes(a_dest), invperm_codomain, invperm_domain
    )
    tmp = GradedArray(m, axes_codomain_m, axes_domain_m)
    perm_dest = TupleTools.invperm((invperm_codomain..., invperm_domain...))
    ndims_cod_dest = ndims_codomain(a_dest)
    perm_codomain = ntuple(i -> perm_dest[i], Val(ndims_cod_dest))
    perm_domain = ntuple(i -> perm_dest[ndims_cod_dest + i], Val(N - ndims_cod_dest))
    TensorAlgebra.bipermutedimsconjadd!(
        a_dest,
        false,
        tmp,
        perm_codomain,
        perm_domain,
        α,
        β
    )
    return a_dest
end

# ============================  concatenation  ============================
# Abelian only. `cat` / `directsum` will be reimplemented for non-abelian fusion.
function TensorAlgebra.concatenate!(dest::GradedArray, dims, args...)
    with_block_indexing() do
        return concatenate_sparse!(blocks(dest), dims, blocks.(args)...)
    end
    return dest
end
Base._cat(dims, as::GradedArray...) = TensorAlgebra.concatenate(dims, as...)

# ============================  project (dense -> symmetric)  ============================

# The generic `projectto!` would scalar-index and error on a forbidden block.
function TensorAlgebra.projectto!(dest::GradedArray, src::AbstractArray)
    TensorAlgebra.projectto!(to_tensormap(dest), src)
    return dest
end

# The `TensorMap` has sorted, merged axes, so for other axes the sector blocks are moved back to
# the stored order. TensorKit's `convert(Array, t)` widens the element type by the sector scalar
# type, so convert back to `T` afterwards. This throws for an integer `T` with a non-abelian sector.
function Base.Array{T, N}(fa::GradedArray{<:Any, <:Any, N}) where {T, N}
    dense = convert(Array{T, N}, convert(Array, to_tensormap(fa)))
    all(isfusedsorted, axes(fa)) && return dense
    sortedlengths = map(g -> Vector(blocklengths(g))[sortperm(sectors(g))], axes(fa))
    invperms = ntuple(d -> Block.(invperm(sortperm(sectors(axes(fa)[d])))), Val(N))
    return parent(BlockedArray(dense, sortedlengths...)[invperms...])
end
# TensorKit's `convert(Array, t)` errors for some rank-0 eltypes, for example `Float32`, from a
# missing VectorInterface `add!` method.
Base.Array{T, 0}(fa::GradedArray{<:Any, <:Any, 0}) where {T} = fill(convert(T, fa[]))

# The dense form at codomain split `K`. Bending the legs to that split applies the fermion signs.
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
    show_biaxes(io, fa)
    ndims(fa) == 0 || println(io)
    show(io, MIME"text/plain"(), matricize(fa))
    return nothing
end
