# =============================================================================
#  FusionArray — always-fused symmetric array backed by a matricized
#  `FusedGradedMatrix`.
# =============================================================================

using LinearAlgebra: LinearAlgebra
using Random: Random, AbstractRNG
using TensorKit: TensorKit as TK

"""
    FusionArray{T,S,N} <: AbstractGradedArray{T,S,N}

Always-fused symmetric array: an `N`-dimensional graded array with a codomain/domain split,
backed by a matricized [`FusedGradedMatrix`](@ref). The external axes are `GradedOneTo` and,
in this initial form, are required to be fused and sorted (each sector once).
"""
struct FusionArray{
        T, S, N, M <: FusedGradedMatrix{T, S}, NC, ND,
    } <: AbstractGradedArray{T, S, N}
    matricized::M
    axes_codomain::NTuple{NC, GradedOneTo{S}}
    axes_domain::NTuple{ND, GradedOneTo{S}}

    function FusionArray(
            matricized::FusedGradedMatrix{T, S},
            axes_codomain::NTuple{NC, GradedOneTo{S}},
            axes_domain::NTuple{ND, GradedOneTo{S}}
        ) where {T, S, NC, ND}
        # In this initial form the external axes must be fused and sorted (each sector once);
        # see `check_fused_sorted`.
        foreach(check_fused_sorted, axes_codomain)
        foreach(check_fused_sorted, axes_domain)
        return new{T, S, NC + ND, typeof(matricized), NC, ND}(
            matricized, axes_codomain, axes_domain
        )
    end
end

# ============================  Accessors  ============================

axes_codomain(fa::FusionArray) = fa.axes_codomain
axes_domain(fa::FusionArray) = fa.axes_domain

# Domain axes are stored codomain-facing (TensorKit's `domain` convention); `axes` dualizes them
# so a domain leg reads as a dual axis, matching TensorKit's `space(t, i)`.
Base.axes(fa::FusionArray) = (axes_codomain(fa)..., map(dual, axes_domain(fa))...)
Base.size(fa::FusionArray) = map(length, axes(fa))

ndims_codomain(fa::FusionArray) = length(axes_codomain(fa))
ndims_domain(fa::FusionArray) = length(axes_domain(fa))

# One-argument `matricize` uses the array's own codomain/domain split, so it is the stored
# matrix directly (see `matricize(::FusionArrayFusionStyle, …)` for re-splitting to another).
TensorAlgebra.matricize(fa::FusionArray) = fa.matricized

# ============================  block indexing (unique fusion)  ============================
# A `Block` picks a per-leg block position on each axis (positional, not sector-keyed). For unique
# fusion those positional sectors name the one codomain/domain fusion-tree pair, whose reduced block
# is a strided sub-region of the fused coupled matrix — exactly what `FusionMap`'s `subblock` exposes.
# The returned `AbelianSectorArray` shares that strided data, so element get/set writes back in place.
# Guarded to unique fusion: for non-abelian the external leg sectors under-determine the block (they
# fix neither the internal fusion sectors nor the vertex multiplicities), so `Block` cannot name it.

# The uncoupled sector of a leg as its fusion tree carries it: dualized on a dual axis (the dual flag
# is tracked separately on the tree), matching TensorKit's external-sector indexing convention.
_uncoupled_sector(r::SectorRange) = isdual(r) ? TKS.dual(label(r)) : label(r)

function view_fusion(a::FusionArray{T, <:Any, N}, I::Block{N}) where {T, N}
    require_unique_fusion(a)
    bk = Int.(Tuple(I))
    sects = ntuple(d -> eachsectoraxis(axes(a, d))[bk[d]], Val(N))
    blockdata = FusionMap(a)[map(_uncoupled_sector, sects)]
    return AbelianSectorArray(sects, blockdata)
end

Base.view(a::FusionArray{T, <:Any, N}, I::Block{N}) where {T, N} = view_fusion(a, I)
# Disambiguate the N=1 case against the `Vararg{Block{1}, N}` method, as `AbelianGradedArray` does.
Base.view(a::FusionArray{T, <:Any, 1}, I::Block{1}) where {T} = view_fusion(a, I)

# A `FusionArray` block is the same unique-fusion `AbelianSectorArray` the abelian backend returns.
# TODO: derive the block data type from the `FusedGradedMatrix` block type (its `D`) rather than
# hardcoding `Array{T, N}`, so non-`Array` storage (GPU, etc.) is preserved — e.g. via
# `Base.promote_op` on `view(A, ::Block)` (the actual returned block type). Also only well-defined
# for unique fusion (blocks are `AbelianSectorArray` only then); tie it to that guard.
function blocktype(::Type{<:FusionArray{T, S, N}}) where {T, S, N}
    return AbelianSectorArray{T, S, N, Array{T, N}}
end
blocktype(a::FusionArray) = blocktype(typeof(a))

# ============================  similar  ============================
# `similar` must build a `FusionArray`, not route through the `AbstractGradedArray` `similar` that
# allocates an `AbelianGradedArray`. Without explicit axes, preserve the prototype's own
# codomain/domain split (like `copy`); with explicit flat axes the split is unrecoverable (see the
# `copy` note), so put all axes in the codomain, matching the broadcast `similar`.
function Base.similar(a::FusionArray, ::Type{T}) where {T}
    return FusionArray(similar(matricize(a), T), axes_codomain(a), axes_domain(a))
end
function Base.similar(
        ::FusionArray, ::Type{T}, axes::Tuple{GradedOneTo{S}, Vararg{GradedOneTo{S}}}
    ) where {T, S}
    return FusionArray{T}(undef, axes, ())
end

# ============================  copyto!  ============================
# Block-wise copy between `FusionArray`s (e.g. a factorization's `copy_input`). The generic
# `AbstractArray` `copyto!` scalar-indexes, which errors on forbidden (symmetry-disallowed) blocks;
# copy the shared coupled-sector matrices directly instead. Requires the same coupled structure,
# which a `similar`-allocated destination has.
function Base.copyto!(dest::FusionArray, src::FusionArray)
    for (c, b) in pairs(matricize(src).blocks)
        copyto!(matricize(dest).blocks[c], b)
    end
    return dest
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

# ============================  matrix product  ============================
# Matrix-matrix product as a contraction over the shared leg, mirroring `AbelianGradedMatrix`. The
# generic `LinearAlgebra` matmul scalar-indexes, which errors on forbidden blocks.
function Base.:*(a::FusionArray{<:Any, <:Any, 2}, b::FusionArray{<:Any, <:Any, 2})
    return TensorAlgebra.contract((1, 3), a, (1, 2), b, (2, 3))
end

# ============================  TensorMap conversion  ============================

"""
    TK.TensorMap(fa::FusionArray)

Convert a `FusionArray` to a `TK.TensorMap`, building the codomain/domain product
spaces from the per-leg axes and copying each coupled-sector block.
"""
function TK.TensorMap(fa::FusionArray)
    Sp = typeof(ElementarySpace(first(axes(fa))))
    codsp = mapreduce(ElementarySpace, TK.:⊗, axes_codomain(fa); init = one(Sp))
    domsp = mapreduce(ElementarySpace, TK.:⊗, axes_domain(fa); init = one(Sp))
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
# matching `similar_map`/`unmatricize`.
function FusionArray{T}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    # Fuse each side's per-leg axes into its coupled `GradedOneTo` (through the GradedArrays
    # interface, which uses the TensorKitSectors fusion rules), seeding empty groups with the
    # trivial sector. `FusedGradedMatrix{T}(undef, …)` then allocates the reduced blocks.
    S = sectortype(first((axes_codomain..., axes_domain...)))
    init = trivial_gradedrange(S)
    coupled_codomain = reduce(tensor_product, axes_codomain; init)
    coupled_domain = reduce(tensor_product, axes_domain; init)
    m = FusedGradedMatrix{T}(undef, coupled_codomain, coupled_domain)
    return FusionArray(m, axes_codomain, axes_domain)
end

# A `FusionArray` source always reproduces a `FusionArray`, independent of the `graded_backend`
# preference (so `FusionArray` stays self-consistent even when it is not the default backend). The
# graded axes make these more specific than the backend-routing `similar_map` in `abeliangradedarray.jl`.
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

# Fill the reduced coupled-sector blocks in place, forwarding to the matricized `FusedGradedMatrix`
# (which fills via `eachblockstoredindex`). Construct with `FusionArray{T}(undef, …)` first.
for f! in (:rand!, :randn!)
    @eval begin
        Random.$f!(rng::AbstractRNG, fa::FusionArray) = (Random.$f!(rng, matricize(fa)); fa)
        Random.$f!(fa::FusionArray) = Random.$f!(Random.default_rng(), fa)
    end
end

# ============================  in-place primitives / algebra  ============================
# The inherited `AbstractGradedArray` `zero!`/`scale!`/`norm`/`real`/`imag` walk
# `eachblockstoredindex`, which `FusionArray` does not implement, so forward to the matricized
# `FusedGradedMatrix`. `+`/`-` are left to the `AbstractArray` broadcast machinery.

TensorAlgebra.zero!(fa::FusionArray) = (zero!(matricize(fa)); fa)
TensorAlgebra.scale!(fa::FusionArray, α::Number) = (scale!(matricize(fa), α); fa)
LinearAlgebra.norm(fa::FusionArray, p::Real = 2) = LinearAlgebra.norm(matricize(fa), p)
Base.fill!(fa::FusionArray, v) = (fill!(matricize(fa), v); fa)
Base.iszero(fa::FusionArray) = iszero(matricize(fa))

# Copy the matricized matrix and reuse the axes. Defined directly (rather than through `similar`)
# because the generic `AbstractGradedArray` `similar` takes flat axes and cannot recover the
# codomain/domain split a `FusionArray` needs; `copy` must preserve it (e.g. `one!!(copy(A), …)`
# relies on the copy keeping `A`'s split so the identity fill lands in the stored matrix).
function Base.copy(fa::FusionArray)
    return FusionArray(copy(matricize(fa)), axes_codomain(fa), axes_domain(fa))
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

# ============================  broadcasting  ============================
# Opt in to the graded linear-broadcast machinery so linear combinations (`a + b`, `2a - b`, …)
# materialize through `bipermutedimsopadd!`. The shared `AbstractGradedArray` `copyto!` and the
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

# ============================  bipermutedimsopadd! (permute primitive)  ============================
# `y = α * op.(permute(x, …)) + β * y`, delegated to the `AbstractTensorMap` `bipermutedimsopadd!`
# (fusion-tree recombination plus braiding/fermion signs). Wrapping `y`/`x` as `FusionMap`s shares
# their blocks, so TensorKit writes the result into `y` in place.

function TensorAlgebra.bipermutedimsopadd!(
        y::FusionArray, op, x::FusionArray,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    TensorAlgebra.bipermutedimsopadd!(
        FusionMap(y), op, FusionMap(x), perm_codomain, perm_domain, α, β
    )
    return y
end

# A `FusedGradedMatrix` source (e.g. a matricized factorization output permuted into a `FusionArray`
# destination) is wrapped zero-copy as a one-codomain/one-domain `FusionArray` — its stored matrix is
# already the `FusionArray` matricized form — and forwarded to the method above. Without this the
# generic `AbstractGradedArray` permute-add would block-index the `FusionArray` destination, which it
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
    TK.twist!(FusionMap(a), dims)
    return a
end

# ============================  TensorAlgebra primitive interface  ============================
# `matricize` returns the stored `FusedGradedMatrix`, `unmatricize` puts axes back on. Because
# `FusedGradedMatrix` already has `mul!` and block-wise factorizations, contraction and
# factorizations ride the generic `TensorAlgebra` machinery with no high-level overloads.

struct FusionArrayFusionStyle <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:FusionArray}) = FusionArrayFusionStyle()

# When the requested split matches the stored split this is the stored matrix. Otherwise it is a
# leg bend (the `matricizeopperm` fast path only reaches here with an identity permutation, so
# legs stay in order and only the codomain/domain boundary moves), which for a `FusionArray` is
# not a free reshape. Re-split with the array's own `bipermutedims`, then take its 1-arg
# `matricize`. This is what lets a contraction over a subset of legs matricize a factor whose
# stored split differs.
function TensorAlgebra.matricize(
        ::FusionArrayFusionStyle,
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
        ::FusionArrayFusionStyle, m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple
    )
    return FusionArray(m, axes_codomain, axes_domain)
end

# ============================  contraction (SectorFusion path)  ============================
# Contraction dispatches through `default_contract_algorithm(::AbstractGradedArray, …)`, which
# fixes the `SectorFusion`/`TwistedSectorFusion` styles, so a `FusionArray` contraction rides
# that path rather than `FusionArrayFusionStyle`.

function TensorAlgebra.matricize(::SectorFusion, fa::FusionArray, ndims_cod::Val)
    return matricize(FusionArrayFusionStyle(), fa, ndims_cod)
end

function TensorAlgebra.unmatricizeperm!(
        ::SectorFusion, a_dest::FusionArray{<:Any, <:Any, N}, m::FusedGradedMatrix,
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

# ============================  project (dense -> symmetric)  ============================
# The dense-to-symmetric projection for the `FusionArray` backend is delegated to TensorKit in the
# shared `unchecked_project_graded` worker (`abeliangradedarray.jl`): it projects over the equivalent
# `ElementarySpace`s and wraps the resulting `TensorMap` with `FusionArray(t)`. `unproject` below is
# the dense inverse used by `TA.project`'s verification.

# Dense form. The matricized `FusionArray` does not implement `eachblockstoredindex` (the generic
# `AbstractGradedArray` dense path), so materialize through the `TensorMap`, whose `convert(Array, …)`
# lays out `(codomain…, domain…)` in the dualized-domain convention `axes(fa)` reports.
Base.Array(fa::FusionArray) = convert(Array, TK.TensorMap(fa))

# `unproject` is the dense inverse of `projectto!` used by `TA.project`'s verification. The dense form
# already carries the array's own codomain/domain split, so the requested split `Val{K}` must match
# `fa`'s codomain length; the `TensorMap` conversion then undoes the domain-leg bend.
function TensorAlgebra.unproject(fa::FusionArray, ::Val{K}) where {K}
    K == ndims_codomain(fa) || throw(
        ArgumentError(
            "`unproject` codomain split $K does not match the FusionArray codomain $(ndims_codomain(fa))"
        )
    )
    return Array(fa)
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
