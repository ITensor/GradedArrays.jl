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
# matrix directly (see `matricize(::FusionArrayStyle, …)` for re-splitting to another).
TensorAlgebra.matricize(fa::FusionArray) = fa.matricized

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
    t = TK.zeros(eltype(fa), codsp, domsp)
    for (c, b) in pairs(matricize(fa).blocks)
        copy!(TK.block(t, label(c)), b)
    end
    return t
end

"""
    FusionArray(t::TK.AbstractTensorMap)

Build a `FusionArray` from a `TensorMap`, taking the per-leg external axes from its codomain
and domain spaces.
"""
function FusionArray(t::TK.AbstractTensorMap)
    axes_codomain = map(GradedOneTo, Tuple(TK.codomain(t)))
    axes_domain = map(GradedOneTo, Tuple(TK.domain(t)))
    return FusionArray(FusedGradedMatrix(t), axes_codomain, axes_domain)
end

# The coupled row/column axes fuse *all* reachable sectors of one side, not only the shared
# (stored) blocks, so a bond axis matches across a contraction even when one side reaches sectors
# the other trims. `fuse` gives that coupled space; `GradedOneTo` carries it to the axis type.
function FusedGradedMatrix(t::TK.AbstractTensorMap)
    cs = collect(TK.blocksectors(t))
    secs = SectorRange.(cs)
    p = sortperm(secs)
    blocks = Dictionary(secs[p], [copy(TK.block(t, c)) for c in cs[p]])
    return FusedGradedMatrix(
        blocks, GradedOneTo(TK.fuse(TK.codomain(t))), GradedOneTo(TK.fuse(TK.domain(t)))
    )
end

# ============================  construction from axes  ============================

# Axes are given codomain-facing (un-dualized), the same convention they are stored in,
# matching `similar_map`/`unmatricize`.
function FusionArray{T}(
        ::UndefInitializer, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    Sp = typeof(ElementarySpace(first((axes_codomain..., axes_domain...))))
    codsp = mapreduce(ElementarySpace, TK.:⊗, axes_codomain; init = one(Sp))
    domsp = mapreduce(ElementarySpace, TK.:⊗, axes_domain; init = one(Sp))
    m = FusedGradedMatrix(TK.zeros(T, codsp, domsp))
    return FusionArray(m, axes_codomain, axes_domain)
end

function TensorAlgebra.similar_map(
        ::FusionArray, ::Type{T}, axes_codomain::Tuple, axes_domain::Tuple
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end

# Fill the reduced coupled-sector blocks in place. Construct with `FusionArray{T}(undef, …)` first.
for f! in (:rand!, :randn!)
    @eval begin
        function Random.$f!(rng::AbstractRNG, fa::FusionArray)
            for b in values(matricize(fa).blocks)
                Random.$f!(rng, b)
            end
            return fa
        end
        Random.$f!(fa::FusionArray) = Random.$f!(Random.default_rng(), fa)
    end
end

# ============================  in-place primitives / algebra  ============================
# The inherited `AbstractGradedArray` `zero!`/`scale!`/`norm` walk `eachblockstoredindex`, which
# `FusionArray` does not implement, so forward to the matricized `FusedGradedMatrix`. `+`/`-` are
# left to the `AbstractArray` broadcast machinery.

TensorAlgebra.zero!(fa::FusionArray) = (zero!(matricize(fa)); fa)
TensorAlgebra.scale!(fa::FusionArray, α::Number) = (scale!(matricize(fa), α); fa)
LinearAlgebra.norm(fa::FusionArray, p::Real = 2) = LinearAlgebra.norm(matricize(fa), p)

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

struct FusionBroadcastStyle{N} <: AbstractGradedStyle{N} end
FusionBroadcastStyle{N}(::Val{M}) where {N, M} = FusionBroadcastStyle{M}()

function BC.BroadcastStyle(::Type{<:FusionArray{<:Any, <:Any, N}}) where {N}
    return FusionBroadcastStyle{N}()
end

# Build the result with all axes in the codomain (matching TensorKit's move-to-codomain convention
# for `+`/`-`), so operands with any codomain/domain split are bent into it by the fold.
function Base.similar(bc::BC.Broadcasted{<:FusionBroadcastStyle}, elt::Type)
    return FusionArray{elt}(undef, axes(flattenlinear(bc)), ())
end

# ============================  bipermutedimsopadd! (permute primitive)  ============================
# `y = α * op.(permute(x, …)) + β * y`, delegated to the `TensorMap` `bipermutedimsopadd!` in
# `TensorAlgebraTensorKitExt` (fusion-tree recombination plus braiding/fermion signs). The
# twisted blocks are copied back into `y`.

function TensorAlgebra.bipermutedimsopadd!(
        y::FusionArray, op, x::FusionArray,
        perm_codomain, perm_domain, α::Number, β::Number
    )
    ty = TK.TensorMap(y)
    TensorAlgebra.bipermutedimsopadd!(
        ty, op, TK.TensorMap(x), perm_codomain, perm_domain, α, β
    )
    for c in TK.blocksectors(ty)
        copy!(matricize(y).blocks[SectorRange(c)], TK.block(ty, c))
    end
    return y
end

# ============================  fermionic twist  ============================
# The contraction twist scales blocks by a per-fusion-tree fermion phase, so delegate to
# `TK.twist!` on a `TensorMap` copy and copy the twisted blocks back. A zero-copy
# `AbstractTensorMap` view over the `FusedGradedMatrix` would remove the copy.
function twist!(a::FusionArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    t = TK.TensorMap(a)
    TK.twist!(t, collect(dims))
    for c in TK.blocksectors(t)
        copy!(matricize(a).blocks[SectorRange(c)], TK.block(t, c))
    end
    return a
end

# ============================  TensorAlgebra primitive interface  ============================
# `matricize` returns the stored `FusedGradedMatrix`, `unmatricize` puts axes back on. Because
# `FusedGradedMatrix` already has `mul!` and block-wise factorizations, contraction and
# factorizations ride the generic `TensorAlgebra` machinery with no high-level overloads.

struct FusionArrayStyle <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:FusionArray}) = FusionArrayStyle()

# When the requested split matches the stored split this is the stored matrix. Otherwise it is a
# leg bend (the `matricizeopperm` fast path only reaches here with an identity permutation, so
# legs are already in order and only the codomain/domain boundary moves), which for a
# `FusionArray` is not a free reshape, so delegate it to TK. This is what lets a
# contraction over a subset of legs matricize a factor whose stored split differs.
function TensorAlgebra.matricize(::FusionArrayStyle, fa::FusionArray, ::Val{K}) where {K}
    K == ndims_codomain(fa) && return matricize(fa)
    N = ndims(fa)
    tb = TK.permute(
        TK.TensorMap(fa),
        (ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K)))
    )
    return FusedGradedMatrix(tb)
end

function TensorAlgebra.unmatricize(
        ::FusionArrayStyle, m::FusedGradedMatrix, axes_codomain::Tuple, axes_domain::Tuple
    )
    return FusionArray(m, axes_codomain, axes_domain)
end

# ============================  contraction (SectorFusion path)  ============================
# Contraction dispatches through `default_contract_algorithm(::AbstractGradedArray, …)`, which
# fixes the `SectorFusion`/`TwistedSectorFusion` styles, so a `FusionArray` contraction rides
# that path rather than `FusionArrayStyle`.

function TensorAlgebra.matricize(::SectorFusion, fa::FusionArray, ndims_cod::Val)
    return matricize(FusionArrayStyle(), fa, ndims_cod)
end

function TensorAlgebra.unmatricizeperm!(
        ::SectorFusion, a_dest::FusionArray{<:Any, <:Any, N}, m::FusedGradedMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    ) where {N}
    # Permute `a_dest` into the matricized leg order to get the matricized-order axes with correct
    # per-leg duality; the permuted data is discarded. Wrap `m` in those axes, then permute back.
    template =
        TensorAlgebra.permutedimsop(identity, a_dest, invperm_codomain, invperm_domain)
    tmp = FusionArray(m, axes_codomain(template), axes_domain(template))
    perm_dest = invperm((invperm_codomain..., invperm_domain...))
    ndims_cod_dest = ndims_codomain(a_dest)
    perm_codomain = ntuple(i -> perm_dest[i], Val(ndims_cod_dest))
    perm_domain = ntuple(i -> perm_dest[ndims_cod_dest + i], Val(N - ndims_cod_dest))
    bipermutedimsopadd!(
        a_dest, identity, tmp, perm_codomain, perm_domain,
        one(eltype(a_dest)), zero(eltype(a_dest))
    )
    return a_dest
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
