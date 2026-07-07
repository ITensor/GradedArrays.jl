using MatrixAlgebraKit: MatrixAlgebraKit as MAK

struct GradedBlockAlgorithm{A <: MAK.AbstractAlgorithm} <: MAK.AbstractAlgorithm
    alg::A
end

# Utility
# -------
for f in [
        :svd_compact, :svd_full, :svd_vals,
        :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null,
        :eig_full, :eig_vals, :eigh_full, :eigh_vals,
        :left_polar, :right_polar,
        :project_hermitian, :project_antihermitian, :project_isometric,
    ]
    f! = Symbol(f, :!)
    @eval function MAK.default_algorithm(
            ::typeof(MAK.$f!), ::Type{T}; kwargs...
        ) where {T <: FusedGradedMatrix}
        return GradedBlockAlgorithm(
            MAK.default_algorithm(
                MAK.$f!, datatype(T);
                kwargs...
            )
        )
    end

    @eval function MAK.copy_input(::typeof(MAK.$f), A::FusedGradedMatrix)
        return FusedGradedMatrix(
            A.codomain, A.domain,
            map(Base.Fix1(MAK.copy_input, MAK.$f), A.blocks)
        )
    end
end

# Generic Implementations
# -----------------------
# utility function to do something with each block
const FusedGradedArray = Union{FusedGradedMatrix, FusedGradedVector}

function _blockdataaxes(a::FusedGradedMatrix, c)
    return (Base.OneTo(get(a.codomain, c, 0)), Base.OneTo(get(a.domain, c, 0)))
end
_blockdataaxes(a::FusedGradedVector, c) = (Base.OneTo(get(a.axis, c, 0)),)

function foreachblock(f, A::FusedGradedArray, As::FusedGradedArray...)
    cs = union(map(keys ∘ Base.Fix2(getproperty, :blocks), (A, As...))...)

    for c in cs
        bs = map((A, As...)) do a
            get(a.blocks, c) do
                return similar(valtype(a.blocks), _blockdataaxes(a, c))
            end
        end
        f(c, bs)
    end

    return nothing
end

# in cases where the factorization/alg does not result in in-place, we try to force it by copying.
_ensure_inplace!(F, F′) = F === F′ || copy!(F, F′)
_ensure_inplace!(F::NTuple{N}, F′::NTuple{N}) where {N} = _ensure_inplace!.(F, F′)

for f! in (
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    )
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
        $(f! in (:eig_full!, :eigh_full!) && :(LinearAlgebra.checksquare(A)))
        foreachblock(A, F...) do _, (Ablock, Fblocks...)
            Fblocks′ = MAK.$f!(Ablock, Fblocks, alg.alg)
            return _ensure_inplace!(Fblocks, Fblocks′)
        end
        return F
    end
end

# Handle these separately because single output instead of tuple
for f! in (
        :qr_null!, :lq_null!,
        :svd_vals!, :eig_vals!, :eigh_vals!,
        :project_hermitian!, :project_antihermitian!, :project_isometric!,
    )
    @eval function MAK.$f!(A::FusedGradedMatrix, N, alg::GradedBlockAlgorithm)
        $(
            f! in (:eig_vals!, :eigh_vals!, :project_hermitian!, :project_antihermitian!) &&
                :(LinearAlgebra.checksquare(A))
        )
        foreachblock(A, N) do _, (Ablock, Nblock)
            Nblock′ = MAK.$f!(Ablock, Nblock, alg.alg)
            return _ensure_inplace!(Nblock, Nblock′)
        end
        return N
    end
end

# Boolean output
for f in [
        :isunitary, :isisometric, :is_left_isometric, :is_right_isometric,
        :ishermitian, :isantihermitian,
    ]
    @eval function MAK.$f(A::FusedGradedMatrix; kwargs...)
        return all(x -> MAK.$f(x; kwargs...), A.blocks)
    end
end

# initialize_outputs: have to compute the correct sizes for all sectors
# since these might be present or missing
# =====================================================================

# helper
function similar_diagonal(A::FusedGradedMatrix, ::Type{T}, V) where {T}
    blocks = map(V) do d
        return Diagonal(similar(Vector{T}, d))
    end
    return FusedGradedMatrix(V, V, blocks)
end

# Singular value decomposition
# ----------------------------
function MAK.initialize_output(
        ::typeof(MAK.svd_full!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    U = similar(A, A.codomain, A.codomain)
    S = similar(A, real(eltype(A)), A.codomain, A.domain)
    Vᴴ = similar(A, A.domain, A.domain)
    return U, S, Vᴴ
end
function MAK.initialize_output(
        ::typeof(MAK.svd_compact!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    V_S = map(x -> min(size(x)...), A.blocks)
    U = similar(A, A.codomain, V_S)
    Tr = real(eltype(A))
    S = similar_diagonal(A, Tr, V_S)
    Vᴴ = similar(A, V_S, A.domain)
    return U, S, Vᴴ
end
function MAK.initialize_output(
        ::typeof(MAK.svd_vals!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    V_S = map(x -> min(size(x)...), A.blocks)
    Tr = real(eltype(A))
    return similar(A, Vector{Tr}, V_S) # TODO: don't hardcode type
end

# Eigenvalue decomposition
# ------------------------
function MAK.initialize_output(
        ::typeof(MAK.eig_full!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    Tc = complex(eltype(A))
    D = similar_diagonal(A, Tc, A.domain)
    V = similar(A, Tc)
    return D, V
end
function MAK.initialize_output(
        ::typeof(MAK.eig_vals!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    Tc = complex(eltype(A))
    return similar(A, Vector{Tc}, A.domain) # TODO: don't hardcode type
end

function MAK.initialize_output(
        ::typeof(MAK.eigh_full!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    Tr = real(eltype(A))
    D = similar_diagonal(A, Tr, A.domain)
    V = similar(A)
    return D, V
end
function MAK.initialize_output(
        ::typeof(MAK.eigh_vals!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    Tr = real(eltype(A))
    return similar(A, Vector{Tr}, A.domain) # TODO: don't hardcode type
end

# QR decomposition
# ----------------
function MAK.initialize_output(
        ::typeof(MAK.qr_full!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    Q = similar(A, A.codomain, A.codomain)
    R = similar(A, A.codomain, A.domain)
    return Q, R
end
function MAK.initialize_output(
        ::typeof(MAK.qr_compact!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    V_Q = map(x -> min(size(x)...), A.blocks)
    Q = similar(A, A.codomain, V_Q)
    R = similar(A, V_Q, A.domain)
    return Q, R
end
function MAK.initialize_output(
        ::typeof(MAK.qr_null!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    V_N = copy(A.codomain)
    for (c, d₁) in pairs(V_N)
        d₂ = get(A.domain, c, 0)
        V_N[c] = max(d₁ - d₂, 0)
    end
    filter!(!iszero, V_N)
    return similar(A, A.codomain, V_N)
end

# LQ decomposition
# ----------------
function MAK.initialize_output(
        ::typeof(MAK.lq_full!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    L = similar(A, A.codomain, A.domain)
    Q = similar(A, A.domain, A.domain)
    return L, Q
end
function MAK.initialize_output(
        ::typeof(MAK.lq_compact!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    V_Q = map(x -> min(size(x)...), A.blocks)
    L = similar(A, A.codomain, V_Q)
    Q = similar(A, V_Q, A.domain)
    return L, Q
end
function MAK.initialize_output(
        ::typeof(MAK.lq_null!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    V_N = copy(A.domain)
    for (c, d₂) in pairs(V_N)
        d₁ = get(A.codomain, c, 0)
        V_N[c] = max(d₂ - d₁, 0)
    end
    filter!(!iszero, V_N)
    return similar(A, V_N, A.domain)
end

# Polar decomposition
# -------------------
function MAK.initialize_output(
        ::typeof(MAK.left_polar!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    W = similar(A)
    P = similar(A, A.domain, A.domain)
    return W, P
end
function MAK.initialize_output(
        ::typeof(MAK.right_polar!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    P = similar(A, A.codomain, A.codomain)
    Wᴴ = similar(A)
    return P, Wᴴ
end

# Projections
# -----------
# Same output conventions as the generic implementations: hermitian and
# antihermitian project in place, isometric writes to a fresh output.
function MAK.initialize_output(
        ::typeof(MAK.project_hermitian!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    return A
end
function MAK.initialize_output(
        ::typeof(MAK.project_antihermitian!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    return A
end
function MAK.initialize_output(
        ::typeof(MAK.project_isometric!),
        A::FusedGradedMatrix,
        alg::GradedBlockAlgorithm
    )
    return similar(A)
end

# Truncation support
# ------------------

# diagview for FusedGradedMatrix: extracts per-block diagonals as a FusedGradedVector
function MAK.diagview(m::FusedGradedMatrix)
    diag_blocks = map(MAK.diagview, m.blocks)
    diag_axis = map(length, diag_blocks)
    return FusedGradedVector(diag_axis, diag_blocks)
end

# Inverse of `diagview`: wrap a FusedGradedVector as a block-diagonal FusedGradedMatrix
# whose inner blocks are `Diagonal`. Keeps the structured form through
# `pow_diag_safe(D) = MAK.diagonal(map(f, MAK.diagview(D)))`, so the next
# `V * MAK.diagonal(...)` stays in the block-diagonal multiplication path instead of
# falling through to LinearAlgebra's scalar-indexing `Diagonal*Matrix` impl.
function MAK.diagonal(v::FusedGradedVector)
    diag_blocks = map(Diagonal, v.blocks)
    return FusedGradedMatrix(v.axis, v.axis, diag_blocks)
end

# `pow_diag_safe!` for a block-diagonal graded matrix: clamp-power each reduced diagonal
# block. Only the reduced (degeneracy) data is touched, and that is correct even in the
# non-abelian case: a diagonal factor is `Diagonal(λ) ⊗ I` per sector, and `f(A ⊗ I) =
# f(A) ⊗ I`, so the power passes straight to the reduced eigenvalues. This is why the
# diagonal power is well defined here whereas a general element-wise `map!` on a graded
# array is not.
function TensorAlgebra.MatrixAlgebra.pow_diag_safe!(
        Dp::FusedGradedMatrix, D::FusedGradedMatrix, p, tol
    )
    foreachblock(MAK.diagview(Dp), MAK.diagview(D)) do _, (σp, σ)
        return map!(d -> _clamped_pow(d, p, tol), σp, σ)
    end
    return Dp
end

# Vendored from `TensorAlgebra.MatrixAlgebra` (not part of its public API): clamp entries
# below `tol` to zero, then raise to `p`; a negative entry above `tol` lets `real(d)^p`
# error for fractional `p`, enforcing the PSD precondition per-power.
_clamped_pow(d, p, tol) = abs(d) < tol ? zero(d) : real(d)^p

# Count how many elements are kept for a given index specification and block size
_count_kept(::Colon, n) = n
_count_kept(ind::AbstractVector{Bool}, _) = count(ind)
_count_kept(ind::AbstractVector, _) = length(ind)

# truncation_error! for FusedGradedVector
# Zeroes out kept values (ind[i]) in each block; returns 2-norm of discarded values.
function MAK.truncation_error!(v::FusedGradedVector, ind::AbstractVector)
    foreach(MAK.truncation_error!, v.blocks, ind)
    return LinearAlgebra.norm(v)
end
function MAK.truncation_error(v::FusedGradedVector, ind::AbstractVector)
    return MAK.truncation_error!(copy(v), ind)
end

# findtruncated / findtruncated_svd for FusedGradedVector
# Both return a Vector where entry i gives the kept indices for block i.

function MAK.findtruncated(v::FusedGradedVector, ::MAK.NoTruncation)
    return [Colon() for _ in v.blocks]
end

# Default: findtruncated_svd falls back to findtruncated (overridden below for some strategies)
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationStrategy)
    return MAK.findtruncated(v, strategy)
end
function MAK.findtruncated_svd(v::FusedGradedVector, ::MAK.NoTruncation)
    return [Colon() for _ in v.blocks]
end

# TruncationByFilter: apply independently per block
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByFilter)
    return [MAK.findtruncated(b, strategy) for b in v.blocks]
end

# TruncationByValue (trunctol): compute global norm for rtol, then apply per block
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * LinearAlgebra.norm(v, strategy.p))
    per_block = MAK.trunctol(; atol, strategy.by, strategy.keep_below, strategy.p)
    return [MAK.findtruncated(b, per_block) for b in v.blocks]
end
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * LinearAlgebra.norm(v, strategy.p))
    per_block = MAK.trunctol(; atol, strategy.by, strategy.keep_below, strategy.p)
    return [MAK.findtruncated_svd(b, per_block) for b in v.blocks]
end

# TruncationByOrder (truncrank k): global top-k across all blocks
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByOrder)
    all_entries = [
        (strategy.by(val), i, j)
            for (i, b) in enumerate(v.blocks)
            for (j, val) in enumerate(b)
    ]
    sort!(all_entries; by = first, strategy.rev)
    kept = [Int[] for _ in v.blocks]
    number_kept = 0
    for (_, i, j) in all_entries
        number_kept += length(gettokenvalue(keys(v.axis), i))
        number_kept > strategy.howmany && break
        push!(kept[i], j)
    end
    sort!.(kept)
    return kept
end
# SVD values are sorted descending within each block but we still need a cross-block comparison
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationByOrder)
    return MAK.findtruncated(v, strategy)
end

# TruncationByError (truncerror): global cumulative error budget, discard smallest first
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByError)
    (isfinite(strategy.p) && strategy.p > 0) ||
        throw(ArgumentError(lazy"p-norm with p=$(strategy.p) not supported"))
    p = strategy.p
    total_norm_p = LinearAlgebra.norm(v, strategy.p)^p
    ϵᵖmax = max(strategy.atol^p, strategy.rtol^p * total_norm_p)

    # Sort all values ascending by abs (smallest first = most likely discarded)
    all_entries = [
        (abs(val), i, j)
            for (i, b) in enumerate(v.blocks)
            for (j, val) in enumerate(b)
    ]
    sort!(all_entries; by = first, rev = true)

    # Greedily keep until error budget is exhausted
    kept = [Int[] for _ in v.blocks]
    total_err_p = total_norm_p
    for (absval, i, j) in all_entries
        total_err_p -= absval^p * length(gettokenvalue(keys(v.axis), i))
        push!(kept[i], j)
        total_err_p > ϵᵖmax || break
    end
    sort!.(kept)
    return kept
end

# TruncationByError: disambiguate against MAK's findtruncated_svd(::AbstractVector, ::TruncationByError)
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationByError)
    return MAK.findtruncated(v, strategy)
end

# TruncationIntersection: intersect per-block results from each component strategy
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationIntersection)
    inds = map(s -> MAK.findtruncated(v, s), strategy.components)
    return [
        mapreduce(Base.Fix2(getindex, i), MAK._ind_intersect, inds)
            for i in 1:length(v.blocks)
    ]
end
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationIntersection)
    inds = map(s -> MAK.findtruncated_svd(v, s), strategy.components)
    return [
        mapreduce(Base.Fix2(getindex, i), MAK._ind_intersect, inds)
            for i in 1:length(v.blocks)
    ]
end

# truncate for FusedGradedMatrix: build reduced-dimension output, dropping fully
# truncated sectors from the bond side only. For U the row (codomain) axis is the
# input codomain and must keep its full sector set, and only the column (domain)
# axis shrinks. Analogously for Vᴴ. For S, both sides are the bond and shrink
# together. Sectors whose singular values are all truncated to zero are dropped
# entirely from the bond, matching the `truncate_space` convention used in
# `TensorMap` factorizations.
function MAK.truncate(
        ::typeof(MAK.svd_trunc!),
        (U, S, Vᴴ)::NTuple{3, FusedGradedMatrix},
        strategy::MAK.TruncationStrategy
    )
    sv = MAK.diagview(S)
    inds = MAK.findtruncated_svd(sv, strategy)
    sectors_all = collect(keys(U.blocks))

    # Slice every sector's blocks first. `inds[i]` may be `Colon()` (notrunc) or a
    # `Vector{Int}` (rank/tol/error truncations), so check emptiness via the resulting
    # column count rather than `isempty(inds[i])`.
    U_blocks_all =
        [U.blocks[sectors_all[i]][:, inds[i]] for i in eachindex(inds)]
    S_blocks_all = [
        Diagonal(MAK.diagview(S.blocks[sectors_all[i]])[inds[i]])
            for i in eachindex(inds)
    ]
    Vᴴ_blocks_all =
        [Vᴴ.blocks[sectors_all[i]][inds[i], :] for i in eachindex(inds)]

    keep = [i for i in eachindex(inds) if size(U_blocks_all[i], 2) > 0]
    sectors_kept = sectors_all[keep]
    bond_dims = [size(U_blocks_all[i], 2) for i in keep]

    # U: rows = input codomain (full), cols = bond (shrunk).
    U_cod = U.codomain
    U_dom = Dictionary{eltype(sectors_kept), Int}(sectors_kept, bond_dims)
    U_blks = Dictionary{eltype(sectors_kept), eltype(typeof(U.blocks))}(
        sectors_kept, U_blocks_all[keep]
    )
    Ũ = FusedGradedMatrix(U_cod, U_dom, U_blks)

    # S: both sides are the bond (shrunk).
    S_side = Dictionary{eltype(sectors_kept), Int}(sectors_kept, bond_dims)
    S_blks = Dictionary{eltype(sectors_kept), eltype(typeof(S.blocks))}(
        sectors_kept, S_blocks_all[keep]
    )
    S̃ = FusedGradedMatrix(S_side, S_side, S_blks)

    # Vᴴ: rows = bond (shrunk), cols = input domain (full).
    Vᴴ_cod = Dictionary{eltype(sectors_kept), Int}(sectors_kept, bond_dims)
    Vᴴ_dom = Vᴴ.domain
    Vᴴ_blks = Dictionary{eltype(sectors_kept), eltype(typeof(Vᴴ.blocks))}(
        sectors_kept, Vᴴ_blocks_all[keep]
    )
    Ṽᴴ = FusedGradedMatrix(Vᴴ_cod, Vᴴ_dom, Vᴴ_blks)

    return (Ũ, S̃, Ṽᴴ), inds
end

for f! in (:eigh_trunc!, :eig_trunc!)
    @eval function MAK.truncate(
            ::typeof(MAK.$f!),
            (D, V)::NTuple{2, FusedGradedMatrix},
            strategy::MAK.TruncationStrategy
        )
        ev = MAK.diagview(D)
        inds = MAK.findtruncated(ev, strategy)
        sectors = collect(keys(D.blocks))
        D_blocks =
            [Diagonal(MAK.diagview(D.blocks[s])[inds[i]]) for (i, s) in enumerate(sectors)]
        V_blocks = [V.blocks[s][:, inds[i]] for (i, s) in enumerate(sectors)]
        D̃ = FusedGradedMatrix(sectors, D_blocks)
        Ṽ = FusedGradedMatrix(sectors, V_blocks)
        return (D̃, Ṽ), inds
    end
end
