using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using TensorAlgebra: MatrixAlgebra as MA

# Length of the main diagonal of a matrix (e.g. the number of singular values a block produces).
diaglength(a::AbstractArray) = minimum(size(a))

struct FusedGradedMatrixAlgorithm{A <: MAK.AbstractAlgorithm} <: MAK.AbstractAlgorithm
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
        return FusedGradedMatrixAlgorithm(
            MAK.default_algorithm(
                MAK.$f!, datatype(T);
                kwargs...
            )
        )
    end

    @eval function MAK.copy_input(::typeof(MAK.$f), A::FusedGradedMatrix)
        return fusedgradedmatrix(
            map(Base.Fix1(MAK.copy_input, MAK.$f), sectordata(A)),
            axis_codomain(A), axis_domain(A)
        )
    end
end

# Bare-matrix factorizations
# ---------------------------
# The plain matrix forms (`MAK.svd_compact(m)`, etc.) route through the matricizing `TensorAlgebra`
# factorizations: matricize to a `FusedGradedMatrix`, run the block factorization, then unmatricize
# back. The factors are returned as graded matrices. This list is shared with the `GradedArray`
# entry points in `gradedarray.jl` (defined there because `GradedArray` is not yet defined here).
# Dispatch must not catch `FusedGradedMatrix`: the matricizing forms produce one, which must
# terminate at its own in-place block algorithm rather than route back here (that would recurse).
# Omitted: `project_antihermitian`/`project_isometric` (no `TensorAlgebra` perm-form) and the
# null-space factorizations, whose `GradedArray` entry points are a follow-up.
const BARE_MATRIX_FACTORIZATIONS = (
    :svd_compact, :svd_full, :svd_vals, :qr_compact, :qr_full, :lq_compact,
    :lq_full, :eig_full, :eig_vals, :eigh_full, :eigh_vals, :left_polar,
    :right_polar, :project_hermitian,
)

# Projections on a fused block (`FusedSectorMatrix`)
# ---------------------------------------------
# The hermitian (antihermitian) projection of a fused block reduces to the same projection of
# its reduced data: by Schur's lemma the block is `I ⊗ data(A)`, with the structural factor
# the identity, and `project(I ⊗ M) = I ⊗ project(M)`, so the projection passes straight to
# the reduced data. This is why it is well defined in the non-abelian case, where the generic
# `AbstractMatrix` path scalar-indexes the block and hits the unique-fusion guard.
#
# `FusedSectorMatrixAlgorithm` wraps the reduced-data algorithm so the block projection dispatches on
# a distinct type (mirroring `FusedGradedMatrixAlgorithm` one level up), forwarding the wrapped inner
# algorithm to the data and staying clear of the generic `AbstractMatrix` projection methods.
struct FusedSectorMatrixAlgorithm{A <: MAK.AbstractAlgorithm} <: MAK.AbstractAlgorithm
    alg::A
end

for f! in (:project_hermitian!, :project_antihermitian!)
    @eval function MAK.default_algorithm(
            ::typeof(MAK.$f!), ::Type{<:FusedSectorMatrix{<:Any, <:Any, D}}; kwargs...
        ) where {D}
        return FusedSectorMatrixAlgorithm(MAK.default_algorithm(MAK.$f!, D; kwargs...))
    end
    @eval function MAK.initialize_output(
            ::typeof(MAK.$f!), A::FusedSectorMatrix, ::FusedSectorMatrixAlgorithm
        )
        return A
    end
    @eval function MAK.$f!(
            A::FusedSectorMatrix,
            out::FusedSectorMatrix,
            alg::FusedSectorMatrixAlgorithm
        )
        MAK.$f!(data(A), data(out), alg.alg)
        return out
    end
end

# Generic Implementations
# -----------------------
# in cases where the factorization/alg does not result in in-place, we try to force it by copying.
_ensure_inplace!(F, F′) = F === F′ || copy!(F, F′)

for f! in (
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    )
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::FusedGradedMatrixAlgorithm)
        $(f! in (:eig_full!, :eigh_full!) && :(LinearAlgebra.checksquare(A)))
        for c in eachsector(A, F...)
            Ac = getsectordata(A, c)
            Fc = map(x -> getsectordata(x, c), F)
            Fc′ = MAK.$f!(Ac, Fc, alg.alg)
            _ensure_inplace!.(Fc, Fc′)
        end
        return F
    end
end

# Handle these separately because single output instead of tuple
for f! in (
        :qr_null!, :lq_null!,
        :svd_vals!, :eig_vals!, :eigh_vals!,
        :project_isometric!,
    )
    @eval function MAK.$f!(A::FusedGradedMatrix, N, alg::FusedGradedMatrixAlgorithm)
        $(f! in (:eig_vals!, :eigh_vals!) && :(LinearAlgebra.checksquare(A)))
        for c in eachsector(A, N)
            Ac = getsectordata(A, c)
            Nc = getsectordata(N, c)
            _ensure_inplace!(Nc, MAK.$f!(Ac, Nc, alg.alg))
        end
        return N
    end
end

# Hermitian/antihermitian projection of a fused matrix is the per-block projection of each
# stored block, reusing the `FusedSectorMatrix` methods above. Both are pure in-place projections
# with the same block structure as the input, so they iterate the stored blocks directly.
for f! in (:project_hermitian!, :project_antihermitian!)
    @eval function MAK.$f!(A::FusedGradedMatrix, out, alg::FusedGradedMatrixAlgorithm)
        LinearAlgebra.checksquare(A)
        for I in eachblockstoredindex(A)
            MAK.$f!(view(A, I), view(out, I), FusedSectorMatrixAlgorithm(alg.alg))
        end
        return out
    end
end

# Boolean output
for f in [
        :isunitary, :isisometric, :is_left_isometric, :is_right_isometric,
        :ishermitian, :isantihermitian,
    ]
    @eval function MAK.$f(A::FusedGradedMatrix; kwargs...)
        return all(x -> MAK.$f(x; kwargs...), sectordata(A))
    end
end

# `one!` (identity fill) is a matrix operation: defined on the matrix storage types by filling each
# stored block, and guarded on the array types (`_matrix_op_error`) so it does not silently
# scalar-index the generic `AbstractMatrix` fallback.
function MAK.one!(A::FusedGradedMatrix)
    for bI in eachblockstoredindex(A)
        MAK.one!(view(A, bI))
    end
    return A
end
MAK.one!(A::FusedSectorMatrix) = (MAK.one!(data(A)); A)
MAK.one!(A::AbstractFusedGradedVector) = _matrix_op_error(MAK.one!, A)
MAK.one!(A::AbstractSectorArray) = _matrix_op_error(MAK.one!, A)
MAK.one!(A::AbstractSectorDelta) = _matrix_op_error(MAK.one!, A)

# initialize_outputs: have to compute the correct sizes for all sectors
# since these might be present or missing
# =====================================================================

# helper: a fresh diagonal factor (SVD singular values, eigenvalues) over the given per-sector
# reduced/bond dimensions. `Diagonal` blocks, backed by a contiguous buffer.
function similar_diagonal(
        A::FusedGradedMatrix,
        ::Type{T},
        bond_dims::FusedGradedOneTo
    ) where {T}
    return FusedGradedDiagonal{T}(undef, bond_dims)
end

# Singular value decomposition
# ----------------------------
function MAK.initialize_output(
        ::typeof(MAK.svd_full!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    U = similar(A, axis_codomain(A), axis_codomain(A))
    S = similar(A, real(eltype(A)), axis_codomain(A), axis_domain(A))
    Vᴴ = similar(A, axis_domain(A), axis_domain(A))
    return U, S, Vᴴ
end
function MAK.initialize_output(
        ::typeof(MAK.svd_compact!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    V_S = FusedGradedOneTo(map(diaglength, sectordata(A)))
    U = similar(A, axis_codomain(A), V_S)
    Tr = real(eltype(A))
    S = similar_diagonal(A, Tr, V_S)
    Vᴴ = similar(A, V_S, axis_domain(A))
    return U, S, Vᴴ
end
function MAK.initialize_output(
        ::typeof(MAK.svd_vals!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    V_S = FusedGradedOneTo(map(diaglength, sectordata(A)))
    Tr = real(eltype(A))
    return similar(A, Vector{Tr}, V_S) # TODO: don't hardcode type
end

# Eigenvalue decomposition
# ------------------------
function MAK.initialize_output(
        ::typeof(MAK.eig_full!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    Tc = complex(eltype(A))
    D = similar_diagonal(A, Tc, axis_domain(A))
    V = similar(A, Tc)
    return D, V
end
function MAK.initialize_output(
        ::typeof(MAK.eig_vals!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    Tc = complex(eltype(A))
    return similar(A, Vector{Tc}, axis_domain(A)) # TODO: don't hardcode type
end

function MAK.initialize_output(
        ::typeof(MAK.eigh_full!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    Tr = real(eltype(A))
    D = similar_diagonal(A, Tr, axis_domain(A))
    V = similar(A)
    return D, V
end
function MAK.initialize_output(
        ::typeof(MAK.eigh_vals!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    Tr = real(eltype(A))
    return similar(A, Vector{Tr}, axis_domain(A)) # TODO: don't hardcode type
end

# QR decomposition
# ----------------
function MAK.initialize_output(
        ::typeof(MAK.qr_full!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    Q = similar(A, axis_codomain(A), axis_codomain(A))
    R = similar(A, axis_codomain(A), axis_domain(A))
    return Q, R
end
function MAK.initialize_output(
        ::typeof(MAK.qr_compact!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    V_Q = FusedGradedOneTo(map(diaglength, sectordata(A)))
    Q = similar(A, axis_codomain(A), V_Q)
    R = similar(A, V_Q, axis_domain(A))
    return Q, R
end
function MAK.initialize_output(
        ::typeof(MAK.qr_null!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    V_N = copy(sectordatalengths(axis_codomain(A)))
    dom = sectordatalengths(axis_domain(A))
    for (c, d₁) in pairs(V_N)
        V_N[c] = max(d₁ - get(dom, c, 0), 0)
    end
    filter!(!iszero, V_N)
    return similar(A, axis_codomain(A), FusedGradedOneTo(V_N))
end

# LQ decomposition
# ----------------
function MAK.initialize_output(
        ::typeof(MAK.lq_full!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    L = similar(A, axis_codomain(A), axis_domain(A))
    Q = similar(A, axis_domain(A), axis_domain(A))
    return L, Q
end
function MAK.initialize_output(
        ::typeof(MAK.lq_compact!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    V_Q = FusedGradedOneTo(map(diaglength, sectordata(A)))
    L = similar(A, axis_codomain(A), V_Q)
    Q = similar(A, V_Q, axis_domain(A))
    return L, Q
end
function MAK.initialize_output(
        ::typeof(MAK.lq_null!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    V_N = copy(sectordatalengths(axis_domain(A)))
    cod = sectordatalengths(axis_codomain(A))
    for (c, d₂) in pairs(V_N)
        V_N[c] = max(d₂ - get(cod, c, 0), 0)
    end
    filter!(!iszero, V_N)
    return similar(A, FusedGradedOneTo(V_N), axis_domain(A))
end

# Polar decomposition
# -------------------
function MAK.initialize_output(
        ::typeof(MAK.left_polar!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    W = similar(A)
    P = similar(A, axis_domain(A), axis_domain(A))
    return W, P
end
function MAK.initialize_output(
        ::typeof(MAK.right_polar!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    P = similar(A, axis_codomain(A), axis_codomain(A))
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
        alg::FusedGradedMatrixAlgorithm
    )
    return A
end
function MAK.initialize_output(
        ::typeof(MAK.project_antihermitian!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    return A
end
function MAK.initialize_output(
        ::typeof(MAK.project_isometric!),
        A::FusedGradedMatrix,
        alg::FusedGradedMatrixAlgorithm
    )
    return similar(A)
end

# Truncation support
# ------------------

function MAK.diagview(m::FusedGradedMatrix)
    return error(
        "`diagview` of a `FusedGradedMatrix` (a write-through view) is not yet supported; use `diag` for a copy of the diagonal"
    )
end

# A `FusedGradedDiagonal` stores its diagonal as a `FusedGradedVector`, so `diagview` returns it
# directly, sharing storage (writing the diagonal writes through to the factor).
MAK.diagview(d::FusedGradedDiagonal) = d.diag

# Inverse of `diagview`: wrap a `FusedGradedVector` as a block-diagonal `FusedGradedDiagonal`,
# sharing storage. Keeps the structured form through
# `pow_diag_safe(D) = MAK.diagonal(map(f, MAK.diagview(D)))`, so the next
# `V * MAK.diagonal(...)` stays in the block-diagonal multiplication path instead of
# falling through to LinearAlgebra's scalar-indexing `Diagonal*Matrix` impl.
MAK.diagonal(v::FusedGradedVector) = FusedGradedDiagonal(v)

# `pow_diag_safe!` for a graded matrix that is diagonal: a `FusedGradedDiagonal`, or a
# `FusedGradedMatrix` that happens to be runtime-diagonal (the `isdiag` fast path in
# `sqrth_invsqrth_safe` powers such a matrix directly, without an eigendecomposition).
# Delegating per reduced block reuses the generic diagonal-only kernel, which is correct even
# in the non-abelian case: a diagonal factor is `Diagonal(λ) ⊗ I` per sector, and `f(A ⊗ I) =
# f(A) ⊗ I`, so the power passes straight to the reduced eigenvalues. This is why the diagonal
# power is well defined here whereas a general element-wise `map!` on a graded array is not.
function MA.pow_diag_safe!(
        Dp::AbstractFusedGradedMatrix, D::AbstractFusedGradedMatrix, p, tol
    )
    for c in eachsector(D)
        MA.pow_diag_safe!(sectordata(Dp, c), sectordata(D, c), p, tol)
    end
    return Dp
end

# Count how many elements are kept for a given index specification and block size
_count_kept(::Colon, n) = n
_count_kept(ind::AbstractVector{Bool}, _) = count(ind)
_count_kept(ind::AbstractVector, _) = length(ind)

# truncation_error! for FusedGradedVector
# Zeroes out kept values (ind[i]) in each block; returns 2-norm of discarded values.
function MAK.truncation_error!(v::FusedGradedVector, ind::AbstractVector)
    foreach(MAK.truncation_error!, sectordata(v), ind)
    return LinearAlgebra.norm(v)
end
function MAK.truncation_error(v::FusedGradedVector, ind::AbstractVector)
    return MAK.truncation_error!(copy(v), ind)
end

# findtruncated / findtruncated_svd for FusedGradedVector
# Both return a Vector where entry i gives the kept indices for block i.

function MAK.findtruncated(v::FusedGradedVector, ::MAK.NoTruncation)
    return [Colon() for _ in sectordata(v)]
end

# Default: findtruncated_svd falls back to findtruncated (overridden below for some strategies)
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationStrategy)
    return MAK.findtruncated(v, strategy)
end
function MAK.findtruncated_svd(v::FusedGradedVector, ::MAK.NoTruncation)
    return [Colon() for _ in sectordata(v)]
end

# TruncationByFilter: apply independently per block
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByFilter)
    return [MAK.findtruncated(b, strategy) for b in sectordata(v)]
end

# TruncationByValue (trunctol): compute global norm for rtol, then apply per block
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * LinearAlgebra.norm(v, strategy.p))
    per_block = MAK.trunctol(; atol, strategy.by, strategy.keep_below, strategy.p)
    return [MAK.findtruncated(b, per_block) for b in sectordata(v)]
end
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationByValue)
    atol = max(strategy.atol, strategy.rtol * LinearAlgebra.norm(v, strategy.p))
    per_block = MAK.trunctol(; atol, strategy.by, strategy.keep_below, strategy.p)
    return [MAK.findtruncated_svd(b, per_block) for b in sectordata(v)]
end

# TruncationByOrder (truncrank k): global top-k across all blocks
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByOrder)
    all_entries = [
        (strategy.by(val), i, j)
            for (i, b) in enumerate(sectordata(v))
            for (j, val) in enumerate(b)
    ]
    sort!(all_entries; by = first, strategy.rev)
    axsectors = sectors(only(axes(v)))
    kept = [Int[] for _ in sectordata(v)]
    number_kept = 0
    for (_, i, j) in all_entries
        number_kept += length(axsectors[i])
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
            for (i, b) in enumerate(sectordata(v))
            for (j, val) in enumerate(b)
    ]
    sort!(all_entries; by = first, rev = true)

    # Greedily keep until error budget is exhausted
    axsectors = sectors(only(axes(v)))
    kept = [Int[] for _ in sectordata(v)]
    total_err_p = total_norm_p
    for (absval, i, j) in all_entries
        total_err_p -= absval^p * length(axsectors[i])
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
            for i in 1:length(sectordata(v))
    ]
end
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationIntersection)
    inds = map(s -> MAK.findtruncated_svd(v, s), strategy.components)
    return [
        mapreduce(Base.Fix2(getindex, i), MAK._ind_intersect, inds)
            for i in 1:length(sectordata(v))
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
        (U, S, Vᴴ)::Tuple{FusedGradedMatrix, FusedGradedDiagonal, FusedGradedMatrix},
        strategy::MAK.TruncationStrategy
    )
    sv = MAK.diagview(S)
    inds = MAK.findtruncated_svd(sv, strategy)
    sectors_all = collect(keys(sectordata(U)))

    # Slice every sector's blocks first. `inds[i]` may be `Colon()` (notrunc) or a
    # `Vector{Int}` (rank/tol/error truncations), so check emptiness via the resulting
    # column count rather than `isempty(inds[i])`.
    U_blocks_all =
        [sectordata(U)[sectors_all[i]][:, inds[i]] for i in eachindex(inds)]
    sv_blocks_all =
        [sectordata(sv)[sectors_all[i]][inds[i]] for i in eachindex(inds)]
    Vᴴ_blocks_all =
        [sectordata(Vᴴ)[sectors_all[i]][inds[i], :] for i in eachindex(inds)]

    keep = [i for i in eachindex(inds) if size(U_blocks_all[i], 2) > 0]
    sectors_kept = sectors_all[keep]
    bond_dims = [size(U_blocks_all[i], 2) for i in keep]

    # U: rows = input codomain (full), cols = bond (shrunk). The sliced blocks are freshly allocated
    # matrices, not buffer views, so let the block dictionary infer their type.
    U_cod = axis_codomain(U)
    U_dom = FusedGradedOneTo(sectors_kept, bond_dims)
    Ũ = fusedgradedmatrix(Dictionary(sectors_kept, U_blocks_all[keep]), U_cod, U_dom)

    # S: the diagonal factor over the shrunk bond.
    S̃ = MAK.diagonal(fusedgradedvector(sectors_kept .=> sv_blocks_all[keep]))

    # Vᴴ: rows = bond (shrunk), cols = input domain (full).
    Vᴴ_cod = FusedGradedOneTo(sectors_kept, bond_dims)
    Vᴴ_dom = axis_domain(Vᴴ)
    Ṽᴴ = fusedgradedmatrix(Dictionary(sectors_kept, Vᴴ_blocks_all[keep]), Vᴴ_cod, Vᴴ_dom)

    return (Ũ, S̃, Ṽᴴ), inds
end

for f! in (:eigh_trunc!, :eig_trunc!)
    @eval function MAK.truncate(
            ::typeof(MAK.$f!),
            (D, V)::Tuple{FusedGradedDiagonal, FusedGradedMatrix},
            strategy::MAK.TruncationStrategy
        )
        ev = MAK.diagview(D)
        inds = MAK.findtruncated(ev, strategy)
        sectors_all = collect(keys(sectordata(D)))

        ev_blocks_all = [sectordata(ev)[sectors_all[i]][inds[i]] for i in eachindex(inds)]
        V_blocks_all = [sectordata(V)[sectors_all[i]][:, inds[i]] for i in eachindex(inds)]

        keep = [i for i in eachindex(inds) if length(ev_blocks_all[i]) > 0]
        sectors_kept = sectors_all[keep]

        D̃ = MAK.diagonal(fusedgradedvector(sectors_kept .=> ev_blocks_all[keep]))
        Ṽ = fusedgradedmatrix(sectors_kept .=> V_blocks_all[keep])
        return (D̃, Ṽ), inds
    end
end
