using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using TensorAlgebra: MatrixAlgebra as MA

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

    # `float` matches `MAK.copy_input` for dense matrices.
    @eval function MAK.copy_input(::typeof(MAK.$f), A::FusedGradedMatrix)
        return copyto!(similar(A, float(eltype(A))), A)
    end
end

# Projections on a fused block (`FusedSectorMatrix`)
# ---------------------------------------------
# The block is `I ⊗ data(A)` and `project(I ⊗ M) = I ⊗ project(M)`, so the (anti)hermitian
# projection acts on the reduced data, also for non-abelian sectors.
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
# Copy when the factorization did not write in place.
_ensure_inplace!(F, F′) = F === F′ || copy!(F, F′)

for f! in (
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    )
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::FusedGradedMatrixAlgorithm)
        $(f! in (:eig_full!, :eigh_full!) && :(checksquare(A)))
        wA, wFs... = promote_sectorsupport(A, F...)
        for i in 1:blocklength(axis_codomain(wA))
            Fi = map(w -> sectordata(w, i), wFs)
            Fi′ = MAK.$f!(sectordata(wA, i), Fi, alg.alg)
            # `foreach` rather than a broadcast, which over these small union-typed tuples costs
            # more than factorizing a small block.
            foreach(_ensure_inplace!, Fi, Fi′)
        end
        return F
    end
end

# These have a single output rather than a tuple.
for f! in (
        :qr_null!, :lq_null!,
        :svd_vals!, :eig_vals!, :eigh_vals!,
        :project_isometric!,
    )
    @eval function MAK.$f!(A::FusedGradedMatrix, N, alg::FusedGradedMatrixAlgorithm)
        $(f! in (:eig_vals!, :eigh_vals!) && :(checksquare(A)))
        wA, wN = promote_sectorsupport(A, N)
        for i in 1:blocklength(axis_codomain(wA))
            Ni = sectordata(wN, i)
            _ensure_inplace!(Ni, MAK.$f!(sectordata(wA, i), Ni, alg.alg))
        end
        return N
    end
end

for f! in (:project_hermitian!, :project_antihermitian!)
    @eval function MAK.$f!(A::FusedGradedMatrix, out, alg::FusedGradedMatrixAlgorithm)
        checksquare(A)
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

# `one!` is a matrix operation, so it errors on the other types rather than scalar-indexing.
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

# The bond axis of a null space: per sector, the multiplicity of `axis` minus that of `other`,
# omitting the sectors where it is not positive.
function nullspace_axis(axis::FusedGradedOneTo, other::FusedGradedOneTo)
    ss = sectortype(axis)[]
    lens = Int[]
    for (s, d) in zip(sectors(axis), datalengths(axis))
        n = d - getsectordatalengths(other, s)
        if n > 0
            push!(ss, s)
            push!(lens, n)
        end
    end
    return FusedGradedOneTo(ss, lens)
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
    V_N = nullspace_axis(axis_codomain(A), axis_domain(A))
    return similar(A, axis_codomain(A), V_N)
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
    V_N = nullspace_axis(axis_domain(A), axis_codomain(A))
    return similar(A, V_N, axis_domain(A))
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
# As for dense matrices, hermitian and antihermitian projections act in place and the isometric
# projection writes to a fresh output.
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

MAK.diagview(d::FusedGradedDiagonal) = d.diag

# Keeps the result block diagonal, so a following `V * MAK.diagonal(...)` does not fall back to
# the scalar-indexing `Diagonal * Matrix` method.
MAK.diagonal(v::FusedGradedVector) = FusedGradedDiagonal(v)

# For a diagonal graded matrix, which may be a `FusedGradedMatrix` that is diagonal at run time.
# Each block is `Diagonal(λ) ⊗ I` and `f(A ⊗ I) = f(A) ⊗ I`, so the power acts on the reduced data,
# also for non-abelian sectors.
function MA.pow_diag_safe!(
        Dp::AbstractFusedGradedMatrix, D::AbstractFusedGradedMatrix, p, tol
    )
    dDp, dD = sectordata(Dp), sectordata(D)
    for c in keys(dD)
        MA.pow_diag_safe!(dDp[c], dD[c], p, tol)
    end
    return Dp
end

_count_kept(::Colon, n) = n
_count_kept(ind::AbstractVector{Bool}, _) = count(ind)
_count_kept(ind::AbstractVector, _) = length(ind)

# Zeroes out the kept values in each block and returns the 2-norm of the discarded ones.
function MAK.truncation_error!(v::FusedGradedVector, ind::AbstractVector)
    foreach(MAK.truncation_error!, sectordata(v), ind)
    return LinearAlgebra.norm(v)
end
function MAK.truncation_error(v::FusedGradedVector, ind::AbstractVector)
    return MAK.truncation_error!(copy(v), ind)
end

# `findtruncated` returns a `Vector` whose entry `i` is the kept indices of block `i`.

function MAK.findtruncated(v::FusedGradedVector, ::MAK.NoTruncation)
    return [Colon() for _ in sectordata(v)]
end

function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationStrategy)
    return MAK.findtruncated(v, strategy)
end
function MAK.findtruncated_svd(v::FusedGradedVector, ::MAK.NoTruncation)
    return [Colon() for _ in sectordata(v)]
end

function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByFilter)
    return [MAK.findtruncated(b, strategy) for b in sectordata(v)]
end

# `rtol` is relative to the norm of all blocks together.
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

# The top `k` across all blocks.
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
# Values are sorted within each block but still need to be compared across blocks.
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationByOrder)
    return MAK.findtruncated(v, strategy)
end

# One error budget across all blocks, discarding the smallest values first.
function MAK.findtruncated(v::FusedGradedVector, strategy::MAK.TruncationByError)
    (isfinite(strategy.p) && strategy.p > 0) ||
        throw(ArgumentError(lazy"p-norm with p=$(strategy.p) not supported"))
    p = strategy.p
    total_norm_p = LinearAlgebra.norm(v, strategy.p)^p
    ϵᵖmax = max(strategy.atol^p, strategy.rtol^p * total_norm_p)

    all_entries = [
        (abs(val), i, j)
            for (i, b) in enumerate(sectordata(v))
            for (j, val) in enumerate(b)
    ]
    sort!(all_entries; by = first, rev = true)

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

# Disambiguate against `MAK.findtruncated_svd(::AbstractVector, ::TruncationByError)`.
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationByError)
    return MAK.findtruncated(v, strategy)
end

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

# Sectors whose singular values are all truncated are dropped from the bond, as TensorKit's
# `truncate_space` does. The other axes of `U` and `Vᴴ` keep all of their sectors.
function MAK.truncate(
        ::typeof(MAK.svd_trunc!),
        (U, S, Vᴴ)::Tuple{FusedGradedMatrix, FusedGradedDiagonal, FusedGradedMatrix},
        strategy::MAK.TruncationStrategy
    )
    sv = MAK.diagview(S)
    inds = MAK.findtruncated_svd(sv, strategy)
    sdU, sdsv, sdVᴴ = sectordata(U), sectordata(sv), sectordata(Vᴴ)
    sectors_all = collect(keys(sdU))

    # `inds[i]` may be `Colon()`, so check emptiness on the sliced blocks.
    U_blocks_all =
        [sdU[sectors_all[i]][:, inds[i]] for i in eachindex(inds)]
    sv_blocks_all =
        [sdsv[sectors_all[i]][inds[i]] for i in eachindex(inds)]
    Vᴴ_blocks_all =
        [sdVᴴ[sectors_all[i]][inds[i], :] for i in eachindex(inds)]

    keep = [i for i in eachindex(inds) if size(U_blocks_all[i], 2) > 0]
    sectors_kept = sectors_all[keep]
    bond_dims = [size(U_blocks_all[i], 2) for i in keep]

    U_cod = axis_codomain(U)
    U_dom = FusedGradedOneTo(sectors_kept, bond_dims)
    Ũ = fusedgradedmatrix(Dictionary(sectors_kept, U_blocks_all[keep]), U_cod, U_dom)

    S̃ = MAK.diagonal(fusedgradedvector(sectors_kept .=> sv_blocks_all[keep]))

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
        sdev, sdV = sectordata(ev), sectordata(V)
        sectors_all = collect(keys(sectordata(D)))

        ev_blocks_all = [sdev[sectors_all[i]][inds[i]] for i in eachindex(inds)]
        V_blocks_all = [sdV[sectors_all[i]][:, inds[i]] for i in eachindex(inds)]

        keep = [i for i in eachindex(inds) if length(ev_blocks_all[i]) > 0]
        sectors_kept = sectors_all[keep]
        bond_dims = [length(ev_blocks_all[i]) for i in keep]

        D̃ = MAK.diagonal(fusedgradedvector(sectors_kept .=> ev_blocks_all[keep]))

        # The row axis comes from `V`, since deriving it from the kept blocks would drop the
        # sectors whose eigenvalues are all truncated.
        Ṽ = fusedgradedmatrix(
            Dictionary(sectors_kept, V_blocks_all[keep]),
            axis_codomain(V),
            FusedGradedOneTo(sectors_kept, bond_dims)
        )
        return (D̃, Ṽ), inds
    end
end
