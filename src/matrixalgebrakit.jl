import MatrixAlgebraKit as MAK

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
    ]
    f! = Symbol(f, :!)
    @eval function MAK.default_algorithm(
            ::typeof(MAK.$f!),
            ::Type{T};
            kwargs...
        ) where {T <: FusedGradedMatrix}
        return GradedBlockAlgorithm(
            MAK.default_algorithm(
                MAK.$f!,
                datatype(BlockSparseArrays.blocktype(T));
                kwargs...
            )
        )
    end

    @eval function MAK.copy_input(::typeof(MAK.$f), A::FusedGradedMatrix)
        return FusedGradedMatrix(
            A.sectors,
            map(Base.Fix1(MAK.copy_input, MAK.$f), A.blocks)
        )
    end

    @eval function MAK.check_input(
            ::typeof(MAK.$f!),
            A::FusedGradedMatrix,
            F::Tuple,
            alg::GradedBlockAlgorithm
        )
        for f in F
            A.sectors == f.sectors || throw(ArgumentError("non-matching sectors"))
        end
        return nothing
    end
    @eval function MAK.check_input(
            ::typeof(MAK.$f!),
            A::FusedGradedMatrix,
            F,
            alg::GradedBlockAlgorithm
        )
        A.sectors == F.sectors || throw(ArgumentError("non-matching sectors"))
        return nothing
    end
end

# Generic Implementations
# -----------------------

# in cases where the factorization/alg does not result in in-place, we try to force it by copying.
_ensure_inplace!(F, F′) = F === F′ || copy!(F, F′)
_ensure_inplace!(F::NTuple{N}, F′::NTuple{N}) where {N} = _ensure_inplace!.(F, F′)

# Single-output: null-space functions return FusedGradedMatrix
for f! in [:qr_null!, :lq_null!]
    @eval function MAK.initialize_output(
            ::typeof(MAK.$f!),
            A::FusedGradedMatrix,
            alg::GradedBlockAlgorithm
        )
        return FusedGradedMatrix(
            A.sectors,
            map(a -> MAK.initialize_output(MAK.$f!, a, alg.alg), A.blocks)
        )
    end
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
        MAK.check_input(MAK.$f!, A, F, alg)
        foreach(A.blocks, F.blocks) do a, f
            f′ = MAK.$f!(a, f, alg.alg)
            return _ensure_inplace!(f′, f)
        end
        return F
    end
end

# Single-output: vals functions return FusedGradedVector
for f! in [:svd_vals!, :eig_vals!, :eigh_vals!]
    @eval function MAK.initialize_output(
            ::typeof(MAK.$f!),
            A::FusedGradedMatrix,
            alg::GradedBlockAlgorithm
        )
        return FusedGradedVector(
            A.sectors,
            map(a -> MAK.initialize_output(MAK.$f!, a, alg.alg), A.blocks)
        )
    end
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
        MAK.check_input(MAK.$f!, A, F, alg)
        foreach(A.blocks, F.blocks) do a, f
            f′ = MAK.$f!(a, f, alg.alg)
            return _ensure_inplace!(f′, f)
        end
        return F
    end
end

# Multi-output
for f! in [
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    ]
    @eval function MAK.initialize_output(
            ::typeof(MAK.$f!),
            A::FusedGradedMatrix,
            alg::GradedBlockAlgorithm
        )
        sectors = A.sectors
        blocks = map(a -> MAK.initialize_output(MAK.$f!, a, alg.alg), A.blocks)
        narg = $(startswith(string(f!), "svd") ? 3 : 2)
        return ntuple(narg) do n
            return FusedGradedMatrix(sectors, getindex.(blocks, n))
        end
    end

    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
        MAK.check_input(MAK.$f!, A, F, alg)
        foreach(A.blocks, getproperty.(F, :blocks)...) do a, f...
            f′ = MAK.$f!(a, f, alg.alg)
            return _ensure_inplace!(f′, f)
        end
        return F
    end
end

# Matrix properties
# -----------------
for f in [
        :isunitary,
        :isisometric,
        :is_left_isometric,
        :is_right_isometric,
        :ishermitian,
        :isantihermitian,
    ]
    @eval function MAK.$f(A::FusedGradedMatrix; kwargs...)
        return all(x -> MAK.$f(x; kwargs...), A.blocks)
    end
end

# Truncation support
# ------------------

# diagview for FusedGradedMatrix: extracts per-block diagonals as a FusedGradedVector
function MAK.diagview(m::FusedGradedMatrix)
    return FusedGradedVector(m.sectors, map(MAK.diagview, m.blocks))
end

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
        number_kept += length(v.sectors[i])
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
        total_err_p -= absval^p * length(v.sectors[i])
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
            for i in eachindex(v.blocks)
    ]
end
function MAK.findtruncated_svd(v::FusedGradedVector, strategy::MAK.TruncationIntersection)
    inds = map(s -> MAK.findtruncated_svd(v, s), strategy.components)
    return [
        mapreduce(Base.Fix2(getindex, i), MAK._ind_intersect, inds)
            for i in eachindex(v.blocks)
    ]
end

# truncate for FusedGradedMatrix: build reduced-dimension output, drop empty sectors
# TODO: how do we handle discarde sectors while keeping U and V square in the sectors?
function MAK.truncate(
        ::typeof(MAK.svd_trunc!),
        (U, S, Vᴴ)::NTuple{3, FusedGradedMatrix},
        strategy::MAK.TruncationStrategy
    )
    sv = MAK.diagview(S)
    inds = MAK.findtruncated_svd(sv, strategy)
    U_blocks = map(U.blocks, inds) do u, ind
        return u[:, ind]
    end
    S_blocks = map(S.blocks, inds) do s, ind
        return Diagonal(MAK.diagview(s)[ind])
    end
    Vᴴ_blocks = map(Vᴴ.blocks, inds) do vᴴ, ind
        return vᴴ[ind, :]
    end
    Ũ = FusedGradedMatrix(U.sectors, U_blocks)
    S̃ = FusedGradedMatrix(S.sectors, S_blocks)
    Ṽᴴ = FusedGradedMatrix(Vᴴ.sectors, Vᴴ_blocks)
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
        D_blocks = map(D.blocks, inds) do d, ind
            return Diagonal(MAK.diagview(d)[ind])
        end
        V_blocks = map(V.blocks, inds) do v, ind
            return v[:, ind]
        end
        D̃ = FusedGradedMatrix(D.sectors, D_blocks)
        Ṽ = FusedGradedMatrix(V.sectors, V_blocks)
        return (D̃, Ṽ), inds
    end
end
