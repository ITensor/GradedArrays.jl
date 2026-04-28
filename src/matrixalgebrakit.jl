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
    @eval function MAK.default_algorithm(::typeof(MAK.$f!), ::Type{T}; kwargs...) where {T <: FusedGradedMatrix}
        return GradedBlockAlgorithm(MAK.default_algorithm(MAK.$f!, datatype(BlockSparseArrays.blocktype(T)); kwargs...))
    end

    @eval function MAK.copy_input(::typeof(MAK.$f), A::FusedGradedMatrix)
        return FusedGradedMatrix(A.sectors, map(Base.Fix1(MAK.copy_input, MAK.$f), A.blocks))
    end

    @eval function MAK.check_input(::typeof(MAK.$f!), A::FusedGradedMatrix, F::Tuple, alg::GradedBlockAlgorithm)
        for f in F
            A.sectors == f.sectors || throw(ArgumentError("non-matching sectors"))
        end
        return nothing
    end
    @eval function MAK.check_input(::typeof(MAK.$f!), A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
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
    @eval function MAK.initialize_output(::typeof(MAK.$f!), A::FusedGradedMatrix, alg::GradedBlockAlgorithm)
        return FusedGradedMatrix(A.sectors, map(a -> MAK.initialize_output(MAK.$f!, a, alg.alg), A.blocks))
    end
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
        MAK.check_input(MAK.$f!, A, F, alg)
        foreach(A.blocks, F.blocks) do a, f
            f′ = MAK.$f!(a, f, alg.alg)
            _ensure_inplace!(f′, f)
        end
        return F
    end
end

# Single-output: vals functions return FusedGradedVector
for f! in [:svd_vals!, :eig_vals!, :eigh_vals!]
    @eval function MAK.initialize_output(::typeof(MAK.$f!), A::FusedGradedMatrix, alg::GradedBlockAlgorithm)
        return FusedGradedVector(A.sectors, map(a -> MAK.initialize_output(MAK.$f!, a, alg.alg), A.blocks))
    end
    @eval function MAK.$f!(A::FusedGradedMatrix, F, alg::GradedBlockAlgorithm)
        MAK.check_input(MAK.$f!, A, F, alg)
        foreach(A.blocks, F.blocks) do a, f
            f′ = MAK.$f!(a, f, alg.alg)
            _ensure_inplace!(f′, f)
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
    @eval function MAK.initialize_output(::typeof(MAK.$f!), A::FusedGradedMatrix, alg::GradedBlockAlgorithm)
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
            _ensure_inplace!(f′, f)
        end
        return F
    end
end

# Matrix properties
# -----------------
for f in [:isunitary, :isisometric, :is_left_isometric, :is_right_isometric, :ishermitian, :isantihermitian]
    @eval function MAK.$f(A::FusedGradedMatrix; kwargs...)
        return all(x -> MAK.$f(x; kwargs...), A.blocks)
    end
end
