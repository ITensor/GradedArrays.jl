using GradedArrays: FusedGradedMatrix, U1, Z2
using LinearAlgebra: Diagonal, I, eigvals, norm, istril, istriu, isposdef
import MatrixAlgebraKit as MAK
using MatrixAlgebraKit: isisometric, isunitary
using Test: @test, @testset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function random_fgm(sectors, cod_dims, dom_dims; T = Float64)
    blocks = [randn(T, m, n) for (m, n) in zip(cod_dims, dom_dims)]
    return FusedGradedMatrix(collect(sectors), blocks)
end

function random_hermitian_fgm(sectors, dims; T = Float64)
    blocks = [MAK.project_hermitian!(randn(T, n, n)) for n in dims]
    return FusedGradedMatrix(collect(sectors), blocks)
end

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))
precision(::Type{T}) where {T} = precision(eltype(T))

function has_positive_diagonal(A)
    T = eltype(A)
    return if T <: Real
        all(≥(zero(T)), diagview(A))
    else
        all(≥(zero(real(T))), real(diagview(A))) &&
            all(≈(zero(real(T))), imag(diagview(A)))
    end
end
isleftnull(N, A; atol::Real = 0, rtol::Real = precision(eltype(A))) =
    isapprox(norm(A' * N), 0; atol = max(atol, norm(A) * rtol))

isrightnull(Nᴴ, A; atol::Real = 0, rtol::Real = precision(eltype(A))) =
    isapprox(norm(A * Nᴴ'), 0; atol = max(atol, norm(A) * rtol))

@testset "Factorizations" begin

    # -----------------------------------------------------------------------
    # Setup: two test matrices (rectangular and square) with U1 sectors
    # -----------------------------------------------------------------------
    sectors_u1 = [U1(0), U1(1), U1(2)]
    cod_dims_u1 = [3, 4, 2]
    dom_dims_u1 = [2, 3, 5]

    A_rect = random_fgm(sectors_u1, cod_dims_u1, dom_dims_u1)
    A_tall = random_fgm(sectors_u1, [4, 5, 3], [2, 3, 2])  # tall: m >= n per block
    A_wide = random_fgm(sectors_u1, [2, 3, 2], [4, 5, 3])  # wide: n >= m per block

    sq_dims_u1 = [3, 4, 2]
    A_sq = random_fgm(sectors_u1, sq_dims_u1, sq_dims_u1)
    A_herm = random_hermitian_fgm(sectors_u1, sq_dims_u1)

    # Z2 sectors for variety
    sectors_z2 = [Z2(0), Z2(1)]
    A_z2 = random_fgm(sectors_z2, [3, 4], [3, 4])

    @testset "GradedBlockAlgorithm" begin
        alg = MAK.select_algorithm(MAK.svd_compact!, A_rect)
        @test alg isa GradedBlockAlgorithm
    end

    # -----------------------------------------------------------------------
    @testset "SVD" begin
        @testset "compact" begin
            U, S, Vᴴ = MAK.svd_compact(A_rect)
            @test U isa FusedGradedMatrix
            @test S isa FusedGradedMatrix
            @test Vᴴ isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, S.blocks)

            # Reconstruction
            @test A_rect ≈ U * S * Vᴴ

            # Properties
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)
            @test isposdef(S)
        end

        @testset "full" begin
            U, S, Vᴴ = MAK.svd_full(A_rect)
            @test U isa FusedGradedMatrix
            @test S isa FusedGradedMatrix
            @test Vᴴ isa FusedGradedMatrix

            # Reconstruction
            @test A_rect ≈ U * S * Vᴴ

            # Properties
            @test isunitary(U)
            @test isunitary(Vᴴ)
            for s in S.blocks
                @test all(isposdef, MAK.diagview(s))
            end
        end

        # @testset "vals" begin
        #     S = MAK.svd_vals(A_rect)
        #     @test S isa FusedGradedMatrix
        #     @test all(S.blocks[i] isa Diagonal for i in eachindex(S.blocks))
        #     @test all(all(>=(0), S.blocks[i].diag) for i in eachindex(S.blocks))
        #     # Singular values match those from compact SVD
        #     _, S2, _ = MAK.svd_compact(A_rect)
        #     for i in eachindex(S.blocks)
        #         @test isapprox(sort(S.blocks[i].diag; rev = true), sort(S2.blocks[i].diag; rev = true); atol = 1.0e-10)
        #     end
        # end
    end

    # -----------------------------------------------------------------------
    @testset "QR" begin
        @testset "compact" begin
            Q, R = MAK.qr_compact(A_rect)
            @test Q isa FusedGradedMatrix
            @test R isa FusedGradedMatrix

            # Reconstruction
            @test Q * R ≈ A_rect

            # Properties
            @test isisometric(Q)
            @test istriu(R)

            # TODO: test positive diagonal
        end

        @testset "full" begin
            Q, R = MAK.qr_full(A_rect)
            @test Q isa FusedGradedMatrix
            @test R isa FusedGradedMatrix

            # Reconstruction
            @test Q * R ≈ A_rect

            # Properties
            @test isunitary(Q)
            @test istriu(R)

            # TODO: test positive diagonal
        end

        @testset "null" begin
            # Use tall matrix so null space per block = 0 (m >= n), or wide for non-trivial null
            N = MAK.qr_null(A_tall)
            @test N isa FusedGradedMatrix

            @test isleftnull(N, A_tall)
            @test isisometric(N)
        end
    end


    # -----------------------------------------------------------------------
    @testset "LQ" begin
        @testset "compact" begin
            L, Q = MAK.lq_compact(A_rect)
            @test L isa FusedGradedMatrix
            @test Q isa FusedGradedMatrix

            # Reconstruction
            @test L * Q ≈ A_rect

            # Properties
            @test istril(L)
            @test isisometric(Q; side = :right)

            # TODO: test positive diagonal
        end

        @testset "full" begin
            L, Q = MAK.lq_full(A_rect)
            @test L isa FusedGradedMatrix
            @test Q isa FusedGradedMatrix

            # Reconstruction
            @test L * Q ≈ A_rect

            # Properties
            @test istril(L)
            @test isunitary(Q)

            # TODO: test positive diagonal
        end

        @testset "null" begin
            # Use wide matrix so null space per block is non-trivial (n >= m)
            N = MAK.lq_null(A_wide)
            @test N isa FusedGradedMatrix

            @test isrightnull(N, A_wide)
            @test isisometric(N; side = :right)
        end
    end

    # -----------------------------------------------------------------------
    @testset "Eig" begin
        @testset "full" begin
            D, V = MAK.eig_full(A_sq)
            @test D isa FusedGradedMatrix
            @test V isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, D.blocks)

            # Reconstruction via eigenvector equation
            @test A_sq * V ≈ V * D
        end

        # @testset "vals" begin
        #     # TODO: eig_vals returns per-sector eigenvalue vectors; FusedGradedVector representation pending
        #     D = MAK.eig_vals(A_sq)
        #     @test D isa FusedGradedMatrix
        # end
    end

    # -----------------------------------------------------------------------
    @testset "Eigh" begin
        @testset "full" begin
            D, V = MAK.eigh_full(A_herm)
            @test D isa FusedGradedMatrix
            @test V isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, D.blocks)

            # Reconstruction
            @test A_herm ≈ V * D * V'

            # Properties
            @test isunitary(V)
        end

        # @testset "vals" begin
        #     # TODO: eigh_vals returns per-sector eigenvalue vectors; FusedGradedVector representation pending
        #     D = MAK.eigh_vals(A_herm)
        #     @test D isa FusedGradedMatrix
        # end
    end

    # -----------------------------------------------------------------------
    @testset "Polar" begin
        @testset "left" begin
            W, P = MAK.left_polar(A_sq)
            @test W isa FusedGradedMatrix
            @test P isa FusedGradedMatrix

            # Reconstruction
            @test W * P ≈ A_sq

            # Properties
            @test isunitary(W)
            @test isposdef(P)
        end

        @testset "right" begin
            P, W = MAK.right_polar(A_sq)
            @test P isa FusedGradedMatrix
            @test W isa FusedGradedMatrix

            # Reconstruction
            @test P * W ≈ A_sq

            # Properties
            @test isunitary(W)
            @test isposdef(P)
        end
    end
end  # @testset "Factorizations"
