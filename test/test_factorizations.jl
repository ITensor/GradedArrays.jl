import MatrixAlgebraKit as MAK
using GradedArrays: AbelianGradedMatrix, FusedGradedMatrix, FusedGradedVector,
    GradedBlockAlgorithm, U1, Z2, data, dual, eachblockstoredindex, gradedrange
using LinearAlgebra: Diagonal, I, eigvals, isposdef, istril, istriu, lmul!, norm, rmul!
using MatrixAlgebraKit: isisometric, isunitary
using Random: randn!
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra
using Test: @test, @testset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
function isleftnull(N, A; atol::Real = 0, rtol::Real = precision(eltype(A)))
    return isapprox(norm(A' * N), 0; atol = max(atol, norm(A) * rtol))
end

function isrightnull(Nᴴ, A; atol::Real = 0, rtol::Real = precision(eltype(A)))
    return isapprox(norm(A * Nᴴ'), 0; atol = max(atol, norm(A) * rtol))
end

@testset "Factorizations" begin
    rng = StableRNG(1234)

    # -----------------------------------------------------------------------
    # Setup: two test matrices (rectangular and square) with U1 sectors
    # -----------------------------------------------------------------------
    sectors_u1 = [U1(0), U1(1), U1(2)]
    cod_dims_u1 = [3, 4, 2]
    dom_dims_u1 = [2, 3, 5]

    A_rect =
        randn!(rng, FusedGradedMatrix{Float64}(undef, sectors_u1, cod_dims_u1, dom_dims_u1))
    A_tall =
        randn!(rng, FusedGradedMatrix{Float64}(undef, sectors_u1, [4, 5, 3], [2, 3, 2]))
    A_wide =
        randn!(rng, FusedGradedMatrix{Float64}(undef, sectors_u1, [2, 3, 2], [4, 5, 3]))

    sq_dims_u1 = [3, 4, 2]
    A_sq = randn!(rng, FusedGradedMatrix{Float64}(undef, sectors_u1, sq_dims_u1))
    A_herm = randn!(rng, FusedGradedMatrix{Float64}(undef, sectors_u1, sq_dims_u1))
    for I in eachblockstoredindex(A_herm)
        MAK.project_hermitian!(data(view(A_herm, I)))
    end

    # Z2 sectors for variety
    sectors_z2 = [Z2(0), Z2(1)]
    A_z2 = randn!(rng, FusedGradedMatrix{Float64}(undef, sectors_z2, [3, 4]))

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
            @test all(x -> x isa Diagonal, values(S.blocks))

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
            for s in values(S.blocks)
                @test all(isposdef, MAK.diagview(s))
            end
        end

        @testset "vals" begin
            S = MAK.svd_vals(A_rect)
            @test S isa FusedGradedVector
            @test all(b isa AbstractVector for b in values(S.blocks))
            @test all(all(>=(0), b) for b in values(S.blocks))
            # Singular values match those from compact SVD
            _, S2, _ = MAK.svd_compact(A_rect)
            for sec in keys(S.blocks)
                @test isapprox(
                    sort(S.blocks[sec]; rev = true),
                    sort(MAK.diagview(S2.blocks[sec]); rev = true);
                    atol = 1.0e-10
                )
            end
        end
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
            @test all(x -> x isa Diagonal, values(D.blocks))

            # Reconstruction via eigenvector equation
            @test A_sq * V ≈ V * D
        end

        @testset "vals" begin
            D = MAK.eig_vals(A_sq)
            @test D isa FusedGradedVector
            @test collect(keys(D.blocks)) == sectors_u1
            # One eigenvalue per row of each square block
            for (i, sec) in enumerate(keys(D.blocks))
                @test length(D.blocks[sec]) == sq_dims_u1[i]
            end
            # Eigenvalues match diagonal of eig_full
            D2, _ = MAK.eig_full(A_sq)
            for sec in keys(D.blocks)
                @test isapprox(
                    sort(D.blocks[sec]; by = real),
                    sort(MAK.diagview(D2.blocks[sec]); by = real);
                    atol = 1.0e-10
                )
            end
        end
    end

    # -----------------------------------------------------------------------
    @testset "Eigh" begin
        @testset "full" begin
            D, V = MAK.eigh_full(A_herm)
            @test D isa FusedGradedMatrix
            @test V isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, values(D.blocks))

            # Reconstruction
            @test A_herm ≈ V * D * V'

            # Properties
            @test isunitary(V)
        end

        @testset "vals" begin
            D = MAK.eigh_vals(A_herm)
            @test D isa FusedGradedVector
            @test length(keys(D.blocks)) == length(sectors_u1)
            # Eigenvalues should be real and match eigh_full
            D2, _ = MAK.eigh_full(A_herm)
            for sec in keys(D.blocks)
                @test isapprox(
                    sort(real.(D.blocks[sec])),
                    sort(real.(MAK.diagview(D2.blocks[sec])));
                    atol = 1.0e-10
                )
            end
        end
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
    # -----------------------------------------------------------------------
    @testset "Truncated SVD" begin
        using MatrixAlgebraKit: notrunc, truncrank, trunctol, truncerror

        @testset "notrunc" begin
            U, S, Vᴴ, ε = MAK.svd_trunc(A_rect; trunc = notrunc())
            @test U isa FusedGradedMatrix
            @test S isa FusedGradedMatrix
            @test Vᴴ isa FusedGradedMatrix
            @test ε ≈ 0 atol = precision(eltype(A_rect))
            @test A_rect ≈ U * S * Vᴴ
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)

            # same sectors as compact SVD
            U0, S0, Vᴴ0 = MAK.svd_compact(A_rect)
            @test keys(U.blocks) == keys(U0.blocks)
            @test all(
                isapprox(S.blocks[s], S0.blocks[s])
                    for s in keys(S.blocks)
            )
        end

        @testset "truncrank" begin
            maxrank = 4
            U, S, Vᴴ, ε = MAK.svd_trunc(A_rect; trunc = truncrank(maxrank))
            @test U isa FusedGradedMatrix
            # total number of kept singular values ≤ maxrank
            @test sum(size(b, 2) for b in values(U.blocks)) <= maxrank
            # reconstruction error ≈ reported truncation error
            @test norm(A_rect - U * S * Vᴴ) ≈ ε atol = precision(eltype(A_rect))
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)
        end

        @testset "trunctol" begin
            atol = 0.5
            U, S, Vᴴ, ε = MAK.svd_trunc(A_rect; trunc = trunctol(; atol))
            @test U isa FusedGradedMatrix
            # all kept singular values are above the tolerance
            for b in values(S.blocks)
                @test all(≥(atol), MAK.diagview(b))
            end
            @test norm(A_rect - U * S * Vᴴ) ≈ ε atol = precision(eltype(A_rect))
        end

        @testset "truncerror" begin
            atol = 0.3
            U, S, Vᴴ, ε = MAK.svd_trunc(A_rect; trunc = truncerror(; atol))
            @test U isa FusedGradedMatrix
            @test ε <= atol + precision(eltype(A_rect))
            @test norm(A_rect - U * S * Vᴴ) ≈ ε atol = precision(eltype(A_rect))
        end

        @testset "combined (truncrank & trunctol)" begin
            U, S, Vᴴ, ε =
                MAK.svd_trunc(A_rect; trunc = truncrank(3) & trunctol(; atol = 0.3))
            @test U isa FusedGradedMatrix
            @test sum(size(b, 2) for b in values(U.blocks)) <= 3
            for b in values(S.blocks)
                @test all(≥(0.3), MAK.diagview(b))
            end
        end

        @testset "svd_trunc_no_error" begin
            U, S, Vᴴ = MAK.svd_trunc_no_error(A_rect; trunc = truncrank(3))
            @test U isa FusedGradedMatrix
            @test sum(size(b, 2) for b in values(U.blocks)) <= 3
        end

        @testset "drops fully truncated sectors from the bond" begin
            # U1(0) carries singular values of order 1, U1(1) only of order 1e-3,
            # so a tolerance between the two scales removes U1(1) from the bond
            # entirely (not just shrinks it).
            A = FusedGradedMatrix(
                [U1(0), U1(1)], [Matrix(1.0I, 2, 2), 1.0e-3 * Matrix(1.0I, 2, 2)]
            )
            U, S, Vᴴ, ε = MAK.svd_trunc(A; trunc = trunctol(; atol = 1.0e-2))
            @test collect(keys(U.blocks)) == [U1(0)]
            @test collect(keys(S.blocks)) == [U1(0)]
            @test collect(keys(Vᴴ.blocks)) == [U1(0)]
            # The dropped sector's weight shows up as the truncation error.
            @test ε ≈ norm(1.0e-3 * Matrix(1.0I, 2, 2)) atol = precision(eltype(A))
        end
    end

    # -----------------------------------------------------------------------
    @testset "Truncated EIGH" begin
        using MatrixAlgebraKit: notrunc, truncrank, trunctol, truncerror

        @testset "notrunc" begin
            D, V, ε = MAK.eigh_trunc(A_herm; trunc = notrunc())
            @test D isa FusedGradedMatrix
            @test V isa FusedGradedMatrix
            @test ε ≈ 0 atol = precision(eltype(A_herm))
            @test A_herm ≈ V * D * V'
            D0, V0 = MAK.eigh_full(A_herm)
            @test keys(D.blocks) == keys(D0.blocks)
        end

        @testset "truncrank" begin
            maxrank = 5
            D, V, ε = MAK.eigh_trunc(A_herm; trunc = truncrank(maxrank))
            @test D isa FusedGradedMatrix
            @test sum(size(b, 2) for b in values(V.blocks)) <= maxrank
            @test isisometric(V)
        end

        @testset "trunctol (keep largest by abs)" begin
            atol = 0.3
            D, V, ε = MAK.eigh_trunc(A_herm; trunc = trunctol(; atol))
            @test D isa FusedGradedMatrix
            for b in values(D.blocks)
                @test all(≥(atol) ∘ abs, MAK.diagview(b))
            end
        end
    end

    # -----------------------------------------------------------------------
    @testset "matrix multiplication" begin
        rng = StableRNG(1234)
        g = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
        h = gradedrange([U1(0) => 3, U1(1) => 2, U1(2) => 4])
        a = randn(rng, Float64, (g, dual(g)))
        b = randn(rng, Float64, (g, dual(h)))
        c = a * b
        @test c isa AbelianGradedMatrix
        # `(a * b)[i, j] == sum_k a[i, k] * b[k, j]`.
        @test Array(c) ≈ Array(a) * Array(b)
        # Result axes: codomain from `a`, domain from `b`.
        @test axes(c, 1) == axes(a, 1)
        @test axes(c, 2) == axes(b, 2)
    end

    @testset "lmul! / rmul! (block-wise matrix-matrix)" begin
        rng = StableRNG(1234)
        dims = [3, 4, 2]
        Sblocks = [Diagonal(randn(rng, n)) for n in dims]
        S = FusedGradedMatrix(sectors_u1, Sblocks)

        # `lmul!(S, C)`: `C <- S * C` block-wise, `S` square (diagonal, as singular values).
        Cblocks = [randn(rng, dims[i], d) for (i, d) in enumerate([2, 3, 5])]
        C = FusedGradedMatrix(sectors_u1, copy.(Cblocks))
        @test lmul!(S, C) === C
        for (i, s) in enumerate(sectors_u1)
            @test C.blocks[s] ≈ Sblocks[i] * Cblocks[i]
        end

        # `rmul!(A, S)`: `A <- A * S` block-wise.
        Ablocks = [randn(rng, d, dims[i]) for (i, d) in enumerate([2, 3, 5])]
        A = FusedGradedMatrix(sectors_u1, copy.(Ablocks))
        @test rmul!(A, S) === A
        for (i, s) in enumerate(sectors_u1)
            @test A.blocks[s] ≈ Ablocks[i] * Sblocks[i]
        end
    end

    @testset "left_orth / right_orth (SVD path)" begin
        # Passing `trunc` selects the SVD-based orth, which folds the singular values into the
        # returned factor with `lmul!(S, C)` / `rmul!(C, S)` on `FusedGradedMatrix`es.
        V, C = MAK.left_orth(A_rect; trunc = MAK.notrunc())
        @test V isa FusedGradedMatrix
        @test C isa FusedGradedMatrix
        @test isisometric(V)
        @test A_rect ≈ V * C

        Cr, Vᴴ = MAK.right_orth(A_rect; trunc = MAK.notrunc())
        @test Vᴴ isa FusedGradedMatrix
        @test isisometric(Vᴴ; side = :right)
        @test A_rect ≈ Cr * Vᴴ
    end

    # -----------------------------------------------------------------------
    # Bare-matrix factorizations delegate to the matricizing `TensorAlgebra` forms.
    @testset "factorizations on a bare AbelianGradedMatrix" begin
        rng = StableRNG(1234)
        g = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
        h = gradedrange([U1(0) => 3, U1(1) => 2, U1(2) => 4])
        m_rect = randn(rng, Float64, (g, dual(h)))
        m_sq = randn(rng, Float64, (g, dual(g)))
        m_herm = MAK.project_hermitian(randn(rng, Float64, (g, dual(g))))

        # helper: compare two `FusedGradedVector`s block-by-block (the broadcasting `-`
        # path they would otherwise take is not supported).
        fgv_approx(x, y) =
            keys(x.blocks) == keys(y.blocks) &&
            all(x.blocks[k] ≈ y.blocks[k] for k in keys(x.blocks))

        @testset "svd_compact" begin
            U, S, Vᴴ = MAK.svd_compact(m_rect)
            @test all(x -> x isa AbelianGradedMatrix, (U, S, Vᴴ))
            @test axes(U, 1) == axes(m_rect, 1)
            @test axes(Vᴴ, 2) == axes(m_rect, 2)
            @test U * S * Vᴴ ≈ m_rect
            @test Array(U) * Array(S) * Array(Vᴴ) ≈ Array(m_rect)
        end

        @testset "svd_full" begin
            U, S, Vᴴ = MAK.svd_full(m_rect)
            @test all(x -> x isa AbelianGradedMatrix, (U, S, Vᴴ))
            @test Array(U) * Array(S) * Array(Vᴴ) ≈ Array(m_rect)
        end

        @testset "svd_vals" begin
            @test fgv_approx(MAK.svd_vals(m_rect), MAK.svd_vals(FusedGradedMatrix(m_rect)))
        end

        @testset "qr_compact / qr_full" begin
            Q, R = MAK.qr_compact(m_rect)
            @test Array(Q) * Array(R) ≈ Array(m_rect)
            Q, R = MAK.qr_full(m_rect)
            @test Array(Q) * Array(R) ≈ Array(m_rect)
        end

        @testset "lq_compact / lq_full" begin
            L, Q = MAK.lq_compact(m_rect)
            @test Array(L) * Array(Q) ≈ Array(m_rect)
            L, Q = MAK.lq_full(m_rect)
            @test Array(L) * Array(Q) ≈ Array(m_rect)
        end

        @testset "eig_full / eig_vals" begin
            D, V = MAK.eig_full(m_sq)
            @test Array(m_sq) * Array(V) ≈ Array(V) * Array(D)
            @test fgv_approx(MAK.eig_vals(m_sq), MAK.eig_vals(FusedGradedMatrix(m_sq)))
        end

        @testset "eigh_full / eigh_vals" begin
            D, V = MAK.eigh_full(m_herm)
            @test Array(m_herm) ≈ Array(V) * Array(D) * Array(V)'
            @test fgv_approx(
                MAK.eigh_vals(m_herm),
                MAK.eigh_vals(FusedGradedMatrix(m_herm))
            )
        end

        @testset "left_polar / right_polar" begin
            W, P = MAK.left_polar(m_sq)
            @test Array(W) * Array(P) ≈ Array(m_sq)
            P, W = MAK.right_polar(m_sq)
            @test Array(P) * Array(W) ≈ Array(m_sq)
        end

        @testset "project_hermitian" begin
            m = randn(rng, Float64, (g, dual(g)))
            @test Array(MAK.project_hermitian(m)) ≈ (Array(m) + Array(m)') / 2
        end
    end

    # -----------------------------------------------------------------------
    @testset "one! on an AbelianGradedMatrix" begin
        rng = StableRNG(1234)
        g = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
        a = randn(rng, Float64, (g, dual(g)))
        a_before = Array(copy(a))

        b = MAK.one!(a)
        # In place: returns `a` and mutates its contents.
        @test b === a
        @test Array(a) != a_before

        # Same as the existing graded identity constructor and the dense identity.
        id = TensorAlgebra.one(randn(rng, Float64, (g, dual(g))), (1,), (2,))
        @test Array(a) ≈ Array(id)
        @test Array(a) ≈ Matrix(1.0I, size(a)...)
    end
end  # @testset "Factorizations"
