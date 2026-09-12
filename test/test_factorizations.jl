import MatrixAlgebraKit as MAK
using GradedArrays: GradedArrays, FusedGradedDiagonal, FusedGradedMatrix,
    FusedGradedMatrixAlgorithm, FusedGradedVector, GradedArray, SectorRange, U1, Z2, dual,
    fusedgradeddiagonal, fusedgradedmatrix, gradedrange, sectordata
using LinearAlgebra:
    Diagonal, I, diag, eigvals, isposdef, istril, istriu, lmul!, norm, rmul!
using MatrixAlgebraKit: isisometric, isunitary
using Random: randn!
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, bipermutedims, invsqrth_safe, matricize, sqrth_safe
using TensorKitSectors: FermionParity
using Test: @test, @test_throws, @testset

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

    A_rect = randn!(
        rng,
        FusedGradedMatrix{Float64}(
            undef,
            gradedrange(sectors_u1 .=> cod_dims_u1),
            gradedrange(sectors_u1 .=> dom_dims_u1)
        )
    )
    A_tall = randn!(
        rng,
        FusedGradedMatrix{Float64}(
            undef,
            gradedrange(sectors_u1 .=> [4, 5, 3]),
            gradedrange(sectors_u1 .=> [2, 3, 2])
        )
    )
    A_wide = randn!(
        rng,
        FusedGradedMatrix{Float64}(
            undef,
            gradedrange(sectors_u1 .=> [2, 3, 2]),
            gradedrange(sectors_u1 .=> [4, 5, 3])
        )
    )

    sq_dims_u1 = [3, 4, 2]
    A_sq = randn!(
        rng,
        FusedGradedMatrix{Float64}(
            undef,
            gradedrange(sectors_u1 .=> sq_dims_u1),
            gradedrange(sectors_u1 .=> sq_dims_u1)
        )
    )
    A_herm = MAK.project_hermitian!(
        randn!(
            rng,
            FusedGradedMatrix{Float64}(
                undef,
                gradedrange(sectors_u1 .=> sq_dims_u1),
                gradedrange(sectors_u1 .=> sq_dims_u1)
            )
        )
    )

    # Z2 sectors for variety
    sectors_z2 = [Z2(0), Z2(1)]
    A_z2 = randn!(
        rng,
        FusedGradedMatrix{Float64}(
            undef,
            gradedrange(sectors_z2 .=> [3, 4]),
            gradedrange(sectors_z2 .=> [3, 4])
        )
    )

    @testset "FusedGradedMatrixAlgorithm" begin
        alg = MAK.select_algorithm(MAK.svd_compact!, A_rect)
        @test alg isa FusedGradedMatrixAlgorithm
    end

    # -----------------------------------------------------------------------
    @testset "SVD" begin
        @testset "compact" begin
            U, S, Vᴴ = MAK.svd_compact(A_rect)
            @test U isa FusedGradedMatrix
            @test S isa FusedGradedDiagonal
            @test Vᴴ isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, values(sectordata(S)))

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
            for s in values(sectordata(S))
                @test all(isposdef, MAK.diagview(s))
            end
        end

        @testset "vals" begin
            S = MAK.svd_vals(A_rect)
            @test S isa FusedGradedVector
            @test all(b isa AbstractVector for b in values(sectordata(S)))
            @test all(all(>=(0), b) for b in values(sectordata(S)))
            # Singular values match those from compact SVD
            _, S2, _ = MAK.svd_compact(A_rect)
            for sec in keys(sectordata(S))
                @test isapprox(
                    sort(sectordata(S)[sec]; rev = true),
                    sort(MAK.diagview(sectordata(S2)[sec]); rev = true);
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
            @test D isa FusedGradedDiagonal
            @test V isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, values(sectordata(D)))

            # Reconstruction via eigenvector equation
            @test A_sq * V ≈ V * D
        end

        @testset "vals" begin
            D = MAK.eig_vals(A_sq)
            @test D isa FusedGradedVector
            @test collect(keys(sectordata(D))) == sectors_u1
            # One eigenvalue per row of each square block
            for (i, sec) in enumerate(keys(sectordata(D)))
                @test length(sectordata(D)[sec]) == sq_dims_u1[i]
            end
            # Eigenvalues match diagonal of eig_full
            D2, _ = MAK.eig_full(A_sq)
            for sec in keys(sectordata(D))
                @test isapprox(
                    sort(sectordata(D)[sec]; by = real),
                    sort(MAK.diagview(sectordata(D2)[sec]); by = real);
                    atol = 1.0e-10
                )
            end
        end
    end

    # -----------------------------------------------------------------------
    @testset "Eigh" begin
        @testset "full" begin
            D, V = MAK.eigh_full(A_herm)
            @test D isa FusedGradedDiagonal
            @test V isa FusedGradedMatrix
            @test all(x -> x isa Diagonal, values(sectordata(D)))

            # Reconstruction
            @test A_herm ≈ V * D * V'

            # Properties
            @test isunitary(V)
        end

        @testset "vals" begin
            D = MAK.eigh_vals(A_herm)
            @test D isa FusedGradedVector
            @test length(keys(sectordata(D))) == length(sectors_u1)
            # Eigenvalues should be real and match eigh_full
            D2, _ = MAK.eigh_full(A_herm)
            for sec in keys(sectordata(D))
                @test isapprox(
                    sort(real.(sectordata(D)[sec])),
                    sort(real.(MAK.diagview(sectordata(D2)[sec])));
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
            @test S isa FusedGradedDiagonal
            @test Vᴴ isa FusedGradedMatrix
            @test ε ≈ 0 atol = precision(eltype(A_rect))
            @test A_rect ≈ U * S * Vᴴ
            @test isisometric(U)
            @test isisometric(Vᴴ; side = :right)

            # same sectors as compact SVD
            U0, S0, Vᴴ0 = MAK.svd_compact(A_rect)
            @test keys(sectordata(U)) == keys(sectordata(U0))
            @test all(
                isapprox(sectordata(S)[s], sectordata(S0)[s])
                    for s in keys(sectordata(S))
            )
        end

        @testset "truncrank" begin
            maxrank = 4
            U, S, Vᴴ, ε = MAK.svd_trunc(A_rect; trunc = truncrank(maxrank))
            @test U isa FusedGradedMatrix
            # total number of kept singular values ≤ maxrank
            @test sum(size(b, 2) for b in values(sectordata(U))) <= maxrank
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
            for b in values(sectordata(S))
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
            @test sum(size(b, 2) for b in values(sectordata(U))) <= 3
            for b in values(sectordata(S))
                @test all(≥(0.3), MAK.diagview(b))
            end
        end

        @testset "svd_trunc_no_error" begin
            U, S, Vᴴ = MAK.svd_trunc_no_error(A_rect; trunc = truncrank(3))
            @test U isa FusedGradedMatrix
            @test sum(size(b, 2) for b in values(sectordata(U))) <= 3
        end

        @testset "drops fully truncated sectors from the bond" begin
            # U1(0) carries singular values of order 1, U1(1) only of order 1e-3,
            # so a tolerance between the two scales removes U1(1) from the bond
            # entirely (not just shrinks it).
            A = fusedgradedmatrix(
                [U1(0), U1(1)] .=>
                    [Matrix(1.0I, 2, 2), 1.0e-3 * Matrix(1.0I, 2, 2)]
            )
            U, S, Vᴴ, ε = MAK.svd_trunc(A; trunc = trunctol(; atol = 1.0e-2))
            @test collect(keys(sectordata(U))) == [U1(0)]
            @test collect(keys(sectordata(S))) == [U1(0)]
            @test collect(keys(sectordata(Vᴴ))) == [U1(0)]
            # The dropped sector's weight shows up as the truncation error.
            @test ε ≈ norm(1.0e-3 * Matrix(1.0I, 2, 2)) atol = precision(eltype(A))
        end
    end

    # -----------------------------------------------------------------------
    @testset "Truncated EIGH" begin
        using MatrixAlgebraKit: notrunc, truncrank, trunctol, truncerror

        @testset "notrunc" begin
            D, V, ε = MAK.eigh_trunc(A_herm; trunc = notrunc())
            @test D isa FusedGradedDiagonal
            @test V isa FusedGradedMatrix
            @test ε ≈ 0 atol = precision(eltype(A_herm))
            @test A_herm ≈ V * D * V'
            D0, V0 = MAK.eigh_full(A_herm)
            @test keys(sectordata(D)) == keys(sectordata(D0))
        end

        @testset "truncrank" begin
            maxrank = 5
            D, V, ε = MAK.eigh_trunc(A_herm; trunc = truncrank(maxrank))
            @test D isa FusedGradedDiagonal
            @test sum(size(b, 2) for b in values(sectordata(V))) <= maxrank
            @test isisometric(V)
        end

        @testset "trunctol (keep largest by abs)" begin
            atol = 0.3
            D, V, ε = MAK.eigh_trunc(A_herm; trunc = trunctol(; atol))
            @test D isa FusedGradedDiagonal
            for b in values(sectordata(D))
                @test all(≥(atol) ∘ abs, MAK.diagview(b))
            end
        end

        # Truncating every eigenvalue out of a sector leaves that sector in `V`'s row axis as a
        # zero-width block. Deriving the row axis from the surviving blocks instead drops it, and
        # the factor then fails to fuse back to the input's coupled axes.
        @testset "a fully truncated sector stays in the row axis" begin
            A_split = fusedgradedmatrix(
                [U1(0), U1(1)] .=> [[10.0 0.0; 0.0 9.0], [1.0 0.0; 0.0 0.5]]
            )
            D, V, ε = MAK.eigh_trunc(A_split; trunc = truncrank(2))
            @test GradedArrays.sectors(GradedArrays.axis_codomain(V)) ==
                GradedArrays.sectors(GradedArrays.axis_codomain(A_split))
            @test size(V, 1) == size(A_split, 1)
            @test A_split * V ≈ V * D
        end
    end

    # -----------------------------------------------------------------------
    @testset "matrix multiplication" begin
        rng = StableRNG(1234)
        g = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
        h = gradedrange([U1(0) => 3, U1(1) => 2, U1(2) => 4])
        # `*` is a matrix operation, defined on the matricized `FusedGradedMatrix`, not the array.
        a = matricize(randn(rng, Float64, (g,), (g,)))
        b = matricize(randn(rng, Float64, (g,), (h,)))
        c = a * b
        @test c isa FusedGradedMatrix
        # `(a * b)[i, j] == sum_k a[i, k] * b[k, j]`.
        @test Array(c) ≈ Array(a) * Array(b)
        # Result axes: codomain from `a`, domain from `b`.
        @test axes(c, 1) == axes(a, 1)
        @test axes(c, 2) == axes(b, 2)
    end

    @testset "lmul! / rmul! (block-wise matrix-matrix)" begin
        rng = StableRNG(1234)
        dims = [3, 4, 2]
        svals = [randn(rng, n) for n in dims]
        Sblocks = [Diagonal(v) for v in svals]
        S = fusedgradeddiagonal(sectors_u1 .=> svals)

        # `lmul!(S, C)`: `C <- S * C` block-wise, `S` square (diagonal, as singular values).
        Cblocks = [randn(rng, dims[i], d) for (i, d) in enumerate([2, 3, 5])]
        C = fusedgradedmatrix(sectors_u1 .=> copy.(Cblocks))
        @test lmul!(S, C) === C
        for (i, s) in enumerate(sectors_u1)
            @test sectordata(C)[s] ≈ Sblocks[i] * Cblocks[i]
        end

        # `rmul!(A, S)`: `A <- A * S` block-wise.
        Ablocks = [randn(rng, d, dims[i]) for (i, d) in enumerate([2, 3, 5])]
        A = fusedgradedmatrix(sectors_u1 .=> copy.(Ablocks))
        @test rmul!(A, S) === A
        for (i, s) in enumerate(sectors_u1)
            @test sectordata(A)[s] ≈ Ablocks[i] * Sblocks[i]
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
    # Factorizations are matrix operations, defined on the matricized `FusedGradedMatrix`. These
    # exercise the `randn((codomain,), (domain,))` array constructor feeding `matricize` into each
    # factorization and check the dense reconstruction. (The block-level factorization behavior is
    # covered above on directly-built `FusedGradedMatrix`es.)
    @testset "factorizations reconstruct on a matricized FusedGradedMatrix" begin
        rng = StableRNG(1234)
        g = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
        h = gradedrange([U1(0) => 3, U1(1) => 2, U1(2) => 4])
        m_rect = matricize(randn(rng, Float64, (g,), (h,)))
        m_sq = matricize(randn(rng, Float64, (g,), (g,)))
        m_herm = MAK.project_hermitian(matricize(randn(rng, Float64, (g,), (g,))))

        @testset "svd_compact / svd_full" begin
            U, S, Vᴴ = MAK.svd_compact(m_rect)
            @test U isa FusedGradedMatrix
            @test S isa FusedGradedDiagonal
            @test Vᴴ isa FusedGradedMatrix
            @test axes(U, 1) == axes(m_rect, 1)
            @test axes(Vᴴ, 2) == axes(m_rect, 2)
            @test U * S * Vᴴ ≈ m_rect
            U, S, Vᴴ = MAK.svd_full(m_rect)
            @test Array(U) * Array(S) * Array(Vᴴ) ≈ Array(m_rect)
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

        @testset "eig_full / eigh_full" begin
            D, V = MAK.eig_full(m_sq)
            @test Array(m_sq) * Array(V) ≈ Array(V) * Array(D)
            D, V = MAK.eigh_full(m_herm)
            @test Array(m_herm) ≈ Array(V) * Array(D) * Array(V)'
        end

        @testset "left_polar / right_polar" begin
            W, P = MAK.left_polar(m_sq)
            @test Array(W) * Array(P) ≈ Array(m_sq)
            P, W = MAK.right_polar(m_sq)
            @test Array(P) * Array(W) ≈ Array(m_sq)
        end

        @testset "project_hermitian" begin
            m = matricize(randn(rng, Float64, (g,), (g,)))
            @test Array(MAK.project_hermitian(m)) ≈ (Array(m) + Array(m)') / 2
        end
    end

    # -----------------------------------------------------------------------
    @testset "one! on a FusedGradedMatrix" begin
        rng = StableRNG(1234)
        g = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
        a = matricize(randn(rng, Float64, (g,), (g,)))
        a_before = Array(copy(a))

        b = MAK.one!(a)
        # In place: returns `a` and mutates its contents.
        @test b === a
        @test Array(a) != a_before

        # Same as the graded identity from the tensor-form constructor and the dense identity.
        id = TensorAlgebra.one(randn(rng, Float64, (g,), (g,)), (1,), (2,))
        @test Array(a) ≈ Array(id)
        @test Array(a) ≈ Matrix(1.0I, size(a)...)
    end

    # -----------------------------------------------------------------------
    # `sqrth_safe`/`invsqrth_safe` on a bare `FusedGradedDiagonal` must stay diagonal through the
    # leg permutation the factorization interface applies, so the fast `pow_diag_safe` path runs
    # instead of densifying and hitting the dense eigenvalue-power path.
    @testset "sqrth_safe on a FusedGradedDiagonal ($(nameof(typeof(sects[1]))))" for sects in
        (
            [U1(0), U1(1), U1(2)],
            SectorRange.([FermionParity(false), FermionParity(true)]),
        )
        for T in (Float64, ComplexF64)
            dims = [n for n in 2:(length(sects) + 1)]
            svals = [abs.(randn(T, n)) .+ 1 for n in dims]
            S = fusedgradeddiagonal(sects .=> svals)

            @test bipermutedims(S, (1,), (2,)) isa FusedGradedDiagonal

            P = sqrth_safe(S, (1,), (2,))
            Pinv = invsqrth_safe(S, (1,), (2,))
            # The pattern-taking form ends in `unmatricize`, which wraps a diagonal `{1,1}`
            # result up to the tensor-level `GradedArray`. The fast
            # `pow_diag_safe` path still runs, so the backing stays a `FusedGradedDiagonal`
            # rather than densifying to the eigenvalue-power path.
            @test matricize(P) isa FusedGradedDiagonal
            @test matricize(Pinv) isa FusedGradedDiagonal
            for (i, s) in enumerate(sects)
                @test diag(sectordata(matricize(P))[s]) ≈ sqrt.(svals[i])
                @test diag(sectordata(matricize(Pinv))[s]) ≈ inv.(sqrt.(svals[i]))
            end

            # A dense-stored runtime-diagonal matrix takes the same fast path, with
            # `pow_diag_safe!` delegating per block.
            M = FusedGradedMatrix(S)
            Pm = sqrth_safe(M, (1,), (2,))
            @test matricize(Pm) isa FusedGradedMatrix
            for (i, s) in enumerate(sects)
                @test diag(sectordata(matricize(Pm))[s]) ≈ sqrt.(svals[i])
            end
        end
    end
end  # @testset "Factorizations"

# The mismatched-support matrices the two testsets below share: the codomain's U1(2) has no
# domain partner (zero-column blocks in the kernels), and mirrored, the domain's U1(3) has no
# codomain partner (zero-row blocks).
function codomain_only_sector_matrix(rng, elt)
    return randn!(
        rng,
        FusedGradedMatrix{elt}(
            undef,
            gradedrange([U1(0) => 3, U1(1) => 2, U1(2) => 2]),
            gradedrange([U1(0) => 2, U1(1) => 4])
        )
    )
end
function domain_only_sector_matrix(rng, elt)
    return randn!(
        rng,
        FusedGradedMatrix{elt}(
            undef,
            gradedrange([U1(0) => 2, U1(1) => 4]),
            gradedrange([U1(0) => 3, U1(1) => 2, U1(3) => 3])
        )
    )
end

# The factorization kernels co-iterate the sorted sector union of the input and outputs with one
# positional walk per array, substituting a zero-size block for a sector an array lacks. Pin the
# two absent-sector cases against the per-block dense reference: an output sector absent from
# `A` (null spaces, and the square factors of the full forms), and a full reconstruction across
# the mismatched sector sets.
@testset "factorization kernels over differing sector sets (eltype=$elt)" for elt in (
        Float64,
        ComplexF64,
    )
    rng = StableRNG(1234)
    A = codomain_only_sector_matrix(rng, elt)
    codl = GradedArrays.sectordatalengths(GradedArrays.axis_codomain(A))

    N = MAK.qr_null(A)
    @test isleftnull(N, A)
    for (c, n1) in pairs(codl)
        Ac = haskey(sectordata(A), c) ? Matrix(sectordata(A)[c]) : zeros(elt, n1, 0)
        Nc_ref = MAK.qr_null(Ac)
        size(Nc_ref, 2) > 0 || continue
        @test sectordata(N)[c] ≈ Nc_ref
    end

    U, S, Vᴴ = MAK.svd_full(A)
    @test MAK.isunitary(U)
    @test MAK.isunitary(Vᴴ)
    @test Array(U * S * Vᴴ) ≈ Array(A)

    Q, R = MAK.qr_full(A)
    @test MAK.isunitary(Q)
    @test Array(Q * R) ≈ Array(A)

    # The mirrored case: a domain sector the codomain lacks, covered by the right null space.
    B = domain_only_sector_matrix(rng, elt)
    Nᴴ = MAK.lq_null(B)
    @test isrightnull(Nᴴ, B)
    for (c, n2) in pairs(GradedArrays.sectordatalengths(GradedArrays.axis_domain(B)))
        Bc = if haskey(sectordata(B), c)
            Matrix(sectordata(B)[c])
        else
            zeros(elt, 0, n2)
        end
        Nᴴc_ref = MAK.lq_null(Bc)
        size(Nᴴc_ref, 1) > 0 || continue
        @test sectordata(Nᴴ)[c] ≈ Nᴴc_ref
    end
end

# Compact factorizations across mismatched sector supports: a sector on one axis with no
# partner on the other flows through the kernels as zero-size blocks (the kernels' sector
# union spans every participant's axis supports, so the exact `setsectors` covering
# precondition holds), and the factors keep the input's axes.
@testset "compact factorizations over differing sector sets (eltype=$elt)" for elt in (
        Float64,
        ComplexF64,
    )
    rng = StableRNG(1234)
    A = codomain_only_sector_matrix(rng, elt)

    Q, R = MAK.qr_compact(A)
    @test isisometric(Q)
    @test Array(Q * R) ≈ Array(A)
    @test GradedArrays.axis_codomain(Q) == GradedArrays.axis_codomain(A)
    @test GradedArrays.axis_domain(R) == GradedArrays.axis_domain(A)

    U, S, Vᴴ = MAK.svd_compact(A)
    @test isisometric(U)
    @test Array(U * S * Vᴴ) ≈ Array(A)
    @test GradedArrays.axis_codomain(U) == GradedArrays.axis_codomain(A)
    @test GradedArrays.axis_domain(Vᴴ) == GradedArrays.axis_domain(A)

    # Mirrored: a domain sector the codomain lacks (zero-row blocks in the kernels).
    B = domain_only_sector_matrix(rng, elt)
    UB, SB, VBᴴ = MAK.svd_compact(B)
    @test Array(UB * SB * VBᴴ) ≈ Array(B)
    @test GradedArrays.axis_codomain(UB) == GradedArrays.axis_codomain(B)
    @test GradedArrays.axis_domain(VBᴴ) == GradedArrays.axis_domain(B)
end

# `copy_input` re-backs the input as one whole-buffer copy; the eltype rule must stay
# MatrixAlgebraKit's `float(eltype)`, also from an integer input, and the result must never
# share the input's buffer.
@testset "copy_input eltype and non-aliasing" begin
    rng = StableRNG(1234)
    Ai = fusedgradedmatrix([U1(0) => [1 2; 3 4], U1(1) => [5 6; 7 8]])
    for f in (
            MAK.qr_compact, MAK.svd_compact, MAK.lq_compact, MAK.eig_full, MAK.eigh_full,
            MAK.left_polar, MAK.project_hermitian,
        )
        Af = MAK.copy_input(f, Ai)
        @test Af isa FusedGradedMatrix{Float64}
        @test Array(Af) == Array(Ai)
    end
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    Af64 = randn!(rng, FusedGradedMatrix{Float64}(undef, g, g))
    Ac = MAK.copy_input(MAK.svd_compact, Af64)
    @test Ac.buffer !== Af64.buffer
    @test Array(Ac) == Array(Af64)

    # Factorizations from integer input keep the dense output eltypes.
    Q, R = MAK.qr_compact(Ai)
    @test eltype(Q) === Float64 && eltype(R) === Float64
    U, S, Vᴴ = MAK.svd_compact(Ai)
    @test eltype(U) === Float64 && eltype(S) === Float64 && eltype(Vᴴ) === Float64
    D, V = MAK.eig_full(Ai)
    @test eltype(D) === ComplexF64 && eltype(V) === ComplexF64
end

# Pin the `setsectors` contract, which the kernels rely on to line up supports before they
# co-iterate: the one-arg form equalizes the axes' supports, the result is a genuine fused array
# sharing the parent's buffer (the added sectors are zero-length), a `cs` that does not cover an
# axis support throws, the axes rebuilt from one shared `cs` store its label vector itself, and
# positional `sectordata` matches the keyed form on stored sectors and carves a zero-size
# buffer view (not a fresh dense block) on an added one.
@testset "setsectors and positional sectordata" begin
    rng = StableRNG(1234)
    A = randn!(
        rng,
        FusedGradedMatrix{Float64}(
            undef,
            gradedrange([U1(0) => 3, U1(1) => 2, U1(2) => 2]),
            gradedrange([U1(0) => 2, U1(1) => 4])
        )
    )
    w = GradedArrays.setsectors(A)
    @test w isa FusedGradedMatrix
    @test w.buffer === A.buffer
    cod, dom = GradedArrays.axis_codomain(w), GradedArrays.axis_domain(w)
    @test GradedArrays.sectors(cod) == GradedArrays.sectors(dom)
    @test GradedArrays.datalengths(cod) == [3, 2, 2]
    @test GradedArrays.datalengths(dom) == [2, 4, 0]

    for (i, c) in enumerate(collect(keys(sectordata(w))))
        @test sectordata(w, i) == sectordata(w)[c]
    end
    for (i, c) in enumerate(collect(keys(sectordata(A))))
        @test sectordata(A, i) == sectordata(A)[c]
    end

    # The added sector's block is a zero-size view into the shared buffer, the same block type as
    # the stored blocks, not a freshly allocated dense array.
    blk = sectordata(w)[SectorRange(U1(2))]
    @test size(blk) == (2, 0)
    @test !(blk isa Array)
    @test blk isa GradedArrays.datatype(typeof(w))

    # Setting a support that is already in place returns the array itself: `w`'s axis supports
    # both equal its stored sector list, so nothing changes.
    @test GradedArrays.setsectors(w, GradedArrays.sectorsupport(w)) === w

    # Exact semantics: `cs` must cover every axis support; an uncovering `cs` throws.
    @test_throws ArgumentError GradedArrays.setsectors(A, [SectorRange(U1(3))])

    # Co-iteration over an explicit sector union: the set arrays line up with the lenient
    # per-sector reads on a null-kernel case (a sector absent from `A`'s storage).
    N = MAK.qr_null(A)
    cs = GradedArrays.sectorsupport(A, N)
    wA = GradedArrays.setsectors(A, cs)
    wN = GradedArrays.setsectors(N, cs)

    # An axis whose support already equals `cs` is returned as-is; the rebuilt axes store
    # `cs`'s bare label vector itself, one object shared across the participants.
    @test GradedArrays.axis_codomain(wA) === GradedArrays.axis_codomain(A)
    ls = GradedArrays.sectorlabels(GradedArrays.axis_domain(wA))
    @test GradedArrays.sectorlabels(GradedArrays.axis_domain(wN)) === ls

    for (i, c) in enumerate(cs), (x, wx) in ((A, wA), (N, wN))
        ref = if haskey(sectordata(x), c)
            sectordata(x)[c]
        else
            zeros(
                eltype(x), map(ax -> GradedArrays.getsectordatalengths(ax, c), axes(x))
            )
        end
        @test size(sectordata(wx, i)) == size(ref)
        @test sectordata(wx, i) == ref
    end
end
