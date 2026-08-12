using GradedArrays: GradedArrays, FusedSectorMatrix, FusedSectorVector, SU2, SectorIdentity,
    SectorOneTo, SectorRange, U1, UniqueSectorArray, data, dataaxes, dual, isdual, sector,
    sector_kron, sectoraxes, sectortype, with_scalar_indexing
using LinearAlgebra: dot, norm, tr
using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using Random: randn!
using StableRNGs: StableRNG
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "FusedSectorMatrix" begin
    @testset "Construction from SectorRange + data" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = FusedSectorMatrix(d, U1(1))
        @test sm isa FusedSectorMatrix{Float64, U1, Matrix{Float64}}
        @test eltype(sm) == Float64
        @test sectoraxes(sm, 1) == U1(1)
    end

    @testset "data and dataaxes" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = FusedSectorMatrix(d, U1(0))
        @test data(sm) === d
        @test dataaxes(sm) == axes(d)
    end

    @testset "sectoraxes" begin
        d = ones(2, 3)
        sm = FusedSectorMatrix(d, U1(1))
        @test sectoraxes(sm) == (U1(1), conj(U1(1)))
        @test sectoraxes(sm, 1) == U1(1)
        @test sectoraxes(sm, 2) == conj(U1(1))
    end

    @testset "sector returns SectorIdentity" begin
        d = ones(2, 3)
        sm = FusedSectorMatrix(d, U1(1))
        si = sector(sm)
        @test si isa SectorIdentity{Float64, U1}
    end

    @testset "sectortype and datatype" begin
        T = FusedSectorMatrix{Float64, U1, Matrix{Float64}}
        @test sectortype(T) == U1
        @test GradedArrays.datatype(T) == Matrix{Float64}
    end

    @testset "axes returns SectorOneTo (U1, dim=1)" begin
        d = ones(3, 4)
        sm = FusedSectorMatrix(d, U1(1))
        a1, a2 = axes(sm)
        @test a1 isa SectorOneTo
        @test a2 isa SectorOneTo
        @test sector(a1) == U1(1)
        @test isdual(a1) == false
        @test isdual(a2) == true
        @test length(a1) == 3
        @test length(a2) == 4
    end

    @testset "axes returns SectorOneTo (SU2 j=1/2, dim=2)" begin
        d = ones(2, 3)
        sm = FusedSectorMatrix(d, SU2(1 // 2))
        a1, a2 = axes(sm)
        @test length(a1) == 4
        @test length(a2) == 6
        @test GradedArrays.datalength(a1) == 2
        @test GradedArrays.datalength(a2) == 3
    end

    @testset "size, getindex, setindex!" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = FusedSectorMatrix(d, U1(0))
        @test size(sm) == (2, 2)
        with_scalar_indexing() do
            @test sm[1, 1] == 1.0
            @test sm[2, 1] == 3.0
            sm[1, 2] = 99.0
            @test sm[1, 2] == 99.0
        end
    end

    @testset "copy" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = FusedSectorMatrix(d, U1(0))
        sm2 = copy(sm)
        @test sectoraxes(sm2) == sectoraxes(sm)
        @test data(sm2) ≈ data(sm)
        with_scalar_indexing() do
            sm2[1, 1] = 999.0
            @test sm[1, 1] == 1.0
        end
    end

    @testset "fill!" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = FusedSectorMatrix(d, U1(0))
        fill!(sm, 0.0)
        @test all(iszero, data(sm))
    end

    @testset "convert" begin
        d = [1 2; 3 4]
        sm = FusedSectorMatrix(d, U1(0))
        T = FusedSectorMatrix{Float64, U1, Matrix{Float64}}
        sm2 = convert(T, sm)
        @test eltype(sm2) == Float64
        with_scalar_indexing() do
            @test sm2[1, 1] === 1.0
        end
    end

    @testset "isdual via axes" begin
        d = ones(2, 3)
        sm = FusedSectorMatrix(d, U1(1))
        @test isdual(axes(sm, 1)) == false
        @test isdual(axes(sm, 2)) == true
    end

    @testset "sector_kron (SectorIdentity, data) → FusedSectorMatrix" begin
        si = SectorIdentity{Float64}(U1(1))
        d = [1.0 2.0; 3.0 4.0]
        sm = sector_kron(si, d)
        @test sm isa FusedSectorMatrix
        @test sectoraxes(sm, 1) == U1(1)
        @test data(sm) === d
    end

    @testset "broadcasting (data-wise, keeps sector)" begin
        sm = FusedSectorMatrix([1.0 2.0; 3.0 4.0], U1(0))
        r = 2.0 .* sm
        @test r isa FusedSectorMatrix
        @test sectoraxes(r) == sectoraxes(sm)
        @test data(r) == 2.0 .* data(sm)
        s = sm .+ sm
        @test s isa FusedSectorMatrix
        @test sectoraxes(s) == sectoraxes(sm)
        @test data(s) == 2.0 .* data(sm)
        @test_throws ArgumentError sm .* sm
    end

    @testset "real / imag (data-wise, keeps sector; $name)" for (name, s) in (
            ("U1", U1(1)),
            ("SU2", SU2(1 // 2)),
        )
        d = randn(ComplexF64, 2, 2)
        sm = FusedSectorMatrix(d, s)
        rm = real(sm)
        imm = imag(sm)
        @test rm isa FusedSectorMatrix
        @test imm isa FusedSectorMatrix
        # The structural sector factor is left intact; only the reduced data takes real/imag parts.
        @test sectoraxes(rm) == sectoraxes(sm)
        @test sectoraxes(imm) == sectoraxes(sm)
        @test data(rm) == real.(d)
        @test data(imm) == imag.(d)
        # `real(sm) + im * imag(sm)` reconstructs the data.
        @test data(rm) + im * data(imm) ≈ d
    end

    @testset "Undef constructor (Int dims)" begin
        sm = FusedSectorMatrix{Float64}(undef, U1(0), 3, 4)
        @test sm isa FusedSectorMatrix{Float64, U1, Matrix{Float64}}
        @test size(data(sm)) == (3, 4)
        @test sectoraxes(sm, 1) == U1(0)
    end

    @testset "Undef constructor (AbstractUnitRange dims)" begin
        sm = FusedSectorMatrix{Float64}(undef, U1(1), Base.OneTo(2), Base.OneTo(5))
        @test sm isa FusedSectorMatrix{Float64, U1, Matrix{Float64}}
        @test size(data(sm)) == (2, 5)
        @test sectoraxes(sm, 1) == U1(1)
    end

    @testset "Undef constructor (fully parameterized)" begin
        sm = FusedSectorMatrix{Float64, U1, Matrix{Float64}}(
            undef, U1(0), Base.OneTo(3), Base.OneTo(4)
        )
        @test sm isa FusedSectorMatrix{Float64, U1, Matrix{Float64}}
        @test size(data(sm)) == (3, 4)
    end

    @testset "conj is disallowed (would flip the first axis to dual)" begin
        sm = FusedSectorMatrix([1.0 2.0; 3.0 4.0], U1(1))
        @test_throws ErrorException conj(sm)
        sv = FusedSectorVector{Float64}(undef, U1(1), 4)
        @test_throws ErrorException conj(sv)
    end

    @testset "tr — sector quantum dimension times reduced-data trace" begin
        d = [1.0 2.0; 3.0 4.0]
        @test tr(FusedSectorMatrix(d, U1(0))) == tr(d)         # dim 1
        @test tr(FusedSectorMatrix(d, SU2(1 // 2))) == 2 * tr(d)  # dim 2
    end

    @testset "reductions match the dense block (quantum dimension folded in)" for s in
        (
            U1(1),
            SU2(1 // 2),
            SU2(1),
        )
        d = [1.0 2.0; 3.0 4.0]
        a = FusedSectorMatrix(d, s)
        @test sum(a) == length(s) * sum(d)     # dim copies of the reduced data
        @test sum(a) == sum(Array(a))
        @test maximum(a) == maximum(Array(a))  # folds a structural zero when dim > 1
        @test minimum(a) == minimum(Array(a))
        @test extrema(a) == extrema(Array(a))
        @test maximum(abs, a) == maximum(abs, Array(a))
        @test_throws ErrorException sum(x -> x + 1, a)  # `sum` requires zero-preserving `f` for now
    end

    @testset "dot, norm, and dense Array factorize through the structural factor" for s in
        (
            U1(1),
            SU2(1 // 2),
            SU2(1),
        )
        a = FusedSectorMatrix{Float64}(undef, s, 2, 3)
        b = FusedSectorMatrix{Float64}(undef, s, 2, 3)
        randn!(a)
        randn!(b)
        # The inner product factorizes into the sector's quantum-dimension weight and the
        # reduced-data inner product, matching the dense form.
        @test dot(a, b) ≈ length(s) * dot(data(a), data(b))
        @test dot(a, b) ≈ dot(Array(a), Array(b))
        # `Array` densifies the structural factor `I ⊗ reduced` to the full extent (the generic
        # elementwise fallback would scalar-index past the reduced data).
        @test size(Array(a)) == size(a)

        av = FusedSectorVector{Float64}(undef, s, 4)
        bv = FusedSectorVector{Float64}(undef, s, 4)
        randn!(av)
        randn!(bv)
        @test dot(av, bv) ≈ length(s) * dot(data(av), data(bv))
        @test dot(av, bv) ≈ dot(Array(av), Array(bv))
        @test length(Array(av)) == length(av)

        # The `p`-norm factorizes through the Kronecker structure and matches the dense form for
        # every `p`, `Inf` included.
        for p in (1, 2, 3, Inf)
            @test norm(a, p) ≈ norm(Array(a), p)
            @test norm(av, p) ≈ norm(Array(av), p)
        end
    end

    @testset "scalar indexing requires unique fusion" begin
        ab = FusedSectorMatrix([1.0 2.0; 3.0 4.0], U1(0))
        na = FusedSectorMatrix{Float64}(undef, SU2(1), 2, 3)
        with_scalar_indexing() do
            @test ab[1, 1] == 1.0
            @test_throws ErrorException na[1, 1]
            @test_throws ErrorException (na[1, 1] = 0.0)
        end
    end

    # The projection acts on the reduced data and factorizes through the structural identity,
    # so it is well defined even in the non-abelian case where scalar indexing is not.
    @testset "project_hermitian!/project_antihermitian! project the reduced data" for s in
        (
            U1(1),
            SU2(1 // 2),
        )
        rng = StableRNG(1234)

        a = randn!(rng, FusedSectorMatrix{Float64}(undef, s, 3, 3))
        d = copy(data(a))
        @test MAK.project_hermitian!(a) === a
        @test data(a) ≈ (d + d') / 2
        @test Array(a) ≈
            (Array(FusedSectorMatrix(d, s)) + Array(FusedSectorMatrix(d, s))') / 2

        b = randn!(rng, FusedSectorMatrix{Float64}(undef, s, 3, 3))
        d2 = copy(data(b))
        @test MAK.project_antihermitian!(b) === b
        @test data(b) ≈ (d2 - d2') / 2
    end
end
