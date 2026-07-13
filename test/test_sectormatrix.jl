using GradedArrays: GradedArrays, AbelianSectorArray, SU2, SectorIdentity, SectorMatrix,
    SectorOneTo, SectorRange, SectorVector, U1, data, dataaxes, dual, isdual, sector,
    sector_kron, sectoraxes, sectortype
using LinearAlgebra: dot, norm, tr
using Random: randn!
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "SectorMatrix" begin
    @testset "Construction from SectorRange + data" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(U1(1), d)
        @test sm isa SectorMatrix{Float64, U1, Matrix{Float64}}
        @test eltype(sm) == Float64
        @test sectoraxes(sm, 1) == U1(1)
    end

    @testset "data and dataaxes" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(U1(0), d)
        @test data(sm) === d
        @test dataaxes(sm) == axes(d)
    end

    @testset "sectoraxes" begin
        d = ones(2, 3)
        sm = SectorMatrix(U1(1), d)
        @test sectoraxes(sm) == (U1(1), conj(U1(1)))
        @test sectoraxes(sm, 1) == U1(1)
        @test sectoraxes(sm, 2) == conj(U1(1))
    end

    @testset "sector returns SectorIdentity" begin
        d = ones(2, 3)
        sm = SectorMatrix(U1(1), d)
        si = sector(sm)
        @test si isa SectorIdentity{Float64, U1}
    end

    @testset "sectortype and datatype" begin
        T = SectorMatrix{Float64, U1, Matrix{Float64}}
        @test sectortype(T) == U1
        @test GradedArrays.datatype(T) == Matrix{Float64}
    end

    @testset "axes returns SectorOneTo (U1, dim=1)" begin
        d = ones(3, 4)
        sm = SectorMatrix(U1(1), d)
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
        sm = SectorMatrix(SU2(1 // 2), d)
        a1, a2 = axes(sm)
        @test length(a1) == 4
        @test length(a2) == 6
        @test GradedArrays.datalength(a1) == 2
        @test GradedArrays.datalength(a2) == 3
    end

    @testset "size, getindex, setindex!" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(U1(0), d)
        @test size(sm) == (2, 2)
        @test sm[1, 1] == 1.0
        @test sm[2, 1] == 3.0
        sm[1, 2] = 99.0
        @test sm[1, 2] == 99.0
    end

    @testset "copy" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(U1(0), d)
        sm2 = copy(sm)
        @test sectoraxes(sm2) == sectoraxes(sm)
        @test data(sm2) ≈ data(sm)
        sm2[1, 1] = 999.0
        @test sm[1, 1] == 1.0
    end

    @testset "fill!" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(U1(0), d)
        fill!(sm, 0.0)
        @test all(iszero, data(sm))
    end

    @testset "convert" begin
        d = [1 2; 3 4]
        sm = SectorMatrix(U1(0), d)
        T = SectorMatrix{Float64, U1, Matrix{Float64}}
        sm2 = convert(T, sm)
        @test eltype(sm2) == Float64
        @test sm2[1, 1] === 1.0
    end

    @testset "isdual via axes" begin
        d = ones(2, 3)
        sm = SectorMatrix(U1(1), d)
        @test isdual(sm, 1) == false
        @test isdual(sm, 2) == true
    end

    @testset "sector_kron (SectorIdentity, data) → SectorMatrix" begin
        si = SectorIdentity{Float64}(U1(1))
        d = [1.0 2.0; 3.0 4.0]
        sm = sector_kron(si, d)
        @test sm isa SectorMatrix
        @test sectoraxes(sm, 1) == U1(1)
        @test data(sm) === d
    end

    @testset "broadcasting (data-wise, keeps sector)" begin
        sm = SectorMatrix(U1(0), [1.0 2.0; 3.0 4.0])
        r = 2.0 .* sm
        @test r isa SectorMatrix
        @test sectoraxes(r) == sectoraxes(sm)
        @test data(r) == 2.0 .* data(sm)
        s = sm .+ sm
        @test s isa SectorMatrix
        @test sectoraxes(s) == sectoraxes(sm)
        @test data(s) == 2.0 .* data(sm)
        @test_throws ArgumentError sm .* sm
    end

    @testset "Undef constructor (Int dims)" begin
        sm = SectorMatrix{Float64}(undef, U1(0), 3, 4)
        @test sm isa SectorMatrix{Float64, U1, Matrix{Float64}}
        @test size(data(sm)) == (3, 4)
        @test sectoraxes(sm, 1) == U1(0)
    end

    @testset "Undef constructor (AbstractUnitRange dims)" begin
        sm = SectorMatrix{Float64}(undef, U1(1), Base.OneTo(2), Base.OneTo(5))
        @test sm isa SectorMatrix{Float64, U1, Matrix{Float64}}
        @test size(data(sm)) == (2, 5)
        @test sectoraxes(sm, 1) == U1(1)
    end

    @testset "Undef constructor (fully parameterized)" begin
        sm = SectorMatrix{Float64, U1, Matrix{Float64}}(
            undef, U1(0), Base.OneTo(3), Base.OneTo(4)
        )
        @test sm isa SectorMatrix{Float64, U1, Matrix{Float64}}
        @test size(data(sm)) == (3, 4)
    end

    @testset "tr — sector quantum dimension times reduced-data trace" begin
        d = [1.0 2.0; 3.0 4.0]
        @test tr(SectorMatrix(U1(0), d)) == tr(d)         # dim 1
        @test tr(SectorMatrix(SU2(1 // 2), d)) == 2 * tr(d)  # dim 2
    end

    @testset "dot, norm, and dense Array factorize through the structural factor" for s in
        (
            U1(1),
            SU2(1 // 2),
            SU2(1),
        )
        a = SectorMatrix{Float64}(undef, s, 2, 3)
        b = SectorMatrix{Float64}(undef, s, 2, 3)
        randn!(a)
        randn!(b)
        # The inner product factorizes into the sector's quantum-dimension weight and the
        # reduced-data inner product, matching the dense form.
        @test dot(a, b) ≈ length(s) * dot(data(a), data(b))
        @test dot(a, b) ≈ dot(Array(a), Array(b))
        # `Array` densifies the structural factor `I ⊗ reduced` to the full extent (the generic
        # elementwise fallback would scalar-index past the reduced data).
        @test size(Array(a)) == size(a)

        av = SectorVector{Float64}(undef, s, 4)
        bv = SectorVector{Float64}(undef, s, 4)
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
        ab = SectorMatrix(U1(0), [1.0 2.0; 3.0 4.0])
        @test ab[1, 1] == 1.0
        na = SectorMatrix{Float64}(undef, SU2(1), 2, 3)
        @test_throws ErrorException na[1, 1]
        @test_throws ErrorException (na[1, 1] = 0.0)
    end
end
