using GradedArrays: GradedArrays, AbelianSectorArray, SU2, SectorIdentity, SectorMatrix,
    SectorOneTo, SectorRange, U1, data, dataaxes, dual, isdual, sector, sector_type,
    sectoraxes, ⊗
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "SectorMatrix" begin
    @testset "Construction from SectorRange + data" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(U1(1), d)
        @test sm isa SectorMatrix{Float64, Matrix{Float64}, U1}
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
        @test sectoraxes(sm) == (U1(1), U1(1)')
        @test sectoraxes(sm, 1) == U1(1)
        @test sectoraxes(sm, 2) == U1(1)'
    end

    @testset "sector returns SectorIdentity" begin
        d = ones(2, 3)
        sm = SectorMatrix(U1(1), d)
        si = sector(sm)
        @test si isa SectorIdentity{Float64, U1}
    end

    @testset "sector_type and datatype" begin
        T = SectorMatrix{Float64, Matrix{Float64}, U1}
        @test sector_type(T) == U1
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
        @test GradedArrays.sector_multiplicity(a1) == 2
        @test GradedArrays.sector_multiplicity(a2) == 3
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
        T = SectorMatrix{Float64, Matrix{Float64}, U1}
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

    @testset "⊗ constructor (SectorIdentity ⊗ data → SectorMatrix)" begin
        si = SectorIdentity{Float64}(U1(1))
        d = [1.0 2.0; 3.0 4.0]
        sm = si ⊗ d
        @test sm isa SectorMatrix
        @test sectoraxes(sm, 1) == U1(1)
        @test data(sm) === d
    end

    @testset "SectorMatrix linear broadcasting" begin
        sm = SectorMatrix(U1(0), [1.0 2.0; 3.0 4.0])

        sm2 = 2.0 .* sm
        @test sm2 isa SectorMatrix
        @test sectoraxes(sm2, 1) == U1(0)
        @test data(sm2) ≈ [2.0 4.0; 6.0 8.0]

        sm3 = sm ./ 2.0
        @test sm3 isa SectorMatrix
        @test data(sm3) ≈ [0.5 1.0; 1.5 2.0]

        sm4 = sm .+ sm
        @test sm4 isa SectorMatrix
        @test data(sm4) ≈ 2.0 .* data(sm)

        sa = AbelianSectorArray((U1(0), U1(0)'), zeros(2, 2))
        sa .= sm
        @test data(sa) ≈ data(sm)
    end
end
