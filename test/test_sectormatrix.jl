using GradedArrays: GradedArrays, SU2, SectorIdentity, SectorMatrix, SectorOneTo,
    SectorRange, U1, data, dataaxes, dual, isdual, label, labels, sector, sector_type,
    sectoraxes
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "SectorMatrix" begin
    @testset "Construction from label + data" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(TKS.U1Irrep(1), d)
        @test sm isa SectorMatrix{Float64, Matrix{Float64}, TKS.U1Irrep}
        @test eltype(sm) == Float64
        @test label(sm) == TKS.U1Irrep(1)
    end

    @testset "Construction from SectorRange" begin
        sr = SectorRange(TKS.U1Irrep(1), true)
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(sr, d)
        @test label(sm) == TKS.U1Irrep(1)
    end

    @testset "data and dataaxes" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(TKS.U1Irrep(0), d)
        @test data(sm) === d
        @test dataaxes(sm) == axes(d)
    end

    @testset "sectoraxes" begin
        d = ones(2, 3)
        sm = SectorMatrix(TKS.U1Irrep(1), d)
        @test sectoraxes(sm) ==
            (SectorRange(TKS.U1Irrep(1), false), SectorRange(TKS.U1Irrep(1), true))
        @test sectoraxes(sm, 1) == SectorRange(TKS.U1Irrep(1), false)
        @test sectoraxes(sm, 2) == SectorRange(TKS.U1Irrep(1), true)
    end

    @testset "labels" begin
        d = ones(2, 2)
        sm = SectorMatrix(TKS.U1Irrep(1), d)
        @test labels(sm) == (TKS.U1Irrep(1), TKS.U1Irrep(1))
    end

    @testset "sector returns SectorIdentity" begin
        d = ones(2, 3)
        sm = SectorMatrix(TKS.U1Irrep(1), d)
        si = sector(sm)
        @test si isa SectorIdentity{Float64, TKS.U1Irrep}
        @test label(si) == TKS.U1Irrep(1)
    end

    @testset "sector_type and datatype" begin
        T = SectorMatrix{Float64, Matrix{Float64}, TKS.U1Irrep}
        @test sector_type(T) == SectorRange{TKS.U1Irrep}
        @test GradedArrays.datatype(T) == Matrix{Float64}
    end

    @testset "axes returns SectorOneTo (U1, dim=1)" begin
        d = ones(3, 4)
        sm = SectorMatrix(TKS.U1Irrep(1), d)
        a1, a2 = axes(sm)
        @test a1 isa SectorOneTo
        @test a2 isa SectorOneTo
        @test label(a1) == TKS.U1Irrep(1)
        @test isdual(a1) == false
        @test isdual(a2) == true
        @test length(a1) == 3
        @test length(a2) == 4
    end

    @testset "axes returns SectorOneTo (SU2 j=1/2, dim=2)" begin
        # SectorMatrix data is multiplicity-sized. For SU2(1/2) with mult=2 × mult=3:
        d = ones(2, 3)
        sm = SectorMatrix(TKS.SU2Irrep(1 // 2), d)
        a1, a2 = axes(sm)
        # length = quantum_dimension * multiplicity = 2 * 2 = 4, 2 * 3 = 6
        @test length(a1) == 4
        @test length(a2) == 6
        @test GradedArrays.sector_multiplicity(a1) == 2
        @test GradedArrays.sector_multiplicity(a2) == 3
    end

    @testset "size, getindex, setindex!" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(TKS.U1Irrep(0), d)
        @test size(sm) == (2, 2)
        @test sm[1, 1] == 1.0
        @test sm[2, 1] == 3.0
        sm[1, 2] = 99.0
        @test sm[1, 2] == 99.0
    end

    @testset "copy" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(TKS.U1Irrep(0), d)
        sm2 = copy(sm)
        @test label(sm2) == label(sm)
        @test data(sm2) ≈ data(sm)
        sm2[1, 1] = 999.0
        @test sm[1, 1] == 1.0
    end

    @testset "fill!" begin
        d = [1.0 2.0; 3.0 4.0]
        sm = SectorMatrix(TKS.U1Irrep(0), d)
        fill!(sm, 0.0)
        @test all(iszero, data(sm))
    end

    @testset "convert" begin
        d = [1 2; 3 4]
        sm = SectorMatrix(TKS.U1Irrep(0), d)
        T = SectorMatrix{Float64, Matrix{Float64}, TKS.U1Irrep}
        sm2 = convert(T, sm)
        @test eltype(sm2) == Float64
        @test sm2[1, 1] === 1.0
    end

    @testset "isdual via axes" begin
        d = ones(2, 3)
        sm = SectorMatrix(TKS.U1Irrep(1), d)
        @test isdual(sm, 1) == false
        @test isdual(sm, 2) == true
    end

    @testset "⊗ constructor (SectorIdentity ⊗ data → SectorMatrix)" begin
        using GradedArrays: ⊗
        si = SectorIdentity{Float64}(TKS.U1Irrep(1))
        d = [1.0 2.0; 3.0 4.0]
        sm = si ⊗ d
        @test sm isa SectorMatrix
        @test label(sm) == TKS.U1Irrep(1)
        @test data(sm) === d
    end

    @testset "SectorMatrix linear broadcasting" begin
        using GradedArrays: AbelianSectorArray, ⊗
        sm = SectorMatrix(TKS.U1Irrep(0), [1.0 2.0; 3.0 4.0])

        # Scalar multiply preserves SectorMatrix
        sm2 = 2.0 .* sm
        @test sm2 isa SectorMatrix
        @test label(sm2) == TKS.U1Irrep(0)
        @test data(sm2) ≈ [2.0 4.0; 6.0 8.0]

        # Scalar divide preserves SectorMatrix
        sm3 = sm ./ 2.0
        @test sm3 isa SectorMatrix
        @test data(sm3) ≈ [0.5 1.0; 1.5 2.0]

        # Addition of two SectorMatrix preserves SectorMatrix
        sm4 = sm .+ sm
        @test sm4 isa SectorMatrix
        @test data(sm4) ≈ 2.0 .* data(sm)

        # Cross-type broadcast: AbelianSectorArray .= SectorMatrix
        sa = AbelianSectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(0)), (false, true), zeros(2, 2)
        )
        sa .= sm
        @test data(sa) ≈ data(sm)
    end
end
