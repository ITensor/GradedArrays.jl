using GradedArrays: SU2, SectorIdentity, SectorRange, U1, dual, isdual, label, labels,
    sector_type, sectoraxes
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "SectorIdentity" begin
    @testset "Construction and eltype" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(1))
        @test si isa SectorIdentity{Float64, TKS.U1Irrep}
        @test eltype(si) == Float64
        @test label(si) == TKS.U1Irrep(1)
    end

    @testset "Construction from SectorRange (dual flag ignored)" begin
        sr = SectorRange(TKS.U1Irrep(1), true)
        si = SectorIdentity{Float64}(sr)
        @test label(si) == TKS.U1Irrep(1)
    end

    @testset "size and axes — U1 (dim=1)" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(0))
        @test size(si) == (1, 1)
        ax1, ax2 = axes(si)
        @test ax1 == SectorRange(TKS.U1Irrep(0), false)
        @test ax2 == SectorRange(TKS.U1Irrep(0), true)
    end

    @testset "size and axes — SU2 j=1/2 (dim=2)" begin
        si = SectorIdentity{Float64}(TKS.SU2Irrep(1 // 2))
        @test size(si) == (2, 2)
        @test axes(si, 1) == SectorRange(TKS.SU2Irrep(1 // 2), false)
        @test axes(si, 2) == SectorRange(TKS.SU2Irrep(1 // 2), true)
    end

    @testset "sectoraxes" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(1))
        @test sectoraxes(si) ==
            (SectorRange(TKS.U1Irrep(1), false), SectorRange(TKS.U1Irrep(1), true))
        @test sectoraxes(si, 1) == SectorRange(TKS.U1Irrep(1), false)
        @test sectoraxes(si, 2) == SectorRange(TKS.U1Irrep(1), true)
    end

    @testset "labels" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(1))
        @test labels(si) == (TKS.U1Irrep(1), TKS.U1Irrep(1))
    end

    @testset "sector_type" begin
        @test sector_type(SectorIdentity{Float64, TKS.U1Irrep}) == SectorRange{TKS.U1Irrep}
    end

    @testset "getindex — identity matrix (U1, 1×1)" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(0))
        @test si[1, 1] == 1.0
    end

    @testset "getindex — identity matrix (SU2 j=1/2, 2×2)" begin
        si = SectorIdentity{Float64}(TKS.SU2Irrep(1 // 2))
        @test si[1, 1] == 1.0
        @test si[1, 2] == 0.0
        @test si[2, 1] == 0.0
        @test si[2, 2] == 1.0
    end

    @testset "copy is identity (structural, no data)" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(1))
        @test copy(si) === si
    end

    @testset "isdual via axes" begin
        si = SectorIdentity{Float64}(TKS.U1Irrep(1))
        @test isdual(si, 1) == false
        @test isdual(si, 2) == true
    end
end
