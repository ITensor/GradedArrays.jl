using GradedArrays:
    SU2, SectorIdentity, SectorRange, U1, dual, isdual, sectoraxes, sectortype
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "SectorIdentity" begin
    @testset "Construction and eltype" begin
        si = SectorIdentity{Float64}(U1(1))
        @test si isa SectorIdentity{Float64, U1}
        @test eltype(si) == Float64
    end

    @testset "Construction from dual SectorRange (dual flag preserved)" begin
        si = SectorIdentity{Float64}(U1(1)')
        @test axes(si, 1) == U1(1)'
        @test axes(si, 2) == U1(1)
    end

    @testset "size and axes — U1 (dim=1)" begin
        si = SectorIdentity{Float64}(U1(0))
        @test size(si) == (1, 1)
        ax1, ax2 = axes(si)
        @test ax1 == U1(0)
        @test ax2 == U1(0)'
    end

    @testset "size and axes — SU2 j=1/2 (dim=2)" begin
        si = SectorIdentity{Float64}(SU2(1 // 2))
        @test size(si) == (2, 2)
        @test axes(si, 1) == SU2(1 // 2)
        @test axes(si, 2) == SU2(1 // 2)'
    end

    @testset "axes as sector ranges" begin
        si = SectorIdentity{Float64}(U1(1))
        @test axes(si) == (U1(1), U1(1)')
        @test axes(si, 1) == U1(1)
        @test axes(si, 2) == U1(1)'
    end

    @testset "sectortype" begin
        @test sectortype(SectorIdentity{Float64, U1}) == U1
    end

    @testset "getindex — identity matrix (U1, 1×1)" begin
        si = SectorIdentity{Float64}(U1(0))
        @test si[1, 1] == 1.0
    end

    @testset "getindex — identity matrix (SU2 j=1/2, 2×2)" begin
        si = SectorIdentity{Float64}(SU2(1 // 2))
        @test si[1, 1] == 1.0
        @test si[1, 2] == 0.0
        @test si[2, 1] == 0.0
        @test si[2, 2] == 1.0
    end

    @testset "copy is identity (structural, no data)" begin
        si = SectorIdentity{Float64}(U1(1))
        @test copy(si) === si
    end

    @testset "isdual via axes" begin
        si = SectorIdentity{Float64}(U1(1))
        @test isdual(si, 1) == false
        @test isdual(si, 2) == true
    end
end
