using BlockArrays: blocklength
using GradedArrays: GradedArrays, GradedOneTo, SU2, SectorOneTo, SectorRange, U1,
    datalength, datalengths, dual, flip, gradedrange, isdual, sector, sectors, sectortype,
    tensor_product
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "SectorOneTo" begin
    @testset "U1 construction and accessors" begin
        si = SectorOneTo(U1(1), 3)
        @test sector(si) == U1(1)
        @test datalength(si) == 3
        @test isdual(si) == false

        # Default multiplicity
        si3 = SectorOneTo(U1(0))
        @test datalength(si3) == 1
        @test isdual(si3) == false
    end

    @testset "U1 dual sector accessor" begin
        si = SectorOneTo(U1(1)', 3)
        @test isdual(si) == true
        @test sector(si) == U1(1)'
    end

    @testset "SU2 construction and accessors" begin
        # SU2 j=1/2: quantum dim = 2
        si = SectorOneTo(SU2(1 // 2), 4)
        @test sector(si) == SU2(1 // 2)
        @test datalength(si) == 4
        @test isdual(si) == false
    end

    @testset "SU2 dual" begin
        si = SectorOneTo(SU2(1)', 2)
        @test sector(si) == SU2(1)'
    end

    @testset "Collection-like interface" begin
        si = SectorOneTo(U1(3), 7)
        @test sectors(si) == [U1(3)]
        @test datalengths(si) == [7]
        @test blocklength(si) == 1
    end

    @testset "length — U1 (abelian, dim=1)" begin
        si = SectorOneTo(U1(5), 3)
        @test length(si) == 1 * 3  # dim * multiplicity
    end

    @testset "length — SU2 (non-abelian)" begin
        # SU2 j=1: TKS.dim(SU2Irrep(1)) == 3
        si = SectorOneTo(SU2(1), 2)
        @test length(si) == 3 * 2  # dim * multiplicity

        # SU2 j=1/2: TKS.dim(SU2Irrep(1//2)) == 2
        si2 = SectorOneTo(SU2(1 // 2), 5)
        @test length(si2) == 2 * 5
    end

    @testset "dual" begin
        si = SectorOneTo(U1(1), 3)
        sid = dual(si)
        @test sector(sid) == U1(1)'
        @test datalength(sid) == 3
        @test isdual(sid) == true
        @test dual(sid) == si  # double dual is identity
    end

    @testset "flip" begin
        si = SectorOneTo(U1(1), 3)
        sif = flip(si)
        @test sector(sif) == flip(U1(1))
        @test datalength(sif) == 3
        @test isdual(sif) == true
    end

    @testset "flip with SU2" begin
        si = SectorOneTo(SU2(1 // 2), 2)
        sif = flip(si)
        @test sector(sif) == flip(SU2(1 // 2))
        @test isdual(sif) == true
        @test datalength(sif) == 2
    end

    @testset "equality" begin
        si1 = SectorOneTo(U1(1), 3)
        si2 = SectorOneTo(U1(1), 3)
        si3 = SectorOneTo(U1(1)', 3)
        si4 = SectorOneTo(U1(2), 3)
        si5 = SectorOneTo(U1(1), 4)

        @test si1 == si2
        @test si1 != si3  # different dual
        @test si1 != si4  # different sector
        @test si1 != si5  # different multiplicity
    end

    @testset "hashing" begin
        si1 = SectorOneTo(U1(1), 3)
        si2 = SectorOneTo(U1(1), 3)
        @test hash(si1) == hash(si2)

        d = Dict(si1 => "hello")
        @test d[si2] == "hello"
    end

    @testset "sectortype" begin
        @test sectortype(SectorOneTo{U1}) == U1
        @test sectortype(SectorOneTo{SU2}) == SU2
    end

    @testset "show" begin
        si = SectorOneTo(U1(1), 3)
        str = sprint(show, si)
        @test contains(str, "SectorOneTo")
        @test contains(str, "3")

        sid = SectorOneTo(U1(1)', 3)
        strd = sprint(show, sid)
        @test endswith(strd, "'")
    end

    @testset "dual sectors accessor for collection interface" begin
        si = SectorOneTo(U1(1)', 3)
        @test sectors(si) == [U1(1)']
    end

    @testset "tensor_product (abelian)" begin
        si0 = SectorOneTo(U1(0), 2)
        si1 = SectorOneTo(U1(1), 3)

        tp = tensor_product(si0, si1)
        @test tp isa SectorOneTo
        @test sector(tp) == U1(1)
        @test datalength(tp) == 6

        @test tensor_product(si1) == si1

        si1d = SectorOneTo(U1(1)', 3)
        tp1 = tensor_product(si1d)
        @test !isdual(tp1)

        si = SectorOneTo(U1(1), 1)
        @test tensor_product(si, si, si) == SectorOneTo(U1(3), 1)
        @test tensor_product(si, si, si, si) == SectorOneTo(U1(4), 1)
    end

    @testset "tensor_product (non-abelian)" begin
        si_half = SectorOneTo(SU2(1 // 2), 1)
        tp = tensor_product(si_half, si_half)
        @test tp isa GradedOneTo
        @test datalengths(tp) == [1, 1]
    end
end
