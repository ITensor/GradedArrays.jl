using BlockArrays: blocklength
using GradedArrays: GradedArrays, GradedIndices, SU2, SectorIndices, SectorRange, U1, dual,
    flip, gradedrange, isdual, label, labels, sector, sector_multiplicities,
    sector_multiplicity, sector_type, sectorrange, sectors, tensor_product
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "SectorIndices" begin
    @testset "U1 construction and accessors" begin
        si = SectorIndices(TKS.U1Irrep(1), 3, false)
        @test label(si) == TKS.U1Irrep(1)
        @test sector_multiplicity(si) == 3
        @test isdual(si) == false
        @test sector(si) == SectorRange(TKS.U1Irrep(1), false)

        # Convenience constructors
        si2 = SectorIndices(TKS.U1Irrep(2), 5)
        @test label(si2) == TKS.U1Irrep(2)
        @test sector_multiplicity(si2) == 5
        @test isdual(si2) == false

        si3 = SectorIndices(TKS.U1Irrep(0))
        @test sector_multiplicity(si3) == 1
        @test isdual(si3) == false
    end

    @testset "U1 dual sector accessor" begin
        si = SectorIndices(TKS.U1Irrep(1), 3, true)
        @test isdual(si) == true
        # For dual, sector should return a dual SectorRange
        @test sector(si) == SectorRange(TKS.U1Irrep(1), true)
    end

    @testset "SU2 construction and accessors" begin
        # SU2 j=1/2: quantum dim = 2
        si = SectorIndices(TKS.SU2Irrep(1 // 2), 4, false)
        @test label(si) == TKS.SU2Irrep(1 // 2)
        @test sector_multiplicity(si) == 4
        @test isdual(si) == false
        @test sector(si) == SectorRange(TKS.SU2Irrep(1 // 2), false)
    end

    @testset "SU2 dual" begin
        si = SectorIndices(TKS.SU2Irrep(1), 2, true)
        @test sector(si) == SectorRange(TKS.SU2Irrep(1), true)
    end

    @testset "Collection-like interface" begin
        si = SectorIndices(TKS.U1Irrep(3), 7, false)
        @test labels(si) == [TKS.U1Irrep(3)]
        @test sectors(si) == [SectorRange(TKS.U1Irrep(3), false)]
        @test sector_multiplicities(si) == [7]
        @test blocklength(si) == 1
    end

    @testset "length — U1 (abelian, dim=1)" begin
        # U1: TKS.dim(U1Irrep(n)) == 1 for all n
        si = SectorIndices(TKS.U1Irrep(5), 3, false)
        @test length(si) == 1 * 3  # dim * multiplicity
    end

    @testset "length — SU2 (non-abelian)" begin
        # SU2 j=1: TKS.dim(SU2Irrep(1)) == 3
        si = SectorIndices(TKS.SU2Irrep(1), 2, false)
        @test length(si) == 3 * 2  # dim * multiplicity

        # SU2 j=1/2: TKS.dim(SU2Irrep(1//2)) == 2
        si2 = SectorIndices(TKS.SU2Irrep(1 // 2), 5, false)
        @test length(si2) == 2 * 5
    end

    @testset "dual" begin
        si = SectorIndices(TKS.U1Irrep(1), 3, false)
        sid = dual(si)
        @test label(sid) == TKS.U1Irrep(1)  # same label
        @test sector_multiplicity(sid) == 3
        @test isdual(sid) == true  # flag flipped
        @test dual(sid) == si  # double dual is identity
    end

    @testset "flip" begin
        si = SectorIndices(TKS.U1Irrep(1), 3, false)
        sif = flip(si)
        @test label(sif) == TKS.dual(TKS.U1Irrep(1))  # label conjugated
        @test sector_multiplicity(sif) == 3
        @test isdual(sif) == true  # flag flipped
    end

    @testset "flip with SU2" begin
        si = SectorIndices(TKS.SU2Irrep(1 // 2), 2, false)
        sif = flip(si)
        @test label(sif) == TKS.dual(TKS.SU2Irrep(1 // 2))
        @test isdual(sif) == true
        @test sector_multiplicity(sif) == 2
    end

    @testset "equality" begin
        si1 = SectorIndices(TKS.U1Irrep(1), 3, false)
        si2 = SectorIndices(TKS.U1Irrep(1), 3, false)
        si3 = SectorIndices(TKS.U1Irrep(1), 3, true)
        si4 = SectorIndices(TKS.U1Irrep(2), 3, false)
        si5 = SectorIndices(TKS.U1Irrep(1), 4, false)

        @test si1 == si2
        @test si1 != si3  # different dual
        @test si1 != si4  # different label
        @test si1 != si5  # different multiplicity
    end

    @testset "hashing" begin
        si1 = SectorIndices(TKS.U1Irrep(1), 3, false)
        si2 = SectorIndices(TKS.U1Irrep(1), 3, false)
        @test hash(si1) == hash(si2)

        # Use in a Dict
        d = Dict(si1 => "hello")
        @test d[si2] == "hello"
    end

    @testset "sector_type" begin
        @test sector_type(SectorIndices{TKS.U1Irrep}) == SectorRange{TKS.U1Irrep}
        @test sector_type(SectorIndices{TKS.SU2Irrep}) == SectorRange{TKS.SU2Irrep}
    end

    @testset "show" begin
        si = SectorIndices(TKS.U1Irrep(1), 3, false)
        str = sprint(show, si)
        @test contains(str, "SectorIndices")
        @test contains(str, "3")

        sid = SectorIndices(TKS.U1Irrep(1), 3, true)
        strd = sprint(show, sid)
        @test endswith(strd, "'")
    end

    @testset "dual sectors accessor for collection interface" begin
        si = SectorIndices(TKS.U1Irrep(1), 3, true)
        # sectors() should return SectorRange with isdual=true
        @test sectors(si) == [SectorRange(TKS.U1Irrep(1), true)]
        # labels() should return the raw label (no conjugation)
        @test labels(si) == [TKS.U1Irrep(1)]
    end

    @testset "tensor_product (abelian)" begin
        si0 = sectorrange(U1(0), 2)
        si1 = sectorrange(U1(1), 3)

        # two-arg
        tp = tensor_product(si0, si1)
        @test tp isa SectorIndices
        @test sector(tp) == U1(1)
        @test sector_multiplicity(tp) == 6

        # single-arg (identity)
        @test tensor_product(si1) == si1

        # single-arg dual (flips)
        si1d = sectorrange(U1(1)', 3)
        tp1 = tensor_product(si1d)
        @test !isdual(tp1)

        # variadic fold
        si = sectorrange(U1(1), 1)
        @test tensor_product(si, si, si) == sectorrange(U1(3), 1)
        @test tensor_product(si, si, si, si) == sectorrange(U1(4), 1)
    end

    @testset "tensor_product (non-abelian)" begin
        # SU2: j=1/2 ⊗ j=1/2 = j=0 ⊕ j=1
        si_half = sectorrange(SU2(TKS.SU2Irrep(1 // 2)), 1)
        tp = tensor_product(si_half, si_half)
        @test tp isa GradedIndices
        @test sector_multiplicities(tp) == [1, 1]
    end
end
