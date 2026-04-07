using BlockArrays: blocklength
using GradedArrays: GradedArrays, GradedIndices, SU2, SectorRange, U1, dual, flip,
    gradedrange, isdual, labels, sector_multiplicities, sector_type, sectors, tensor_product
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "GradedIndices" begin
    @testset "gradedrange from raw TKS.Sector labels (U1)" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        @test g isa GradedIndices{TKS.U1Irrep}
        @test labels(g) == [TKS.U1Irrep(0), TKS.U1Irrep(1)]
        @test sector_multiplicities(g) == [2, 3]
        @test isdual(g) == false
    end

    @testset "gradedrange from SectorRange labels (U1)" begin
        g = GradedArrays.gradedrange([U1(0) => 2, U1(1) => 3])
        @test g isa GradedIndices{TKS.U1Irrep}
        @test labels(g) == [TKS.U1Irrep(0), TKS.U1Irrep(1)]
        @test sector_multiplicities(g) == [2, 3]
        @test isdual(g) == false
    end

    @testset "gradedrange from dual SectorRange labels" begin
        g = GradedArrays.gradedrange([U1(0)' => 2, U1(1)' => 3])
        @test g isa GradedIndices{TKS.U1Irrep}
        @test labels(g) == [TKS.U1Irrep(0), TKS.U1Irrep(1)]
        @test sector_multiplicities(g) == [2, 3]
        @test isdual(g) == true
    end

    @testset "gradedrange mixed isdual throws" begin
        @test_throws ArgumentError GradedArrays.gradedrange([U1(0) => 2, U1(1)' => 3])
    end

    @testset "dual via adjoint (U1)" begin
        g = GradedArrays.gradedrange([U1(0) => 2, U1(1) => 3])
        gd = g'
        @test isdual(gd) == true
        @test labels(gd) == labels(g)
        @test sector_multiplicities(gd) == sector_multiplicities(g)
    end

    @testset "double dual is identity" begin
        g = GradedArrays.gradedrange([U1(0) => 2, U1(1) => 3])
        @test dual(dual(g)) == g
        @test g'' == g
    end

    @testset "sectors accessor — non-dual" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        @test sectors(g) == [U1(0), U1(1)]
    end

    @testset "sectors accessor — dual" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])'
        @test sectors(g) == [U1(0)', U1(1)']
    end

    @testset "blocklength" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        @test blocklength(g) == 2
    end

    @testset "length — U1 (abelian, dim=1)" begin
        # U1: TKS.dim(U1Irrep(n)) == 1 for all n
        g = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        @test length(g) == 1 * 2 + 1 * 3  # dim * mult summed
    end

    @testset "length — SU2 (non-abelian)" begin
        # SU2 j=0: dim=1, j=1/2: dim=2, j=1: dim=3
        g = GradedArrays.gradedrange(
            [
                TKS.SU2Irrep(0) => 1, TKS.SU2Irrep(1 // 2) => 2, TKS.SU2Irrep(1) => 3,
            ]
        )
        @test length(g) == 1 * 1 + 2 * 2 + 3 * 3  # 1 + 4 + 9 = 14
    end

    @testset "length — empty" begin
        g = GradedArrays.gradedrange(Pair{TKS.U1Irrep, Int}[])
        @test length(g) == 0
        @test blocklength(g) == 0
    end

    @testset "flip" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(1) => 3, TKS.U1Irrep(2) => 5])
        gf = flip(g)
        @test labels(gf) == [TKS.dual(TKS.U1Irrep(1)), TKS.dual(TKS.U1Irrep(2))]
        @test sector_multiplicities(gf) == [3, 5]
        @test isdual(gf) == true
    end

    @testset "flip on dual" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(1) => 3])'
        gf = flip(g)
        @test labels(gf) == [TKS.dual(TKS.U1Irrep(1))]
        @test isdual(gf) == false  # was dual, flip toggles
    end

    @testset "equality" begin
        g1 = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        g2 = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        g3 = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 4])
        g4 = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])'

        @test g1 == g2
        @test g1 != g3  # different multiplicity
        @test g1 != g4  # different dual
    end

    @testset "hashing" begin
        g1 = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        g2 = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        @test hash(g1) == hash(g2)

        d = Dict(g1 => "value")
        @test d[g2] == "value"
    end

    @testset "sector_type" begin
        @test sector_type(GradedIndices{TKS.U1Irrep}) == SectorRange{TKS.U1Irrep}
        @test sector_type(GradedIndices{TKS.SU2Irrep}) == SectorRange{TKS.SU2Irrep}
    end

    @testset "show" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
        str = sprint(show, g)
        @test contains(str, "GradedIndices")
        @test contains(str, "=> 2")
        @test contains(str, "=> 3")
        @test !endswith(str, "'")

        gd = g'
        strd = sprint(show, gd)
        @test endswith(strd, "'")
    end

    @testset "repeated sectors allowed" begin
        g = GradedArrays.gradedrange([TKS.U1Irrep(1) => 2, TKS.U1Irrep(1) => 3])
        @test labels(g) == [TKS.U1Irrep(1), TKS.U1Irrep(1)]
        @test sector_multiplicities(g) == [2, 3]
        @test blocklength(g) == 2
        @test length(g) == 1 * 2 + 1 * 3  # 5
    end

    @testset "SU2 gradedrange" begin
        g = GradedArrays.gradedrange(
            [
                TKS.SU2Irrep(0) => 1, TKS.SU2Irrep(1 // 2) => 2,
            ]
        )
        @test g isa GradedIndices{TKS.SU2Irrep}
        @test blocklength(g) == 2
        @test length(g) == 1 * 1 + 2 * 2  # 5

        gd = g'
        @test isdual(gd) == true
        @test sectors(gd) == [SU2(TKS.SU2Irrep(0))', SU2(TKS.SU2Irrep(1 // 2))']
    end

    @testset "SU2 gradedrange from SectorRange" begin
        g = GradedArrays.gradedrange([SU2(TKS.SU2Irrep(0)) => 1, SU2(TKS.SU2Irrep(1)) => 2])
        @test g isa GradedIndices{TKS.SU2Irrep}
        @test labels(g) == [TKS.SU2Irrep(0), TKS.SU2Irrep(1)]
        @test sector_multiplicities(g) == [1, 2]
    end

    @testset "mismatched labels and multiplicities" begin
        @test_throws ArgumentError GradedIndices(
            [TKS.U1Irrep(0)], Int[1, 2], false
        )
    end

    @testset "tensor_product (abelian)" begin
        g1 = gradedrange([U1(0) => 2, U1(1) => 3])
        g2 = gradedrange([U1(0) => 1, U1(-1) => 2])

        # two-arg: fuses and sorts
        tp = tensor_product(g1, g2)
        @test tp isa GradedIndices
        @test !isdual(tp)
        # sectors should be sorted and merged
        @test sectors(tp) == sort(sectors(tp))

        # single-arg: mergesort + flip_dual
        g = gradedrange([U1(1) => 2, U1(0) => 3])
        @test tensor_product(g) == gradedrange([U1(0) => 3, U1(1) => 2])

        # single-arg dual: flips then mergesorts
        gd = gradedrange([U1(1) => 2, U1(0) => 3])'
        tp_d = tensor_product(gd)
        @test !isdual(tp_d)

        # variadic fold
        g_small = gradedrange([U1(0) => 1, U1(1) => 1])
        tp3 = tensor_product(g_small, g_small, g_small)
        @test tp3 isa GradedIndices
        @test !isdual(tp3)
        tp4 = tensor_product(g_small, g_small, g_small, g_small)
        @test tp4 isa GradedIndices
    end

    @testset "tensor_product (non-abelian)" begin
        # SU2: j=0 ⊕ j=1/2 fused with itself
        g = gradedrange([SU2(TKS.SU2Irrep(0)) => 1, SU2(TKS.SU2Irrep(1 // 2)) => 1])
        tp = tensor_product(g, g)
        @test tp isa GradedIndices
        @test !isdual(tp)
    end
end
