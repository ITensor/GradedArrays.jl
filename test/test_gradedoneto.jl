using BlockArrays: blocklength
using GradedArrays: GradedArrays, GradedOneTo, SU2, SectorRange, U1, datalengths, dual,
    flip, gradedrange, isdual, sectors, sectortype, tensor_product
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "GradedOneTo" begin
    @testset "gradedrange from SectorRange (U1)" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        @test g isa GradedOneTo{U1}
        @test sectors(g) == [U1(0), U1(1)]
        @test datalengths(g) == [2, 3]
        @test isdual(g) == false
    end

    @testset "gradedrange from dual SectorRange labels" begin
        g = gradedrange([U1(0)' => 2, U1(1)' => 3])
        @test g isa GradedOneTo{U1}
        @test sectors(g) == [U1(0)', U1(1)']
        @test datalengths(g) == [2, 3]
        @test isdual(g) == true
    end

    @testset "gradedrange mixed isdual throws" begin
        @test_throws ArgumentError gradedrange([U1(0) => 2, U1(1)' => 3])
    end

    @testset "dual via adjoint (U1)" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        gd = g'
        @test isdual(gd) == true
        @test sectors(gd) == dual.(sectors(g))
        @test datalengths(gd) == datalengths(g)
    end

    @testset "double dual is identity" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        @test dual(dual(g)) == g
        @test g'' == g
    end

    @testset "sectors accessor — non-dual" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        @test sectors(g) == [U1(0), U1(1)]
    end

    @testset "sectors accessor — dual" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])'
        @test sectors(g) == [U1(0)', U1(1)']
    end

    @testset "blocklength" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        @test blocklength(g) == 2
    end

    @testset "length — U1 (abelian, dim=1)" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        @test length(g) == 1 * 2 + 1 * 3  # dim * mult summed
    end

    @testset "length — SU2 (non-abelian)" begin
        g = gradedrange(
            [
                SU2(0) => 1, SU2(1 // 2) => 2,
                SU2(1) => 3,
            ]
        )
        @test length(g) == 1 * 1 + 2 * 2 + 3 * 3  # 1 + 4 + 9 = 14
    end

    @testset "length — empty" begin
        g = gradedrange(Pair{U1, Int}[])
        @test length(g) == 0
        @test blocklength(g) == 0
    end

    @testset "flip" begin
        g = gradedrange([U1(1) => 3, U1(2) => 5])
        gf = flip(g)
        @test sectors(gf) == [flip(U1(1)), flip(U1(2))]
        @test datalengths(gf) == [3, 5]
        @test isdual(gf) == true
    end

    @testset "flip on dual" begin
        g = gradedrange([U1(1) => 3])'
        gf = flip(g)
        @test sectors(gf) == [flip(U1(1)')]
        @test isdual(gf) == false  # was dual, flip toggles
    end

    @testset "equality" begin
        g1 = gradedrange([U1(0) => 2, U1(1) => 3])
        g2 = gradedrange([U1(0) => 2, U1(1) => 3])
        g3 = gradedrange([U1(0) => 2, U1(1) => 4])
        g4 = gradedrange([U1(0) => 2, U1(1) => 3])'

        @test g1 == g2
        @test g1 != g3  # different multiplicity
        @test g1 != g4  # different dual
    end

    @testset "hashing" begin
        g1 = gradedrange([U1(0) => 2, U1(1) => 3])
        g2 = gradedrange([U1(0) => 2, U1(1) => 3])
        @test hash(g1) == hash(g2)

        d = Dict(g1 => "value")
        @test d[g2] == "value"
    end

    @testset "sectortype" begin
        @test sectortype(GradedOneTo{U1}) == U1
        @test sectortype(GradedOneTo{SU2}) == SU2
    end

    @testset "show" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        str = sprint(show, g)
        @test contains(str, "gradedrange")
        @test contains(str, "=> 2")
        @test contains(str, "=> 3")
        @test !contains(str, "'")
        @test !contains(str, "dual")

        # Dual axes factor the `dual` to the outside.
        gd = g'
        strd = sprint(show, gd)
        @test !endswith(strd, "'")
        @test startswith(strd, "dual(gradedrange(")
        @test endswith(strd, "))")
    end

    @testset "repeated sectors allowed" begin
        g = gradedrange([U1(1) => 2, U1(1) => 3])
        @test sectors(g) == [U1(1), U1(1)]
        @test datalengths(g) == [2, 3]
        @test blocklength(g) == 2
        @test length(g) == 1 * 2 + 1 * 3  # 5
    end

    @testset "SU2 gradedrange" begin
        g = gradedrange(
            [
                SU2(0) => 1, SU2(1 // 2) => 2,
            ]
        )
        @test g isa GradedOneTo{SU2}
        @test blocklength(g) == 2
        @test length(g) == 1 * 1 + 2 * 2  # 5

        gd = g'
        @test isdual(gd) == true
        @test sectors(gd) == [SU2(0)', SU2(1 // 2)']
    end

    @testset "SU2 gradedrange from SectorRange" begin
        g = gradedrange([SU2(0) => 1, SU2(1) => 2])
        @test g isa GradedOneTo{SU2}
        @test sectors(g) == [SU2(0), SU2(1)]
        @test datalengths(g) == [1, 2]
    end

    @testset "mismatched sectors and multiplicities" begin
        @test_throws ArgumentError GradedOneTo(
            [U1(0)], Int[1, 2], false
        )
    end

    @testset "tensor_product (abelian)" begin
        g1 = gradedrange([U1(0) => 2, U1(1) => 3])
        g2 = gradedrange([U1(0) => 1, U1(-1) => 2])

        tp = tensor_product(g1, g2)
        @test tp isa GradedOneTo
        @test !isdual(tp)
        @test sectors(tp) == sort(sectors(tp))

        g = gradedrange([U1(1) => 2, U1(0) => 3])
        @test tensor_product(g) == gradedrange([U1(0) => 3, U1(1) => 2])

        gd = gradedrange([U1(1) => 2, U1(0) => 3])'
        tp_d = tensor_product(gd)
        @test !isdual(tp_d)

        g_small = gradedrange([U1(0) => 1, U1(1) => 1])
        tp3 = tensor_product(g_small, g_small, g_small)
        @test tp3 isa GradedOneTo
        @test !isdual(tp3)
        tp4 = tensor_product(g_small, g_small, g_small, g_small)
        @test tp4 isa GradedOneTo
    end

    @testset "tensor_product (non-abelian)" begin
        g = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        tp = tensor_product(g, g)
        @test tp isa GradedOneTo
        @test !isdual(tp)
    end
end
