using GradedArrays:
    GradedArrays,
    SectorProduct,
    SU2,
    TrivialSector,
    U1,
    Z,
    ×,
    arguments,
    dual,
    gradedrange,
    label,
    quantum_dimension,
    sector,
    sector_type,
    sectorproduct,
    sectorrange,
    space_isequal,
    trivial
using TensorProducts: ⊗
using Test: @test, @test_broken, @test_throws, @testset
using TestExtras: @constinferred
using BlockArrays: blocklengths
import TensorKitSectors as TKS

@testset "Test Ordered Products" begin
    @testset "Ordered Constructor" begin
        s = SectorProduct(TKS.U1Irrep(1))
        @test length(arguments(s)) == 1
        @test (@constinferred quantum_dimension(s)) == 1
        @test (@constinferred dual(s)) == SectorProduct(TKS.U1Irrep(-1))
        @test arguments(s)[1] == TKS.U1Irrep(1)
        @test (@constinferred trivial(s)) == SectorProduct(TKS.U1Irrep(0))

        s = SectorProduct(TKS.U1Irrep(1), TKS.U1Irrep(2))
        @test length(arguments(s)) == 2
        @test (@constinferred quantum_dimension(s)) == 1
        @test (@constinferred dual(s)) == SectorProduct(TKS.U1Irrep(-1), TKS.U1Irrep(-2))
        @test arguments(s)[1] == TKS.U1Irrep(1)
        @test arguments(s)[2] == TKS.U1Irrep(2)
        @test (@constinferred trivial(s)) == SectorProduct(TKS.U1Irrep(0), TKS.U1Irrep(0))

        s = U1(1) × SU2(1 // 2) × U1(3)
        @test s ≡ sectorproduct(U1(1), SU2(1 // 2), U1(3))
        @test s ≡ ×(U1(1), SU2(1 // 2), U1(3))
        @test length(arguments(s)) == 3
        @test (@constinferred quantum_dimension(s)) == 2
        @test (@constinferred dual(s)) == U1(-1) × SU2(1 // 2) × U1(-3)
        @test arguments(s)[1] == U1(1)
        @test arguments(s)[2] == SU2(1 // 2)
        @test arguments(s)[3] == U1(3)
        @test (@constinferred trivial(s)) == SectorProduct(U1(0), SU2(0), U1(0))

        s = TrivialSector() × U1(3) × SU2(1 / 2)
        @test length(arguments(s)) == 3
        @test (@constinferred quantum_dimension(s)) == 2
        @test dual(s) == TrivialSector() × U1(-3) × SU2(1 // 2)
        @test (@constinferred trivial(s)) == SectorProduct(TrivialSector(), U1(0), SU2(0))
        @test s > trivial(s)
    end

    @testset "Ordered comparisons" begin
        # convention: missing arguments are filled with singlets
        @test sectorproduct(U1(1), SU2(1)) == sectorproduct(U1(1), SU2(1))
        @test sectorproduct(U1(1), SU2(0)) != sectorproduct(U1(1), SU2(1))
        @test sectorproduct(U1(0), SU2(1)) != sectorproduct(U1(1), SU2(1))
        @test_broken sectorproduct(U1(1)) != U1(1)
        @test sectorproduct(U1(1)) == sectorproduct(U1(1), U1(0))
        @test sectorproduct(U1(1)) != sectorproduct(U1(1), U1(1))
        @test sectorproduct(U1(0), SU2(0)) == label(TrivialSector())
        @test sectorproduct(U1(0), SU2(0)) == sectorproduct(TrivialSector(), SU2(0))
        @test sectorproduct(U1(0), SU2(0)) == sectorproduct(U1(0), TrivialSector())
        @test sectorproduct(U1(0), SU2(0)) == sectorproduct(TrivialSector(), TrivialSector())

        @test sectorproduct(U1(0)) < sectorproduct((U1(1)))
        @test sectorproduct(U1(0), U1(2)) < sectorproduct((U1(1)), U1(0))
        @test sectorproduct(U1(0)) < sectorproduct(U1(0), U1(1))
        @test sectorproduct(U1(0)) > sectorproduct(U1(0), U1(-1))
    end

    @testset "Quantum dimension and GradedUnitRange" begin
        g = gradedrange([(U1(0) × Z{2}(0)) => 1, (U1(1) × Z{2}(0)) => 2])  # abelian
        @test (@constinferred quantum_dimension(g)) == 3

        g = gradedrange(
            [  # non-abelian
                (SU2(0) × SU2(0)) => 1,
                (SU2(1) × SU2(0)) => 1,
                (SU2(0) × SU2(1)) => 1,
                (SU2(1) × SU2(1)) => 1,
            ]
        )
        @test (@constinferred quantum_dimension(g)) == 16
        @test (@constinferred blocklengths(g)) == [1, 3, 3, 9]

        @test space_isequal(
            gradedrange([U1(1) => 2]) × SU2(1), gradedrange([U1(1) × SU2(1) => 2])
        )
        @test space_isequal(
            SU2(1) × gradedrange([U1(1) => 2]), gradedrange([SU2(1) × U1(1) => 2])
        )

        # mixed group
        g = gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1])
        @test (@constinferred quantum_dimension(g)) == 4
        @test (@constinferred blocklengths(g)) == [1, 3]
        g = gradedrange([(SU2(0) × U1(0) × SU2(1 // 2)) => 1, (SU2(0) × U1(1) × SU2(1 // 2)) => 1])
        @test (@constinferred quantum_dimension(g)) == 4
        @test (@constinferred blocklengths(g)) == [2, 2]
    end

    @testset "Fusion of Abelian products" begin
        p1 = ×(U1(1))
        p2 = ×(U1(2))
        @test (@constinferred p1 ⊗ TrivialSector()) == p1
        @test (@constinferred TrivialSector() ⊗ p2) == p2
        @test (@constinferred p1 ⊗ p2) == ×(U1(3))

        p11 = U1(1) × U1(1)
        @test p11 ⊗ p11 == U1(2) × U1(2)

        p123 = U1(1) × U1(2) × U1(3)
        @test p123 ⊗ p123 == U1(2) × U1(4) × U1(6)

        s1 = sectorproduct(U1(1), Z{2}(1))
        s2 = sectorproduct(U1(0), Z{2}(0))
        @test s1 ⊗ s2 == U1(1) × Z{2}(1)
    end

    @testset "Fusion of NonAbelian products" begin
        p0 = ×(SU2(0))
        ph = ×(SU2(1 // 2))
        @test space_isequal(
            (@constinferred p0 ⊗ TrivialSector()), gradedrange([sectorproduct(SU2(0)) => 1])
        )
        @test space_isequal(
            (@constinferred TrivialSector() ⊗ ph), gradedrange([sectorproduct(SU2(1 // 2)) => 1])
        )

        phh = SU2(1 // 2) × SU2(1 // 2)
        @test space_isequal(
            phh ⊗ phh,
            gradedrange(
                [
                    (SU2(0) × SU2(0)) => 1,
                    (SU2(1) × SU2(0)) => 1,
                    (SU2(0) × SU2(1)) => 1,
                    (SU2(1) × SU2(1)) => 1,
                ]
            ),
        )
        @test space_isequal(
            phh ⊗ phh,
            gradedrange(
                [
                    (SU2(0) × SU2(0)) => 1,
                    (SU2(1) × SU2(0)) => 1,
                    (SU2(0) × SU2(1)) => 1,
                    (SU2(1) × SU2(1)) => 1,
                ]
            ),
        )
    end

    @testset "Fusion of different length Categories" begin
        @test (U1(1) × U1(0)) ⊗ ×(U1(1)) == U1(2) × U1(0)
        @test space_isequal(
            (@constinferred (SU2(0) × SU2(0)) ⊗ ×(SU2(1))), gradedrange([(SU2(1) × SU2(0)) => 1])
        )

        @test space_isequal(
            (@constinferred (SU2(1) × U1(1)) ⊗ ×(SU2(0))), gradedrange([SU2(1) × U1(1) => 1])
        )
        @test space_isequal(
            (@constinferred U1(1) × SU2(1) ⊗ ×(U1(2))), gradedrange([U1(3) × SU2(1) => 1])
        )

        # check incompatible sectors
        p12 = Z{2}(1) × U1(2)
        z12 = Z{2}(1) × Z{2}(1)
        @test_throws ArgumentError p12 ⊗ z12
    end

    @testset "GradedUnitRange fusion rules" begin
        s1 = U1(1) × SU2(1 // 2)
        s2 = U1(0) × SU2(1 // 2)
        g1 = gradedrange([s1 => 2])
        g2 = gradedrange([s2 => 1])
        @test space_isequal(g1 ⊗ g2, gradedrange([U1(1) × SU2(0) => 2, U1(1) × SU2(1) => 2]))
    end
end

@testset "Test Named Sector Products" begin
    @testset "Construct from × of NamedTuples" begin
        s = (A = U1(1),) × (B = Z{2}(0),)
        @test length(arguments(s)) == 2
        @test arguments(s)[:A] == U1(1)
        @test arguments(s)[:B] == Z{2}(0)
        @test (@constinferred quantum_dimension(s)) == 1
        @test (@constinferred dual(s)) == (A = U1(-1),) × (B = Z{2}(0),)
        @test (@constinferred trivial(s)) == (A = U1(0),) × (B = Z{2}(0),)

        s = (A = U1(1),) × (B = SU2(2),)
        @test length(arguments(s)) == 2
        @test arguments(s)[:A] == U1(1)
        @test arguments(s)[:B] == SU2(2)
        @test (@constinferred quantum_dimension(s)) == 5
        @test (@constinferred dual(s)) == (A = U1(-1),) × (B = SU2(2),)
        @test (@constinferred trivial(s)) == (A = U1(0),) × (B = SU2(0),)
        @test s == (B = SU2(2),) × (A = U1(1),)

        s1 = (A = U1(1),) × (B = Z{2}(0),)
        s2 = (A = U1(1),) × (C = Z{2}(0),)
        @test_throws ArgumentError s1 × s2

        g = gradedrange([(Nf = U1(0),) => 2, (Nf = U1(1),) => 3])
        @test sector_type(g) <: GradedArrays.SectorRange{<:SectorProduct}
        sr = sectorrange((; S = SU2(1 // 2)) => 1)
        @test length(sr) == 2
        @test space_isequal(sr, sectorrange(×((; S = SU2(1 // 2))), 1))
        @test space_isequal(sr, sectorrange(×((; S = SU2(1 // 2))), 1:2))
        g = gradedrange([(; S = SU2(1 // 2)) => 1])
        @test length(g) == 2
        @test space_isequal(g, gradedrange([×((; S = SU2(1 // 2))) => 1]))

        @test (A = U1(1),) × ((B = SU2(2),) × (C = U1(1),)) isa
            typeof((A = U1(1),) × (B = SU2(2),) × (C = U1(1),))
    end

    @testset "Construct from Pairs" begin
        s = ×("A" => U1(2))
        @test length(arguments(s)) == 1
        @test arguments(s)[:A] == U1(2)
        @test s == ×((; A = U1(2)))
        @test (@constinferred quantum_dimension(s)) == 1
        @test (@constinferred dual(s)) == ×("A" => U1(-2))
        @test (@constinferred trivial(s)) == ×((; A = U1(0)))

        s = ×("B" => SU2(1 // 2), :C => Z{2}(1))
        @test length(arguments(s)) == 2
        @test arguments(s)[:B] == SU2(1 // 2)
        @test arguments(s)[:C] == Z{2}(1)
        @test (@constinferred quantum_dimension(s)) == 2
    end

    @testset "Comparisons with unspecified labels" begin
        # convention: arguments evaluate as equal if unmatched labels are trivial
        # this is different from ordered tuple convention
        q2 = ×((; N = U1(2)))
        q20 = (N = U1(2),) × (J = SU2(0),)
        @test q20 == q2
        @test !(q20 < q2)
        @test !(q2 < q20)

        q21 = (N = U1(2),) × (J = SU2(1),)
        @test q21 != q2
        @test q20 < q21
        @test q2 < q21

        a = (A = U1(0),) × (B = U1(2),)
        b = (B = U1(2),) × (C = U1(0),)
        @test a == b
        c = (B = U1(2),) × (C = U1(1),)
        @test a != c
    end

    @testset "Quantum dimension and GradedUnitRange" begin
        g = gradedrange([(; A = U1(0)) × (; B = Z{2}(0)) => 1, (; A = U1(1)) × (; B = Z{2}(0)) => 2])  # abelian
        @test (@constinferred quantum_dimension(g)) == 3

        g = gradedrange(
            [  # non-abelian
                (; A = SU2(0)) × (; B = SU2(0)) => 1,
                (; A = SU2(1)) × (; B = SU2(0)) => 1,
                (; A = SU2(0)) × (; B = SU2(1)) => 1,
                (; A = SU2(1)) × (; B = SU2(1)) => 1,
            ]
        )
        @test (@constinferred quantum_dimension(g)) == 16

        # mixed group
        g = gradedrange(
            [
                (; A = U1(2)) × (; B = SU2(0)) × (; C = Z{2}(0)) => 1,
                (; A = U1(2)) × (; B = SU2(1)) × (; C = Z{2}(0)) => 1,
            ]
        )
        @test (@constinferred quantum_dimension(g)) == 4
        g = gradedrange(
            [
                (; A = SU2(0)) × (; B = Z{2}(0)) × (; C = SU2(1 // 2)) => 1,
                (; A = SU2(0)) × (; B = Z{2}(1)) × (; C = SU2(1 // 2)) => 1,
            ]
        )
        @test (@constinferred quantum_dimension(g)) == 4
    end

    @testset "Fusion of Abelian products" begin
        q00 = ×((;))
        q10 = ×((; A = U1(1)))
        q01 = ×((; B = U1(1)))
        q11 = (; A = U1(1)) × (; B = U1(1))

        @test (@constinferred q10 ⊗ q10) == ×((; A = U1(2)))
        @test (@constinferred q01 ⊗ q00) == q01
        @test (@constinferred q00 ⊗ q01) == q01
        @test (@constinferred q10 ⊗ q01) == q11
        @test q11 ⊗ q11 == (; A = U1(2)) × (; B = U1(2))

        s11 = (; A = U1(1)) × (; B = Z{2}(1))
        s10 = ×((; A = U1(1)))
        s01 = ×((; B = Z{2}(1)))
        @test (@constinferred s01 ⊗ q00) == s01
        @test (@constinferred q00 ⊗ s01) == s01
        @test (@constinferred s10 ⊗ s01) == s11
        @test s11 ⊗ s11 == (; A = U1(2)) × (; B = Z{2}(0))
    end

    @testset "Fusion of NonAbelian products" begin
        p0 = ×((;))
        pha = ×((; A = SU2(1 // 2)))
        phb = ×((; B = SU2(1 // 2)))
        phab = (; A = SU2(1 // 2)) × (; B = SU2(1 // 2))

        @test space_isequal(
            (@constinferred pha ⊗ pha), gradedrange([×((; A = SU2(0))) => 1, ×((; A = SU2(1))) => 1])
        )
        @test space_isequal((@constinferred pha ⊗ p0), gradedrange([pha => 1]))
        @test space_isequal((@constinferred p0 ⊗ phb), gradedrange([phb => 1]))
        @test space_isequal((@constinferred pha ⊗ phb), gradedrange([phab => 1]))

        @test space_isequal(
            phab ⊗ phab,
            gradedrange(
                [
                    (; A = SU2(0)) × (; B = SU2(0)) => 1,
                    (; A = SU2(1)) × (; B = SU2(0)) => 1,
                    (; A = SU2(0)) × (; B = SU2(1)) => 1,
                    (; A = SU2(1)) × (; B = SU2(1)) => 1,
                ]
            ),
        )
    end

    @testset "Fusion of mixed Abelian and NonAbelian products" begin
        q0h = ×((; J = SU2(1 // 2)))
        q10 = (N = U1(1),) × (J = SU2(0),)
        # Put names in reverse order sometimes:
        q1h = (J = SU2(1 // 2),) × (N = U1(1),)
        q11 = (N = U1(1),) × (J = SU2(1),)
        q20 = (N = U1(2),) × (J = SU2(0),)  # julia 1.6 does not accept gradedrange without J
        q2h = (N = U1(2),) × (J = SU2(1 // 2),)
        q21 = (N = U1(2),) × (J = SU2(1),)
        q22 = (N = U1(2),) × (J = SU2(2),)

        @test space_isequal(q1h ⊗ q1h, gradedrange([q20 => 1, q21 => 1]))
        @test space_isequal(q10 ⊗ q1h, gradedrange([q2h => 1]))
        @test space_isequal((@constinferred q0h ⊗ q1h), gradedrange([q10 => 1, q11 => 1]))
        @test space_isequal(q11 ⊗ q11, gradedrange([q20 => 1, q21 => 1, q22 => 1]))
    end

    @testset "GradedUnitRange fusion rules" begin
        s1 = (; A = U1(1)) × (; B = SU2(1 // 2))
        s2 = (; A = U1(0)) × (; B = SU2(1 // 2))
        g1 = gradedrange([s1 => 2])
        g2 = gradedrange([s2 => 1])
        s3 = (; A = U1(1)) × (; B = SU2(0))
        s4 = (; A = U1(1)) × (; B = SU2(1))
        @test space_isequal(g1 ⊗ g2, gradedrange([s3 => 2, s4 => 2]))

        sA = ×((; A = U1(1)))
        sB = ×((; B = SU2(1 // 2)))
        sAB = (; A = U1(1)) × (; B = SU2(1 // 2))
        gA = gradedrange([sA => 2])
        gB = gradedrange([sB => 1])
        @test space_isequal(gA ⊗ gB, gradedrange([sAB => 2]))
    end
end

@testset "Mixing implementations" begin
    st1 = ×(U1(1))
    sA1 = ×((; A = U1(1)))

    @test_throws MethodError sA1 != st1
    @test_throws MethodError sA1 < st1
    @test_throws MethodError st1 < sA1
    @test_throws MethodError st1 ⊗ sA1
    @test_throws MethodError sA1 ⊗ st1
    @test_throws MethodError st1 × sA1
    @test_throws MethodError sA1 × st1
end

@testset "Empty SymmetrySector" begin
    st1 = ×(U1(1))
    sA1 = ×((; A = U1(1)))

    for s in (SectorProduct(()), SectorProduct((;)))
        @test s == TrivialSector()
        @test s == SectorProduct(())
        @test s == SectorProduct((;))

        @test !(s < SectorProduct())
        @test !(s < SectorProduct())

        @test (@constinferred s × SectorProduct(())) == s
        @test (@constinferred s × SectorProduct((;))) == s
        @test label(@constinferred s ⊗ SectorProduct(())) == s
        @test label(@constinferred s ⊗ SectorProduct((;))) == s

        @test (@constinferred dual(s)) == s
        @test (@constinferred trivial(s)) == s
        @test (@constinferred quantum_dimension(s)) == 1

        g0 = gradedrange([s => 2])
        @test space_isequal((@constinferred ⊗(g0, g0)), gradedrange([s => 4]))

        @test (@constinferred s × U1(1)) == st1
        @test (@constinferred U1(1) × s) == st1
        @test (@constinferred s × st1) == st1
        @test (@constinferred st1 × s) == st1
        @test (@constinferred s × sA1) == sA1
        @test (@constinferred sA1 × s) == sA1

        @test (@constinferred U1(1) ⊗ s) == st1
        @test (@constinferred s ⊗ U1(1)) == st1
        @test (@constinferred SU2(0) ⊗ s) == gradedrange([SectorProduct(SU2(0)) => 1])
        @test (@constinferred s ⊗ SU2(0)) == gradedrange([SectorProduct(SU2(0)) => 1])

        @test (@constinferred st1 ⊗ s) == st1
        @test (@constinferred SectorProduct(SU2(0)) ⊗ s) ==
            gradedrange([SectorProduct(SU2(0)) => 1])
        @test (@constinferred SectorProduct(SU2(1), U1(2)) ⊗ s) ==
            gradedrange([SectorProduct(SU2(1), U1(2)) => 1])

        @test (@constinferred sA1 ⊗ s) == sA1
        @test (@constinferred ×((; A = SU2(0))) ⊗ s) == gradedrange([×((; A = SU2(0))) => 1])
        @test (@constinferred (; B = SU2(1)) × (; C = U1(2)) ⊗ s) ==
            gradedrange([(; B = SU2(1)) × (; C = U1(2)) => 1])

        # Empty behaves as empty NamedTuple
        @test_broken s != U1(0)
        @test s == SectorProduct(U1(0))
        @test s == ×((; A = U1(0)))
        @test ×((; A = U1(0))) == s
        @test s != sA1
        @test s != st1

        @test s < st1
        @test SectorProduct(U1(-1)) > s
        @test s < sA1
        @test s < ×((; A = U1(-1)))
        @test !(s < ×((; A = U1(0))))
        @test !(s > ×((; A = U1(0))))
    end
end
