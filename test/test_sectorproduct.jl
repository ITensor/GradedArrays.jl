using BlockArrays: blocklengths
using GradedArrays: SU2, Sector, SectorOneTo, SectorProduct, Trivial, U1, Z, arguments,
    dual, fSU2, fU1, fZ2, flip, gradedrange, istrivial, sectorproduct, sectortype,
    tensor_product, trivial, ×
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

@testset "Test Ordered Products" begin
    @testset "Ordered Constructor" begin
        s = Sector((TKS.U1Irrep(1),))
        @test length(arguments(s)) == 1
        @test (@constinferred length(s)) == 1
        @test (@constinferred flip(dual(s))) == Sector((TKS.U1Irrep(-1),))
        @test arguments(s)[1] == U1(1)
        @test (@constinferred trivial(s)) == Sector((TKS.U1Irrep(0),))

        s = Sector(TKS.U1Irrep(1), TKS.U1Irrep(2))
        @test length(arguments(s)) == 2
        @test (@constinferred length(s)) == 1
        @test (@constinferred flip(dual(s))) == Sector(TKS.U1Irrep(-1), TKS.U1Irrep(-2))
        @test arguments(s)[1] == U1(1)
        @test arguments(s)[2] == U1(2)
        @test (@constinferred trivial(s)) == Sector(TKS.U1Irrep(0), TKS.U1Irrep(0))

        s = U1(1) × SU2(1 // 2) × U1(3)
        @test s ≡ sectorproduct(U1(1), SU2(1 // 2), U1(3))
        @test s ≡ ×(U1(1), SU2(1 // 2), U1(3))
        @test s ≡ Sector(U1(1), SU2(1 // 2), U1(3))
        @test length(arguments(s)) == 3
        @test (@constinferred length(s)) == 2
        @test (@constinferred flip(dual(s))) == U1(-1) × SU2(1 // 2) × U1(-3)
        @test arguments(s)[1] == U1(1)
        @test arguments(s)[2] == SU2(1 // 2)
        @test arguments(s)[3] == U1(3)
        @test (@constinferred trivial(s)) == Sector(U1(0), SU2(0), U1(0))

        # `×` over one sector is normalization and nothing more, so no one-factor product comes
        # out of it. Writing the container asks for one.
        @test ×(U1(1)) ≡ U1(1)
        @test Sector((U1(1),)) ≢ U1(1)

        # `Trivial` is the unit and drops out.
        s = Trivial() × U1(3) × SU2(1 / 2)
        @test s ≡ U1(3) × SU2(1 // 2)
        @test length(arguments(s)) == 2
        @test (@constinferred length(s)) == 2
        @test flip(dual(s)) == U1(-3) × SU2(1 // 2)
        @test (@constinferred trivial(s)) == Sector(U1(0), SU2(0))
        @test s > trivial(s)
    end

    @testset "Ordered comparisons" begin
        # A position identifies a factor only within its own product, so the arity is part of the
        # sector's identity and no argument is ever filled in.
        @test Sector(U1(1), SU2(1)) == Sector(U1(1), SU2(1))
        @test Sector(U1(1), SU2(0)) != Sector(U1(1), SU2(1))
        @test Sector(U1(0), SU2(1)) != Sector(U1(1), SU2(1))
        @test Sector((U1(1),)) != U1(1)
        @test U1(1) != Sector((U1(1),))
        @test Sector((U1(1),)) != Sector(U1(1), U1(0))
        @test Sector((U1(1),)) != Sector(U1(1), U1(1))
        @test Sector(U1(0), SU2(0)) != Sector(U1(0), U1(0))

        # Nothing equals `Trivial` but itself, so a product of trivial sectors does not
        # either. Whether a product denotes no symmetry is `istrivial`'s question.
        @test istrivial(Sector(U1(0), SU2(0)))
        @test Sector(U1(0), SU2(0)) != Trivial()
        @test Trivial() != Sector(U1(0), SU2(0))

        # Same arity and same symmetries orders as TensorKit orders the matching space.
        @test Sector((U1(0),)) < Sector((U1(1),))
        @test Sector(U1(0), U1(2)) > Sector(U1(1), U1(0))
        # Different arities order by arity, so that mixed vectors of sectors can be sorted.
        @test Sector((U1(0),)) < Sector(U1(0), U1(1))
        @test Sector((U1(0),)) < Sector(U1(0), U1(-1))
    end

    @testset "Quantum dimension and GradedOneTo" begin
        g = gradedrange([(U1(0) × Z{2}(0)) => 1, (U1(1) × Z{2}(0)) => 2])  # abelian
        @test (@constinferred length(g)) == 3

        g = gradedrange(
            [  # non-abelian
                (SU2(0) × SU2(0)) => 1,
                (SU2(1) × SU2(0)) => 1,
                (SU2(0) × SU2(1)) => 1,
                (SU2(1) × SU2(1)) => 1,
            ]
        )
        @test (@constinferred length(g)) == 16
        @test (@constinferred blocklengths(g)) == [1, 3, 3, 9]

        # mixed group
        g = gradedrange([(U1(2) × SU2(0) × Z{2}(0)) => 1, (U1(2) × SU2(1) × Z{2}(0)) => 1])
        @test (@constinferred length(g)) == 4
        @test (@constinferred blocklengths(g)) == [1, 3]
        g = gradedrange(
            [
                (SU2(0) × U1(0) × SU2(1 // 2)) => 1,
                (SU2(0) × U1(1) × SU2(1 // 2)) => 1,
            ]
        )
        @test (@constinferred length(g)) == 4
        @test (@constinferred blocklengths(g)) == [2, 2]
    end

    @testset "Fusion of Abelian products" begin
        p1 = Sector((U1(1),))
        p2 = Sector((U1(2),))
        @test (@constinferred tensor_product(p1, Trivial())) == p1
        @test (@constinferred tensor_product(Trivial(), p2)) == p2
        @test (@constinferred tensor_product(p1, p2)) == Sector((U1(3),))

        p11 = U1(1) × U1(1)
        @test tensor_product(p11, p11) == U1(2) × U1(2)

        p123 = U1(1) × U1(2) × U1(3)
        @test tensor_product(p123, p123) == U1(2) × U1(4) × U1(6)

        s1 = Sector(U1(1), Z{2}(1))
        s2 = Sector(U1(0), Z{2}(0))
        @test tensor_product(s1, s2) == U1(1) × Z{2}(1)
    end

    @testset "Fusion of NonAbelian products" begin
        p0 = Sector((SU2(0),))
        ph = Sector((SU2(1 // 2),))
        @test (@constinferred tensor_product(p0, Trivial())) == gradedrange([p0 => 1])
        @test (@constinferred tensor_product(Trivial(), ph)) == gradedrange([ph => 1])

        phh = SU2(1 // 2) × SU2(1 // 2)
        @test tensor_product(phh, phh) == gradedrange(
            [
                (SU2(0) × SU2(0)) => 1,
                (SU2(1) × SU2(0)) => 1,
                (SU2(0) × SU2(1)) => 1,
                (SU2(1) × SU2(1)) => 1,
            ]
        )
    end

    @testset "Fusion rejects mismatched ordered products" begin
        # Positional arguments identify a symmetry by slot, so the operands have to agree slot
        # for slot on both the arity and the symmetry.
        @test_throws ArgumentError tensor_product(U1(1) × U1(0), Sector((U1(1),)))
        @test_throws ArgumentError tensor_product(Sector((U1(1),)), U1(1) × U1(0))
        @test_throws ArgumentError tensor_product(SU2(0) × SU2(0), Sector((SU2(1),)))
        @test_throws ArgumentError tensor_product(SU2(1) × U1(1), Sector((SU2(0),)))
        @test_throws ArgumentError tensor_product(Z{2}(1) × U1(2), Z{2}(1) × Z{2}(1))
    end

    @testset "GradedOneTo fusion rules" begin
        s1 = U1(1) × SU2(1 // 2)
        s2 = U1(0) × SU2(1 // 2)
        g1 = gradedrange([s1 => 2])
        g2 = gradedrange([s2 => 1])
        @test tensor_product(g1, g2) ==
            gradedrange([U1(1) × SU2(0) => 2, U1(1) × SU2(1) => 2])
    end
end

@testset "Test Named Sector Products" begin
    @testset "Construct from × of NamedTuples" begin
        s = (A = U1(1),) × (B = Z{2}(0),)
        @test length(arguments(s)) == 2
        @test arguments(s)[:A] == U1(1)
        @test arguments(s)[:B] == Z{2}(0)
        @test (@constinferred length(s)) == 1
        @test (@constinferred flip(dual(s))) == (A = U1(-1),) × (B = Z{2}(0),)
        @test (@constinferred trivial(s)) == (A = U1(0),) × (B = Z{2}(0),)

        s = (A = U1(1),) × (B = SU2(2),)
        @test length(arguments(s)) == 2
        @test arguments(s)[:A] == U1(1)
        @test arguments(s)[:B] == SU2(2)
        @test (@constinferred length(s)) == 5
        @test (@constinferred flip(dual(s))) == (A = U1(-1),) × (B = SU2(2),)
        @test (@constinferred trivial(s)) == (A = U1(0),) × (B = SU2(0),)
        @test s == (B = SU2(2),) × (A = U1(1),)

        s1 = (A = U1(1),) × (B = Z{2}(0),)
        s2 = (A = U1(1),) × (C = Z{2}(0),)
        @test_throws ArgumentError s1 × s2

        g = gradedrange([(Nf = U1(0),) => 2, (Nf = U1(1),) => 3])
        @test sectortype(g) <: SectorProduct
        sr = SectorOneTo(×((; S = SU2(1 // 2))), 1)
        @test length(sr) == 2
        g = gradedrange([(; S = SU2(1 // 2)) => 1])
        @test length(g) == 2
        @test g == gradedrange([×((; S = SU2(1 // 2))) => 1])

        @test (A = U1(1),) × ((B = SU2(2),) × (C = U1(1),)) isa
            typeof((A = U1(1),) × (B = SU2(2),) × (C = U1(1),))
    end

    @testset "Construct from keywords" begin
        s = Sector(; A = U1(2))
        @test length(arguments(s)) == 1
        @test arguments(s)[:A] == U1(2)
        @test s == ×((; A = U1(2)))
        @test (@constinferred length(s)) == 1
        @test (@constinferred flip(dual(s))) == Sector(; A = U1(-2))
        @test (@constinferred trivial(s)) == ×((; A = U1(0)))

        s = Sector(; B = SU2(1 // 2), C = Z{2}(1))
        @test length(arguments(s)) == 2
        @test arguments(s)[:B] == SU2(1 // 2)
        @test arguments(s)[:C] == Z{2}(1)
        @test (@constinferred length(s)) == 2

        # No keywords is the same specification as an explicit `(;)`.
        @test Sector() ≡ Sector((;))
    end

    @testset "Comparisons with unspecified labels" begin
        # convention: arguments evaluate as equal if unmatched labels are trivial
        # this is different from ordered tuple convention
        q2 = ×((; N = U1(2)))
        q20 = (N = U1(2),) × (J = SU2(0),)
        @test q20 == q2
        @test !(q20 < q2)
        @test !(q2 < q20)
        @test hash(q20) == hash(q2)

        q21 = (N = U1(2),) × (J = SU2(1),)
        @test q21 != q2
        @test q20 < q21
        @test q2 < q21

        a = (A = U1(0),) × (B = U1(2),)
        b = (B = U1(2),) × (C = U1(0),)
        @test a == b
        @test hash(a) == hash(b)
        c = (B = U1(2),) × (C = U1(1),)
        @test a != c
    end

    @testset "Quantum dimension and GradedOneTo" begin
        g = gradedrange(
            [
                (; A = U1(0)) × (; B = Z{2}(0)) => 1,
                (; A = U1(1)) × (; B = Z{2}(0)) => 2,
            ]
        )  # abelian
        @test (@constinferred length(g)) == 3

        g = gradedrange(
            [  # non-abelian
                (; A = SU2(0)) × (; B = SU2(0)) => 1,
                (; A = SU2(1)) × (; B = SU2(0)) => 1,
                (; A = SU2(0)) × (; B = SU2(1)) => 1,
                (; A = SU2(1)) × (; B = SU2(1)) => 1,
            ]
        )
        @test (@constinferred length(g)) == 16

        # mixed group
        g = gradedrange(
            [
                (; A = U1(2)) × (; B = SU2(0)) × (; C = Z{2}(0)) => 1,
                (; A = U1(2)) × (; B = SU2(1)) × (; C = Z{2}(0)) => 1,
            ]
        )
        @test (@constinferred length(g)) == 4
        g = gradedrange(
            [
                (; A = SU2(0)) × (; B = Z{2}(0)) × (; C = SU2(1 // 2)) => 1,
                (; A = SU2(0)) × (; B = Z{2}(1)) × (; C = SU2(1 // 2)) => 1,
            ]
        )
        @test (@constinferred length(g)) == 4
    end

    @testset "Fusion of Abelian products" begin
        q00 = ×((;))
        q10 = ×((; A = U1(1)))
        q01 = ×((; B = U1(1)))
        q11 = (; A = U1(1)) × (; B = U1(1))

        @test (@constinferred tensor_product(q10, q10)) == ×((; A = U1(2)))
        @test (@constinferred tensor_product(q01, q00)) == q01
        @test (@constinferred tensor_product(q00, q01)) == q01
        @test (@constinferred tensor_product(q10, q01)) == q11
        @test tensor_product(q11, q11) == (; A = U1(2)) × (; B = U1(2))

        s11 = (; A = U1(1)) × (; B = Z{2}(1))
        s10 = ×((; A = U1(1)))
        s01 = ×((; B = Z{2}(1)))
        @test (@constinferred tensor_product(s01, q00)) == s01
        @test (@constinferred tensor_product(q00, s01)) == s01
        @test (@constinferred tensor_product(s10, s01)) == s11
        @test tensor_product(s11, s11) == (; A = U1(2)) × (; B = Z{2}(0))
    end

    @testset "Fusion of NonAbelian products" begin
        p0 = ×((;))
        pha = ×((; A = SU2(1 // 2)))
        phb = ×((; B = SU2(1 // 2)))
        phab = (; A = SU2(1 // 2)) × (; B = SU2(1 // 2))

        @test (@constinferred tensor_product(pha, pha)) ==
            gradedrange([×((; A = SU2(0))) => 1, ×((; A = SU2(1))) => 1])
        @test (@constinferred tensor_product(pha, p0)) == gradedrange([pha => 1])
        @test (@constinferred tensor_product(p0, phb)) == gradedrange([phb => 1])
        @test (@constinferred tensor_product(pha, phb)) == gradedrange([phab => 1])

        @test tensor_product(phab, phab) == gradedrange(
            [
                (; A = SU2(0)) × (; B = SU2(0)) => 1,
                (; A = SU2(1)) × (; B = SU2(0)) => 1,
                (; A = SU2(0)) × (; B = SU2(1)) => 1,
                (; A = SU2(1)) × (; B = SU2(1)) => 1,
            ]
        )
    end

    @testset "Fusion of mixed Abelian and NonAbelian products" begin
        q0h = ×((; J = SU2(1 // 2)))
        q10 = (N = U1(1),) × (J = SU2(0),)
        # Put names in reverse order sometimes:
        q1h = (J = SU2(1 // 2),) × (N = U1(1),)
        q11 = (N = U1(1),) × (J = SU2(1),)
        q20 = (N = U1(2),) × (J = SU2(0),)
        q2h = (N = U1(2),) × (J = SU2(1 // 2),)
        q21 = (N = U1(2),) × (J = SU2(1),)
        q22 = (N = U1(2),) × (J = SU2(2),)

        @test tensor_product(q1h, q1h) == gradedrange([q20 => 1, q21 => 1])
        @test tensor_product(q10, q1h) == gradedrange([q2h => 1])
        @test (@constinferred tensor_product(q0h, q1h)) == gradedrange([q10 => 1, q11 => 1])
        @test tensor_product(q11, q11) == gradedrange([q20 => 1, q21 => 1, q22 => 1])
    end

    @testset "GradedOneTo fusion rules" begin
        s1 = (; A = U1(1)) × (; B = SU2(1 // 2))
        s2 = (; A = U1(0)) × (; B = SU2(1 // 2))
        g1 = gradedrange([s1 => 2])
        g2 = gradedrange([s2 => 1])
        s3 = (; A = U1(1)) × (; B = SU2(0))
        s4 = (; A = U1(1)) × (; B = SU2(1))
        @test tensor_product(g1, g2) == gradedrange([s3 => 2, s4 => 2])

        sA = ×((; A = U1(1)))
        sB = ×((; B = SU2(1 // 2)))
        sAB = (; A = U1(1)) × (; B = SU2(1 // 2))
        gA = gradedrange([sA => 2])
        gB = gradedrange([sB => 1])
        @test tensor_product(gA, gB) == gradedrange([sAB => 2])
    end
end

@testset "Mixing implementations" begin
    st1 = Sector((U1(1),))
    sA1 = Sector(; A = U1(1))

    # The two indexings describe different objects, so they never compare equal and neither
    # multiplying nor fusing them has a meaning: the result would have to be indexed both ways.
    @test sA1 != st1
    @test st1 != sA1
    @test !isequal(sA1, st1)
    @test hash(sA1) != hash(st1)
    @test_throws ArgumentError st1 × sA1
    @test_throws ArgumentError sA1 × st1
    @test_throws ArgumentError tensor_product(st1, sA1)
    @test_throws ArgumentError tensor_product(sA1, st1)
    # Ordering across the two is arbitrary, but it is still an order, so mixed vectors sort.
    @test isless(sA1, st1) != isless(st1, sA1)
end

@testset "Comparison does not cross the library boundary" begin
    # A product and the TensorKitSectors sector it converts to are values of two libraries'
    # types. `Sector` is how you cross, and the two conversions are inverses.
    for s in (Sector((U1(1),)), Sector(U1(1), U1(2)), fU1(2), Sector(; A = U1(1)))
        c = TKS.Sector(s)
        @test s != c
        @test c != s
        @test Sector(c) == s
    end
    @test fU1(2) != TKS.FermionNumber(2)
    @test Sector(TKS.FermionNumber(2)) == fU1(2)
    @test fSU2(1 // 2) != TKS.FermionSpin(1 // 2)
    @test Sector(TKS.FermionSpin(1 // 2)) == fSU2(1 // 2)
end

@testset "fSU2" begin
    # The parity follows from the spin, odd exactly when `2j` is, so the constructor takes only
    # the spin and the second factor is not free to disagree with the first.
    @test fSU2(1 // 2) == Sector(SU2(1 // 2), fZ2(true))
    @test fSU2(1) == Sector(SU2(1), fZ2(false))
    @test fSU2(3 // 2) == Sector(SU2(3 // 2), fZ2(true))
    @test fSU2(1 // 2) != fSU2(1)
    @test sectortype(fSU2(1 // 2)) == fSU2
    @test istrivial(fSU2(0))
    @test (@constinferred trivial(fSU2(1 // 2))) == fSU2(0)
    @test (@constinferred length(fSU2(1 // 2))) == 2

    # A spin-parity pair the constructor would never produce is not a fermion spin, so it keeps
    # the component spelling rather than borrowing the alias.
    @test sprint(show, Sector(SU2(1 // 2), fZ2(false))) == "(SU2(1/2) × fZ2(0))"

    g = gradedrange([fSU2(0) => 1, fSU2(1 // 2) => 2])
    @test sprint(show, g) == "gradedrange([fSU2(0) => 1, fSU2(1/2) => 2])"
end

@testset "Trivial as the unit of the product" begin
    st1 = Sector((U1(1),))
    sA1 = Sector(; A = U1(1))
    u = Trivial()

    @test ×() ≡ u
    @test (@constinferred flip(dual(u))) == u
    @test (@constinferred trivial(u)) == u
    @test (@constinferred length(u)) == 1

    @test (@constinferred u × u) ≡ u
    @test (@constinferred u × U1(1)) ≡ U1(1)
    @test (@constinferred U1(1) × u) ≡ U1(1)
    @test (@constinferred u × st1) ≡ st1
    @test (@constinferred st1 × u) ≡ st1
    @test (@constinferred u × sA1) ≡ sA1
    @test (@constinferred sA1 × u) ≡ sA1

    @test (@constinferred tensor_product(u, U1(1))) == U1(1)
    @test (@constinferred tensor_product(U1(1), u)) == U1(1)
    @test (@constinferred tensor_product(st1, u)) == st1
    @test (@constinferred tensor_product(u, st1)) == st1
    @test (@constinferred tensor_product(sA1, u)) == sA1
    @test (@constinferred tensor_product(u, sA1)) == sA1
    @test (@constinferred tensor_product(Sector((SU2(0),)), u)) ==
        gradedrange([Sector((SU2(0),)) => 1])
    @test (@constinferred tensor_product(u, Sector((SU2(0),)))) ==
        gradedrange([Sector((SU2(0),)) => 1])
    @test (@constinferred tensor_product(Sector(SU2(1), U1(2)), u)) ==
        gradedrange([Sector(SU2(1), U1(2)) => 1])
    @test (@constinferred tensor_product(Sector(; A = SU2(0)), u)) ==
        gradedrange([Sector(; A = SU2(0)) => 1])
    @test (@constinferred tensor_product(Sector(; B = SU2(1), C = U1(2)), u)) ==
        gradedrange([Sector(; B = SU2(1), C = U1(2)) => 1])

    g0 = gradedrange([u => 2])
    @test (@constinferred tensor_product(g0, g0)) == gradedrange([u => 4])

    # Equal only to itself, including against products whose every argument is trivial.
    @test u != U1(0)
    @test u != st1
    @test u != sA1
    @test u != Sector(())
    @test u != Sector((;))
    @test u < st1
    @test u < sA1
    @test st1 > u
end

@testset "Empty products" begin
    # An explicit empty container specifies a product of that shape. Specifying no symmetry at
    # all is `Trivial`, which is a different thing.
    for s in (Sector(()), Sector((;)))
        @test s != Trivial()
        @test Trivial() != s
        @test istrivial(s)
        @test (@constinferred flip(dual(s))) == s
        @test (@constinferred trivial(s)) == s
        @test (@constinferred length(s)) == 1
        @test (@constinferred s × Trivial()) ≡ s
        @test (@constinferred Trivial() × s) ≡ s
        @test (@constinferred tensor_product(s, s)) == s

        g0 = gradedrange([s => 2])
        @test (@constinferred tensor_product(g0, g0)) == gradedrange([s => 4])
    end

    @test Sector(()) != Sector((;))
    @test Sector((;)) != Sector(())

    # The empty positional product absorbs positional factors and the empty named one absorbs
    # named factors. Neither absorbs the other kind.
    @test (@constinferred Sector(()) × U1(1)) == Sector((U1(1),))
    @test (@constinferred U1(1) × Sector(())) == Sector((U1(1),))
    @test (@constinferred Sector(()) × Sector((U1(1),))) == Sector((U1(1),))
    @test (@constinferred Sector((;)) × Sector(; A = U1(1))) == Sector(; A = U1(1))
    @test_throws ArgumentError Sector((;)) × U1(1)
    @test_throws ArgumentError Sector(()) × Sector(; A = U1(1))

    # A name a product does not carry is that symmetry's trivial sector, so the empty named
    # product equals any named product whose arguments are all trivial. The positional one has no
    # such rule, since its arity is part of its identity.
    @test Sector((;)) == Sector(; A = U1(0))
    @test Sector(; A = U1(0)) == Sector((;))
    @test hash(Sector((;))) == hash(Sector(; A = U1(0)))
    @test Sector((;)) != Sector(; A = U1(1))
    @test Sector(()) != Sector((U1(0),))
end
