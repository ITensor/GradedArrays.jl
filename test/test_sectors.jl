using GradedArrays: CU1, SU2, SUN, Sector, TensorKitSector, TrivialSector, U1, Z, dual,
    flip, istrivial, label, modulus, sectortype, trivial
using SUNRepresentations: SUNRepresentations
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

fundamental(::Type{SUN{N}}) where {N} = SUN{N}((1, zeros(Int, N - 2)...))

@testset "Test SymmetrySectors Types" begin
    @testset "TrivialSector" begin
        q = TrivialSector()

        @test sectortype(q) === TrivialSector
        @test sectortype(typeof(q)) === TrivialSector
        @test (@constinferred length(q)) == 1
        @test q == q
        @test trivial(q) == q
        @test istrivial(q)

        @test q != U1(0)
        @test U1(0) != q
        @test q != Sector(U1(0), SU2(0))
        @test Sector(U1(0), SU2(0)) != q
        @test q != Sector(; Nf = U1(0))
        @test Sector(; Nf = U1(0)) != q

        @test flip(dual(q)) == q
        @test !isless(q, q)
    end

    @testset "U(1)" begin
        q1 = U1(1)
        q2 = U1(2)
        q3 = U1(3)

        @test sectortype(q1) === U1
        @test sectortype(typeof(q1)) === U1
        @test length(q1) == 1
        @test length(q2) == 1
        @test (@constinferred length(q1)) == 1

        @test trivial(q1) == U1(0)
        @test trivial(U1) == U1(0)
        @test istrivial(U1(0))

        @test flip(dual(U1(2))) == U1(-2)
        @test isless(U1(1), U1(2))
        @test !isless(U1(2), U1(1))
        @test U1(Int8(1)) == U1(1)
        @test U1(UInt32(1)) == U1(1)

        @test TrivialSector() < U1(-1)
        @test TrivialSector() < U1(1)
        @test U1(Int8(1)) < U1(Int32(2))
    end

    @testset "Z₂" begin
        z0 = Z{2}(0)
        z1 = Z{2}(1)

        @test trivial(Z{2}) == Z{2}(0)
        @test istrivial(Z{2}(0))

        @test length(z0) == 1
        @test length(z1) == 1
        @test (@constinferred length(z0)) == 1

        @test flip(dual(z0)) == z0
        @test flip(dual(z1)) == z1
        @test modulus(z1) == 2

        @test isless(Z{2}(0), Z{2}(1))
        @test !isless(Z{2}(1), Z{2}(0))
        @test Z{2}(0) == z0
        @test Z{2}(-3) == z1

        @test TrivialSector() < Z{2}(1)
        @test_throws MethodError U1(0) < Z{2}(1)
        @test Z{2}(0) != Z{2}(1)
        @test Z{2}(0) != Z{3}(0)
        @test Z{2}(0) != U1(0)
    end

    @testset "O(2)" begin
        s0e = CU1(0, 0)
        s0o = CU1(0, 1)
        s12 = CU1(1 // 2, 2)
        s1 = CU1(1, 2)

        # `s` follows from `j` except at `j == 0`, so it defaults.
        @test CU1(1 // 2) == s12
        @test CU1(0) == s0e
        # An upstream sector converts to this type rather than to the escape hatch.
        @test Sector(TKS.CU1Irrep(0, 1)) === s0o
        # Upstream rejects the pairs that are not irreps.
        @test_throws ErrorException CU1(0, 2)
        @test_throws ErrorException CU1(1, 0)

        @test trivial(CU1) == s0e
        @test istrivial(s0e)
        @test !istrivial(s0o)

        @test (@constinferred length(s0e)) == 1
        @test (@constinferred length(s0o)) == 1
        @test (@constinferred length(s12)) == 2
        @test (@constinferred length(s1)) == 2

        @test (@constinferred flip(dual(s0e))) == s0e
        @test (@constinferred flip(dual(s0o))) == s0o
        @test (@constinferred flip(dual(s12))) == s12
        @test (@constinferred flip(dual(s1))) == s1

        @test s0e < s0o < s12 < s1
        @test s0o > TrivialSector()
        @test TrivialSector() < s12
    end

    @testset "SU(2)" begin
        j1 = SU2(0)
        j2 = SU2(1 // 2)  # Rational will be cast to HalfInteger
        j3 = SU2(1)
        j4 = SU2(3 // 2)

        # alternative constructors
        @test j2 == SU2(1 / 2)  # Float will be cast to HalfInteger
        @test_throws MethodError SU2((1,))  # avoid confusion between tuple and half-integer interfaces

        @test trivial(SU2) == SU2(0)
        @test istrivial(SU2(0))

        @test length(j1) == 1
        @test length(j2) == 2
        @test length(j3) == 3
        @test length(j4) == 4
        @test (@constinferred length(j1)) == 1

        @test flip(dual(j1)) == j1
        @test flip(dual(j2)) == j2
        @test flip(dual(j3)) == j3
        @test flip(dual(j4)) == j4

        @test j1 < j2 < j3 < j4
        @test !(j2 < TrivialSector())
        @test TrivialSector() < j2
    end

    @testset "SU(N)" begin
        f3 = SUN{3}((1, 0))
        f4 = SUN{4}((1, 0, 0))
        ad3 = SUN{3}((2, 1))
        ad4 = SUN{4}((2, 1, 1))

        # Both spellings, and the trailing zero the shorter one implies.
        @test SUN{3}((1, 0)) === SUN{3}((1, 0, 0))
        @test label(f3) == (1, 0, 0)

        # Adding a constant to every entry is the same representation, so the weight is stored
        # shifted to end in zero and equality is equality of representations.
        @test SUN{3}((2, 1, 1)) == SUN{3}((1, 0, 0))
        @test label(SUN{3}((2, 1, 1))) == (1, 0, 0)
        @test hash(SUN{3}((2, 1, 1))) == hash(SUN{3}((1, 0, 0)))
        @test istrivial(SUN{3}((1, 1, 1)))

        # A weight must be non-increasing, and must have N or N-1 entries.
        @test_throws ArgumentError SUN{3}((0, 1, 0))
        @test_throws ArgumentError SUN{3}((1, 2))
        @test_throws ArgumentError SUN{3}((1, 0, 0, 0))

        @test trivial(SUN{3}) == SUN{3}((0, 0))
        @test istrivial(SUN{3}((0, 0)))
        @test trivial(SUN{4}) == SUN{4}((0, 0, 0))
        @test istrivial(SUN{4}((0, 0, 0)))

        @test fundamental(SUN{3}) == f3
        @test fundamental(SUN{4}) == f4

        @test flip(dual(f3)) == SUN{3}((1, 1))
        @test flip(dual(f4)) == SUN{4}((1, 1, 1))
        @test flip(dual(ad3)) == ad3
        @test flip(dual(ad4)) == ad4

        @test length(f3) == 3
        @test length(f4) == 4
        @test length(ad3) == 8
        @test length(ad4) == 15
        @test length(SUN{3}((4, 2))) == 27
        @test length(SUN{3}((3, 3))) == 10
        @test length(SUN{3}((3, 0))) == 10
        @test length(SUN{3}((0, 0))) == 1
        @test (@constinferred length(f3)) == 3
    end

    @testset "Fibonacci" begin
        ı = Sector(TKS.FibonacciAnyon(:I))
        τ = Sector(TKS.FibonacciAnyon(:τ))

        @test trivial(TensorKitSector{TKS.FibonacciAnyon}) == ı
        @test istrivial(ı)

        @test flip(dual(ı)) == ı
        @test flip(dual(τ)) == τ

        @test (@constinferred length(ı)) == 1.0
        @test (@constinferred length(τ)) == ((1 + √5) / 2)

        @test ı < τ
    end

    @testset "Ising" begin
        ı = Sector(TKS.IsingAnyon(:I))
        σ = Sector(TKS.IsingAnyon(:σ))
        ψ = Sector(TKS.IsingAnyon(:ψ))

        @test trivial(TensorKitSector{TKS.IsingAnyon}) == ı
        @test istrivial(ı)

        @test flip(dual(ı)) == ı
        @test flip(dual(σ)) == σ
        @test flip(dual(ψ)) == ψ

        @test (@constinferred length(ı)) == 1.0
        @test (@constinferred length(σ)) == √2
        @test (@constinferred length(ψ)) == 1.0

        @test ı < σ < ψ
    end
end
