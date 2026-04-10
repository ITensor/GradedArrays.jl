using GradedArrays: Fib, Ising, O2, SU2, SectorRange, TrivialSector, U1, Z, dual, flip,
    istrivial, modulus, sector_type, trivial, zero_odd
using SUNRepresentations: SUNRepresentations
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

const SU{N} = SectorRange{SUNRepresentations.SUNIrrep{N}}
fundamental(::Type{SU{N}}) where {N} = SU{N}((1, zeros(Int, N - 2)...))

@testset "Test SymmetrySectors Types" begin
    @testset "TrivialSector" begin
        q = TrivialSector()

        @test sector_type(q) === TrivialSector
        @test sector_type(typeof(q)) === TrivialSector
        @test (@constinferred length(q)) == 1
        @test q == q
        @test trivial(q) == q
        @test istrivial(q)

        @test dual(q) == q
        @test !isless(q, q)
    end

    @testset "U(1)" begin
        q1 = U1(1)
        q2 = U1(2)
        q3 = U1(3)

        @test sector_type(q1) === U1
        @test sector_type(typeof(q1)) === U1
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

        @test U1(0) == TrivialSector()
        @test TrivialSector() == U1(0)
        @test U1(-1) < TrivialSector()
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

        @test Z{2}(0) == TrivialSector()
        @test TrivialSector() < Z{2}(1)
        @test_throws MethodError U1(0) < Z{2}(1)
        @test Z{2}(0) != Z{2}(1)
        @test Z{2}(0) != Z{3}(0)
        @test Z{2}(0) != U1(0)
    end

    @testset "O(2)" begin
        s0e = O2(0)
        s0o = O2(-1)
        s12 = O2(1 // 2)
        s1 = O2(1)

        @test trivial(O2) == s0e
        @test istrivial(s0e)
        @test zero_odd(O2) == s0o

        @test (@constinferred length(s0e)) == 1
        @test (@constinferred length(s0o)) == 1
        @test (@constinferred length(s12)) == 2
        @test (@constinferred length(s1)) == 2

        @test (@constinferred flip(dual(s0e))) == s0e
        @test (@constinferred flip(dual(s0o))) == s0o
        @test (@constinferred flip(dual(s12))) == s12
        @test (@constinferred flip(dual(s1))) == s1

        @test s0e < s0o < s12 < s1
        @test s0e == TrivialSector()
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
        @test SU2(0) == TrivialSector()
        @test !(j2 < TrivialSector())
        @test TrivialSector() < j2
    end

    @testset "SU(N)" begin
        f3 = SU{3}((1, 0))
        f4 = SU{4}((1, 0, 0))
        ad3 = SU{3}((2, 1))
        ad4 = SU{4}((2, 1, 1))

        @test trivial(SU{3}) == SU{3}((0, 0))
        @test istrivial(SU{3}((0, 0)))
        @test trivial(SU{4}) == SU{4}((0, 0, 0))
        @test istrivial(SU{4}((0, 0, 0)))
        @test SU{3}((0, 0)) == TrivialSector()
        @test SU{4}((0, 0, 0)) == TrivialSector()

        @test fundamental(SU{3}) == f3
        @test fundamental(SU{4}) == f4

        @test flip(dual(f3)) == SU{3}((1, 1))
        @test flip(dual(f4)) == SU{4}((1, 1, 1))
        @test flip(dual(ad3)) == ad3
        @test flip(dual(ad4)) == ad4

        @test length(f3) == 3
        @test length(f4) == 4
        @test length(ad3) == 8
        @test length(ad4) == 15
        @test length(SU{3}((4, 2))) == 27
        @test length(SU{3}((3, 3))) == 10
        @test length(SU{3}((3, 0))) == 10
        @test length(SU{3}((0, 0))) == 1
        @test (@constinferred length(f3)) == 3
    end

    @testset "Fibonacci" begin
        ı = Fib("1")
        τ = Fib("τ")

        @test trivial(Fib) == ı
        @test istrivial(ı)
        @test ı == TrivialSector()

        @test flip(dual(ı)) == ı
        @test flip(dual(τ)) == τ

        @test (@constinferred length(ı)) == 1.0
        @test (@constinferred length(τ)) == ((1 + √5) / 2)

        @test ı < τ
    end

    @testset "Ising" begin
        ı = Ising("1")
        σ = Ising("σ")
        ψ = Ising("ψ")

        @test trivial(Ising) == ı
        @test istrivial(ı)
        @test ı == TrivialSector()

        @test flip(dual(ı)) == ı
        @test flip(dual(σ)) == σ
        @test flip(dual(ψ)) == ψ

        @test (@constinferred length(ı)) == 1.0
        @test (@constinferred length(σ)) == √2
        @test (@constinferred length(ψ)) == 1.0

        @test ı < σ < ψ
    end
end
