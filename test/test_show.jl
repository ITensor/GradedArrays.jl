using GradedArrays:
    GradedArrays, Fib, GradedOneTo, Ising, O2, SU2, TrivialSector, U1, dual, gradedrange
using KroneckerArrays: ×
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

@testset "show SymmetrySector" begin
    q1 = U1(1)
    @test sprint(show, q1) == "$U1(1)"

    s0e = O2(0)
    s0o = O2(-1)
    s12 = O2(1 // 2)
    s1 = O2(1)
    @test isnothing(show(devnull, [s0o, s0e, s12]))
    @test sprint(show, s0e) == "$O2(0)"
    @test sprint(show, s0o) == "$O2(-1)"
    @test sprint(show, s12) == "$O2(1/2)"
    @test sprint(show, s0e) == "$O2(0)"

    j1 = SU2(0)
    @test sprint(show, j1) == "$SU2(0)"

    @test sprint(show, Fib.(("1", "τ"))) == "($Fib(\"1\"), $Fib(\"τ\"))"
    @test sprint(show, Ising.(("1", "σ", "ψ"))) ==
        "($Ising(\"1\"), $Ising(\"σ\"), $Ising(\"ψ\"))"

    s = (A = U1(1),) × (B = SU2(2),)
    @test sprint(show, s) == "((A=$U1(1),) × (B=$SU2(2),))"
    s = TrivialSector() × U1(3) × SU2(1 / 2)
    @test sprint(show, s) == "($TrivialSector() × $U1(3) × $SU2(1/2))"
end

@testset "show GradedOneTo" begin
    x = U1(0)
    y = U1(1)
    z = U1(2)
    g1 = gradedrange([x => 2, y => 3, z => 2])
    @test g1 isa GradedOneTo

    lx = repr(TKS.U1Irrep(0))
    ly = repr(TKS.U1Irrep(1))
    lz = repr(TKS.U1Irrep(2))
    @test sprint(show, g1) == "GradedOneTo([$lx => 2, $ly => 3, $lz => 2])"

    g1d = dual(g1)
    @test sprint(show, g1d) == "GradedOneTo([$lx => 2, $ly => 3, $lz => 2])'"
end
