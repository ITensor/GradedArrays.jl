using GradedArrays: GradedArrays, Fib, FusedGradedMatrix, GradedOneTo, Ising, O2, SU2,
    SectorOneTo, TrivialSector, U1, dual, gradedrange
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

    @test sprint(show, g1) ==
        "GradedOneTo([$U1(0) => 2, $U1(1) => 3, $U1(2) => 2])"

    g1d = dual(g1)
    @test sprint(show, g1d) ==
        "GradedOneTo([$U1(0) => 2, $U1(1) => 3, $U1(2) => 2])'"
end

@testset "GradedOneTo show uses compact sector format" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    s = sprint(show, g)
    @test s == "GradedOneTo([$U1(0) => 2, $U1(1) => 3])"
    @test sprint(show, dual(g)) == "GradedOneTo([$U1(0) => 2, $U1(1) => 3])'"
    @test !occursin("Irrep", s)
end

@testset "FusedGradedMatrix show uses compact sector format" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), ones(3, 3)])
    s = sprint(show, MIME("text/plain"), m)
    @test occursin("$U1", s)
    @test !occursin("Irrep", s)
end

@testset "SectorOneTo show uses compact sector format" begin
    r = SectorOneTo(U1(1), 3)
    s = sprint(show, r)
    @test occursin("$U1", s)
    @test !occursin("Irrep", s)
end
