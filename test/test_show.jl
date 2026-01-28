using BlockArrays: BlockedOneTo, BlockedUnitRange
using GradedArrays: GradedArrays, Fib, GradedUnitRange, Ising, O2, SU2, SectorUnitRange,
    TrivialSector, U1, ×, gradedrange, sectorrange
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

@testset "show GradedUnitRange" begin
    x = U1(0)
    y = U1(1)
    z = U1(2)
    g1 = gradedrange([x => 2, y => 3, z => 2])
    @test sprint(show, g1) == "GradedUnitRange[$x => 2, $y => 3, $z => 2]"
    @test sprint(show, MIME("text/plain"), g1) ==
        "GradedUnitRange{$U1}\n" *
        "sectorrange($x => 1:2)\n" *
        "sectorrange($y => 3:5)\n" *
        "sectorrange($z => 6:7)"

    g1d = dual(g1)
    @test sprint(show, g1d) == "GradedUnitRange[$x' => 2, $y' => 3, $z' => 2]"
    @test sprint(show, MIME("text/plain"), g1d) ==
        "GradedUnitRange{$U1}\n" *
        "sectorrange($x' => 1:2)\n" *
        "sectorrange($y' => 3:5)\n" *
        "sectorrange($z' => 6:7)"
end

@testset "show GradedArray" begin
    elt = Float64
    r = gradedrange([U1(0) => 2, U1(1) => 2])

    a = zeros(elt, r)
    a[1] = one(elt)
    @test sprint(show, "text/plain", a) ==
        "2-blocked 4-element GradedVector{$(elt), …, …, …}:\n" *
        " $(one(elt))\n $(zero(elt))\n ───\n  ⋅ \n  ⋅ "

    a = zeros(elt, r, r)
    a[1, 1] = one(elt)
    @test sprint(show, "text/plain", a) ==
        "2×2-blocked 4×4 GradedMatrix{$(elt), …, …, …}:\n" *
        " $(one(elt))  $(zero(elt))  │   ⋅    ⋅ \n" *
        " $(zero(elt))  $(zero(elt))  │   ⋅    ⋅ \n" *
        " ──────────┼──────────\n  ⋅    ⋅   │   ⋅    ⋅ \n  ⋅    ⋅   │   ⋅    ⋅ "

    a = zeros(elt, r, r, r)
    a[1, 1, 1] = one(elt)
    @test sprint(show, "text/plain", a) ==
        "2×2×2-blocked 4×4×4 GradedArray{$(elt), 3, …, …, …}:\n" *
        "[:, :, 1] =\n $(one(elt))  $(zero(elt))   ⋅    ⋅ \n" *
        " $(zero(elt))  $(zero(elt))   ⋅    ⋅ \n" *
        "  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n" *
        "\n[:, :, 2] =\n $(zero(elt))  $(zero(elt))   ⋅    ⋅ \n" *
        " $(zero(elt))  $(zero(elt))   ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n" *
        "  ⋅    ⋅    ⋅    ⋅ \n\n[:, :, 3] =\n  ⋅    ⋅    ⋅    ⋅ \n" *
        "  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n" *
        "\n[:, :, 4] =\n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n" *
        "  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ "
end
