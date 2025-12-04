# test show separately as it may behave differently locally and on CI.
# sometimes displays GradedArrays.GradedUnitRange and sometimes GradedUnitRange depending
# on exact setup

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
    @test sprint(show, s) == "(TrivialSector() × $U1(3) × $SU2(1/2))"
end

@testset "show GradedUnitRange" begin
    g1 = gradedrange(["x" => 2, "y" => 3, "z" => 2])
    @test sprint(show, g1) == "$GradedUnitRange[\"x\" => 2, \"y\" => 3, \"z\" => 2]"
    @test sprint(show, MIME("text/plain"), g1) ==
        "$GradedUnitRange{Int64, $SectorUnitRange{Int64, String, Base.OneTo{Int64}}, $BlockedOneTo{Int64, Vector{Int64}}, Vector{Int64}}\n$SectorUnitRange x => 1:2\n$SectorUnitRange y => 3:5\n$SectorUnitRange z => 6:7"

    g2 = gradedrange(1, ["x" => 2, "y" => 3, "z" => 2])
    @test sprint(show, g2) == "GradedUnitRange[\"x\" => 2, \"y\" => 3, \"z\" => 2]"
    @test sprint(show, MIME("text/plain"), g2) ==
        "$GradedUnitRange{Int64, $SectorUnitRange{Int64, String, Base.OneTo{Int64}}, $BlockedUnitRange{Int64, Vector{Int64}}, Vector{Int64}}\n$SectorUnitRange x => 1:2\n$SectorUnitRange y => 3:5\n$SectorUnitRange z => 6:7"

    g1d = gradedrange(["x" => 2, "y" => 3, "z" => 2]; isdual = true)
    @test sprint(show, g1d) == "$GradedUnitRange dual [\"x\" => 2, \"y\" => 3, \"z\" => 2]"
    @test sprint(show, MIME("text/plain"), g1d) ==
        "$GradedUnitRange{Int64, $SectorUnitRange{Int64, String, Base.OneTo{Int64}}, $BlockedOneTo{Int64, Vector{Int64}}, Vector{Int64}}\n$SectorUnitRange dual(x) => 1:2\n$SectorUnitRange dual(y) => 3:5\n$SectorUnitRange dual(z) => 6:7"
end

@testset "show GradedArray" begin
    elt = Float64
    r = gradedrange([U1(0) => 2, U1(1) => 2])

    a = zeros(elt, r)
    a[1] = one(elt)
    @test sprint(show, "text/plain", a) ==
        "2-blocked 4-element GradedVector{$(elt), Vector{$(elt)}, …, …}:\n $(one(elt))\n $(zero(elt))\n ───\n  ⋅ \n  ⋅ "

    a = zeros(elt, r, r)
    a[1, 1] = one(elt)
    @test sprint(show, "text/plain", a) ==
        "2×2-blocked 4×4 GradedMatrix{$(elt), Matrix{$(elt)}, …, …}:\n $(one(elt))  $(zero(elt))  │   ⋅    ⋅ \n $(zero(elt))  $(zero(elt))  │   ⋅    ⋅ \n ──────────┼──────────\n  ⋅    ⋅   │   ⋅    ⋅ \n  ⋅    ⋅   │   ⋅    ⋅ "

    a = zeros(elt, r, r, r)
    a[1, 1, 1] = one(elt)
    @test sprint(show, "text/plain", a) ==
        "2×2×2-blocked 4×4×4 GradedArray{$(elt), 3, Array{$(elt), 3}, …, …}:\n[:, :, 1] =\n $(one(elt))  $(zero(elt))   ⋅    ⋅ \n $(zero(elt))  $(zero(elt))   ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n\n[:, :, 2] =\n $(zero(elt))  $(zero(elt))   ⋅    ⋅ \n $(zero(elt))  $(zero(elt))   ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n\n[:, :, 3] =\n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n\n[:, :, 4] =\n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅    ⋅ "
end
