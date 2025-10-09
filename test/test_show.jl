# test show separately as it may behave differently locally and on CI.
# sometimes displays GradedArrays.GradedUnitRange and sometimes GradedUnitRange depending
# on exact setup

using Test: @test, @testset
using GradedArrays: ×, Fib, Ising, O2, SU, SU2, TrivialSector, U1, gradedrange, sectorrange

@testset "show SymmetrySector" begin
    q1 = U1(1)
    @test sprint(show, q1) == "U1(1)"

    s0e = O2(0)
    s0o = O2(-1)
    s12 = O2(1 // 2)
    s1 = O2(1)
    @test isnothing(show(devnull, [s0o, s0e, s12]))
    @test sprint(show, s0e) == "O2(0)"
    @test sprint(show, MIME("text/plain"), s0e) == "O2(0e)"
    @test sprint(show, s0o) == "O2(-1)"
    @test sprint(show, MIME("text/plain"), s0o) == "O2(0o)"
    @test sprint(show, s12) == "O2(1/2)"
    @test sprint(show, MIME("text/plain"), s12) == "O2(±1/2)"

    j1 = SU2(0)
    @test sprint(show, j1) == "SU2(0)"
    @test sprint(show, MIME("text/plain"), j1) == "S = 0"

    f3 = SU{3}((1, 0))
    ad3 = SU{3}((2, 1))
    @test sprint(show, f3) == "SU{3}((1, 0))"
    @test sprint(show, MIME("text/plain"), f3) == "┌─┐\n└─┘"
    @test sprint(show, MIME("text/plain"), ad3) == "┌─┬─┐\n├─┼─┘\n└─┘"

    @test sprint(show, Fib.(("1", "τ"))) == "(Fib(1), Fib(τ))"
    @test sprint(show, Ising.(("1", "σ", "ψ"))) == "(Ising(1), Ising(σ), Ising(ψ))"

    s = (A = U1(1),) × (B = SU2(2),)
    @test sprint(show, s) == "((A=U1(1),) × (B=SU2(2),))"
    s = TrivialSector() × U1(3) × SU2(1 / 2)
    @test sprint(show, s) == "(GradedArrays.TrivialSector() × U1(3) × SU2(1/2))"
end

@testset "show GradedUnitRange" begin
    sr = sectorrange(SU((1, 0)), 2)
    @test sprint(show, sr) == "SectorUnitRange SU{3}((1, 0)) => Base.OneTo(6)"

    g1 = gradedrange(["x" => 2, "y" => 3, "z" => 2])
    @test sprint(show, g1) == "GradedUnitRange[\"x\" => 2, \"y\" => 3, \"z\" => 2]"
    @test sprint(show, MIME("text/plain"), g1) ==
        "GradedArrays.GradedUnitRange{Int64, GradedArrays.SectorUnitRange{Int64, String, Base.OneTo{Int64}}, BlockArrays.BlockedOneTo{Int64, Vector{Int64}}, Vector{Int64}}\nSectorUnitRange x => 1:2\nSectorUnitRange y => 3:5\nSectorUnitRange z => 6:7"

    g2 = gradedrange(1, ["x" => 2, "y" => 3, "z" => 2])
    @test sprint(show, g2) == "GradedUnitRange[\"x\" => 2, \"y\" => 3, \"z\" => 2]"
    @test sprint(show, MIME("text/plain"), g2) ==
        "GradedArrays.GradedUnitRange{Int64, GradedArrays.SectorUnitRange{Int64, String, Base.OneTo{Int64}}, BlockArrays.BlockedUnitRange{Int64, Vector{Int64}}, Vector{Int64}}\nSectorUnitRange x => 1:2\nSectorUnitRange y => 3:5\nSectorUnitRange z => 6:7"

    g1d = gradedrange(["x" => 2, "y" => 3, "z" => 2]; isdual = true)
    @test sprint(show, g1d) == "GradedUnitRange dual [\"x\" => 2, \"y\" => 3, \"z\" => 2]"
    @test sprint(show, MIME("text/plain"), g1d) ==
        "GradedArrays.GradedUnitRange{Int64, GradedArrays.SectorUnitRange{Int64, String, Base.OneTo{Int64}}, BlockArrays.BlockedOneTo{Int64, Vector{Int64}}, Vector{Int64}}\nSectorUnitRange dual(x) => 1:2\nSectorUnitRange dual(y) => 3:5\nSectorUnitRange dual(z) => 6:7"

    g = gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 2])
    @test sprint(show, "text/plain", g) ==
        "GradedArrays.GradedUnitRange{Int64, GradedArrays.SectorUnitRange{Int64, GradedArrays.SU{3, 2}, Base.OneTo{Int64}}, BlockArrays.BlockedOneTo{Int64, Vector{Int64}}, Vector{Int64}}\nSectorUnitRange SU{3}((0, 0)) => 1:2\nSectorUnitRange SU{3}((1, 0)) => 3:8"
    @test sprint(show, g) == "GradedUnitRange[SU{3}((0, 0)) => 2, SU{3}((1, 0)) => 2]"
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
