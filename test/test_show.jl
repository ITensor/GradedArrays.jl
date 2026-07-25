using BlockArrays: Block
using GradedArrays: GradedArrays, AbelianGradedArray, AbelianSectorArray, Fib,
    FusedGradedMatrix, GradedOneTo, Ising, O2, SU2, SectorMatrix, SectorOneTo, SectorRange,
    TrivialSector, U1, Z, dual, gradedrange, ×
using TensorKitSectors: TensorKitSectors as TKS, FermionNumber, FermionParity, U1Irrep, ⊠
using Test: @test, @testset

@testset "show SymmetrySector" begin
    q1 = U1(1)
    @test sprint(show, q1) == "U1(1)"

    s0e = O2(0)
    s0o = O2(-1)
    s12 = O2(1 // 2)
    s1 = O2(1)
    @test isnothing(show(devnull, [s0o, s0e, s12]))
    @test sprint(show, s0e) == "O2(0)"
    @test sprint(show, s0o) == "O2(-1)"
    @test sprint(show, s12) == "O2(1/2)"
    @test sprint(show, s0e) == "O2(0)"

    j1 = SU2(0)
    @test sprint(show, j1) == "SU2(0)"

    @test sprint(show, Fib.(("1", "τ"))) == "(Fib(\"1\"), Fib(\"τ\"))"
    @test sprint(show, Ising.(("1", "σ", "ψ"))) ==
        "(Ising(\"1\"), Ising(\"σ\"), Ising(\"ψ\"))"

    s = (A = U1(1),) × (B = SU2(2),)
    @test sprint(show, s) == "((A=U1(1),) × (B=SU2(2),))"
    s = TrivialSector() × U1(3) × SU2(1 / 2)
    @test sprint(show, s) == "(TrivialSector() × U1(3) × SU2(1/2))"
end

@testset "compact display of Z, FermionParity, and product sectors" begin
    @test sprint(show, Z{2}(1)) == "Z{2}(1)"
    @test sprint(show, SectorRange(FermionParity(1))) == "FermionParity(1)"

    fn = SectorRange(FermionNumber(2))
    @test sprint(show, fn) == "FermionNumber(2)"
    @test sprint(show, dual(fn)) == "dual(FermionNumber(2))"

    @test sprint(show, SectorRange(U1Irrep(1) ⊠ U1Irrep(2))) == "(U1(1) × U1(2))"
    # Parity 1 disagrees with the even charge 2, so this is not a `FermionNumber`.
    @test sprint(show, SectorRange(U1Irrep(2) ⊠ FermionParity(1))) ==
        "(U1(2) × FermionParity(1))"

    g = gradedrange(
        [
            SectorRange(FermionNumber(0)) => 1,
            SectorRange(FermionNumber(1)) => 2,
        ]
    )
    s = sprint(show, g)
    @test s == "gradedrange([FermionNumber(0) => 1, FermionNumber(1) => 2])"
    @test !occursin("Irrep", s)
    @test !occursin("ProductSector", s)
    @test !occursin("GradedArrays.", s)
end

@testset "show GradedOneTo" begin
    x = U1(0)
    y = U1(1)
    z = U1(2)
    g1 = gradedrange([x => 2, y => 3, z => 2])
    @test g1 isa GradedOneTo

    @test sprint(show, g1) ==
        "gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])"

    # Duality is factored to the outside as `dual(gradedrange([...]))` rather
    # than decorated on each sector; we don't reuse `'` because Julia already
    # uses `'` for range adjoints.
    g1d = dual(g1)
    @test sprint(show, g1d) ==
        "dual(gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2]))"
end

@testset "GradedOneTo show uses compact sector format" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    s = sprint(show, g)
    @test s == "gradedrange([U1(0) => 2, U1(1) => 3])"
    @test sprint(show, dual(g)) ==
        "dual(gradedrange([U1(0) => 2, U1(1) => 3]))"
    @test !occursin("Irrep", s)
end

@testset "FusedGradedMatrix show uses compact sector format" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), ones(3, 3)])
    s = sprint(show, MIME("text/plain"), m)
    @test occursin("U1", s)
    @test !occursin("Irrep", s)
end

@testset "SectorOneTo show uses compact sector format" begin
    r = SectorOneTo(U1(1), 3)
    s = sprint(show, r)
    @test occursin("U1", s)
    @test !occursin("Irrep", s)
end

@testset "AbelianSectorArray display shows Kronecker structure" begin
    sa = AbelianSectorArray((U1(0), dual(U1(1))), [1.0 2.0; 3.0 4.0])
    s = sprint(show, sa)
    @test occursin("⊗", s)

    s_plain = sprint(show, MIME("text/plain"), sa)
    @test occursin("⊗", s_plain)
    @test occursin("AbelianSectorMatrix", s_plain)
end

@testset "SectorMatrix display shows Kronecker structure" begin
    sm = SectorMatrix(U1(1), [1.0 2.0; 3.0 4.0])
    s = sprint(show, sm)
    @test occursin("⊗", s)

    s_plain = sprint(show, MIME("text/plain"), sm)
    @test occursin("⊗", s_plain)
    @test occursin("SectorMatrix", s_plain)
    @test occursin("U1(1)", s_plain)
end

@testset "compact type summary in display header" begin
    g = gradedrange([U1(0) => 2, U1(1) => 2])
    s = sprint(show, MIME("text/plain"), zeros(Float64, g, dual(g)))
    @test occursin("AbelianGradedMatrix{Float64, …, Matrix{Float64}}", s)
    @test !occursin("SectorRange", s)
    @test !occursin("GradedArrays.", s)

    gf = gradedrange([FermionNumber(0) => 1, FermionNumber(1) => 2])
    sf = sprint(show, MIME("text/plain"), zeros(ComplexF64, gf, dual(gf)))
    @test occursin("AbelianGradedMatrix{ComplexF64, …, Matrix{ComplexF64}}", sf)
    @test !occursin("ProductSector", sf)
    @test !occursin("Irrep", sf)

    # Higher-order arrays keep the order `N` explicit in the header.
    s3 = sprint(show, MIME("text/plain"), zeros(Float32, g, dual(g), g))
    @test occursin("AbelianGradedArray{Float32, …, 3, Array{Float32, 3}}", s3)

    m = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), ones(3, 3)])
    @test occursin(
        "FusedGradedMatrix{Float64, …, Matrix{Float64}}",
        sprint(show, MIME("text/plain"), m)
    )
end

@testset "AbelianGradedArray text/plain display" begin
    g = gradedrange([U1(0) => 2, U1(1) => 2])
    a = zeros(Float64, g, dual(g))
    a[Block(1, 1)] = [1.0 2.0; 3.0 4.0]
    a[Block(2, 2)] = [5.0 6.0; 7.0 8.0]

    s = sprint(show, MIME("text/plain"), a)
    @test occursin("AbelianGradedMatrix", s)
    # Unstored blocks show as dots
    @test occursin("⋅", s)
    # Block separators
    @test occursin("│", s)
    @test occursin("─", s)
    @test occursin("┼", s)
    # Stored values are present
    @test occursin("1.0", s)
    @test occursin("8.0", s)
end

@testset "FusedGradedMatrix text/plain display" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]])

    s = sprint(show, MIME("text/plain"), m)
    @test occursin("FusedGradedMatrix", s)
    # Unstored blocks show as dots
    @test occursin("⋅", s)
    # Block separators
    @test occursin("│", s)
    @test occursin("─", s)
    @test occursin("┼", s)
    # Stored values are present
    @test occursin("1.0", s)
    @test occursin("8.0", s)
end
