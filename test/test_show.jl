using BlockArrays: Block
using GradedArrays: GradedArrays, CU1, FusedGradedMatrix, FusedSectorMatrix, GradedOneTo,
    SU, SU2, Sector, SectorOneTo, Trivial, U1, UniqueSectorArray, Z, dual, fSU2, fU1, fZ2,
    fusedgradedmatrix, gradedrange, with_scalar_indexing, ×
using TensorKitSectors: TensorKitSectors as TKS, FermionParity, U1Irrep, ⊠
using Test: @test, @testset

@testset "show SymmetrySector" begin
    q1 = U1(1)
    @test sprint(show, q1) == "U1(1)"

    j1 = SU2(0)
    @test sprint(show, j1) == "SU2(0)"

    @test sprint(show, Trivial()) == "Trivial()"

    # A sector displays as the constructor call that rebuilds it, so several labels print as
    # several arguments.
    @test sprint(show, CU1(0, 0)) == "CU1(0, 0)"
    @test sprint(show, CU1(0, 1)) == "CU1(0, 1)"
    @test sprint(show, CU1(1 // 2)) == "CU1(1/2, 2)"
    @test sprint(show, SU{3}(1, 0, 0)) == "SU{3}(1, 0)"
    @test sprint(show, SU{4}(2, 1, 1, 0)) == "SU{4}(1, 0, 1)"

    # Each form prints the spelling that reconstructs it: the infix product for two or more
    # positional factors, and the explicit `Sector` call for everything `×` cannot spell.
    s = (A = U1(1),) × (B = SU2(2),)
    @test sprint(show, s) == "Sector(; A = U1(1), B = SU2(2))"
    s = Trivial() × U1(3) × SU2(1 / 2)
    @test sprint(show, s) == "(U1(3) × SU2(1/2))"
    @test sprint(show, Sector((U1(3),))) == "Sector((U1(3),))"
    @test sprint(show, Sector(())) == "Sector(())"
    @test sprint(show, Sector((;))) == "Sector((;))"
end

@testset "compact display of Z, FermionParity, and product sectors" begin
    @test sprint(show, Z{2}(1)) == "Z{2}(1)"
    @test sprint(show, fZ2(true)) == "fZ2(1)"

    fn = fU1(2)
    @test sprint(show, fn) == "fU1(2)"
    @test sprint(show, dual(fn)) == "dual(fU1(2))"
    fs = fSU2(1 // 2)
    @test sprint(show, fs) == "fSU2(1/2)"
    @test sprint(show, dual(fs)) == "dual(fSU2(1/2))"

    @test sprint(show, Sector(U1Irrep(1) ⊠ U1Irrep(2))) == "(U1(1) × U1(2))"
    # Parity 1 disagrees with the even charge 2, so this is not a `FermionNumber`.
    @test sprint(show, Sector(U1Irrep(2) ⊠ FermionParity(1))) ==
        "(U1(2) × fZ2(1))"

    g = gradedrange([fU1(0) => 1, fU1(1) => 2])
    s = sprint(show, g)
    @test s == "gradedrange([fU1(0) => 1, fU1(1) => 2])"
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
    m = fusedgradedmatrix([U1(0), U1(1)] .=> [ones(2, 2), ones(3, 3)])
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

@testset "UniqueSectorArray display shows Kronecker structure" begin
    sa = UniqueSectorArray([1.0 2.0; 3.0 4.0], (U1(0), dual(U1(1))))
    s = sprint(show, sa)
    @test occursin("⊗", s)

    s_plain = sprint(show, MIME("text/plain"), sa)
    @test occursin("⊗", s_plain)
    @test occursin("$(typeof(sa))", s_plain)
end

@testset "FusedSectorMatrix display shows Kronecker structure" begin
    sm = FusedSectorMatrix([1.0 2.0; 3.0 4.0], U1(1))
    s = sprint(show, sm)
    @test occursin("⊗", s)

    s_plain = sprint(show, MIME("text/plain"), sm)
    @test occursin("⊗", s_plain)
    @test occursin("FusedSectorMatrix", s_plain)
    @test occursin("U1(1)", s_plain)
end

@testset "compact type summary in display header" begin
    m = fusedgradedmatrix([U1(0), U1(1)] .=> [ones(2, 2), ones(3, 3)])
    @test occursin(
        "FusedGradedMatrix{Float64, …, Vector{Float64}}",
        sprint(show, MIME("text/plain"), m)
    )
end

@testset "compact axis lines in array display" begin
    # The axis's own show is unchanged and still round-trips through the constructor.
    g = gradedrange([U1(0) => 2, U1(1) => 2])
    @test sprint(show, g) == "gradedrange([U1(0) => 2, U1(1) => 2])"
    @test sprint(show, dual(g)) == "dual(gradedrange([U1(0) => 2, U1(1) => 2]))"
end

@testset "GradedArray text/plain display" begin
    # A `GradedArray` prints a codomain/domain header, one axis line per leg, then the matricized
    # `FusedGradedMatrix`.
    g = gradedrange([U1(0) => 2, U1(1) => 2])
    a = zeros(Float64, (g,), (g,))
    with_scalar_indexing() do
        return a[1, 1] = 1.0
    end

    s = sprint(show, MIME("text/plain"), a)
    @test occursin("GradedArray (codomain 1, domain 1)", s)
    @test occursin("Codomain Dim 1: gradedrange([U1(0) => 2, U1(1) => 2])", s)
    @test occursin("Domain Dim 1: gradedrange([U1(0) => 2, U1(1) => 2])", s)
    # The matricized `FusedGradedMatrix` is shown below the header.
    @test occursin("FusedGradedMatrix", s)
    @test occursin("⋅", s)   # unstored blocks show as dots
    @test occursin("┼", s)   # block separators
    @test occursin("1.0", s) # stored value

    # The compact one-line `show` is the summary header.
    @test sprint(show, a) == "4×4 GradedArray (codomain 1, domain 1)"
end

@testset "FusedGradedMatrix text/plain display" begin
    m = fusedgradedmatrix([U1(0), U1(1)] .=> [[1.0 2.0; 3.0 4.0], [5.0 6.0; 7.0 8.0]])

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
