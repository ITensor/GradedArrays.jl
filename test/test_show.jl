using BlockArrays: Block
using GradedArrays: GradedArrays, CU1, FusedGradedMatrix, FusedSectorMatrix, GradedOneTo,
    SU, SU2, Sector, SectorOneTo, TensorKitSector, Trivial, U1, UniqueSectorArray, Z, Z2,
    dual, fSU2, fU1, fZ2, fusedgradedmatrix, fusedgradedvector, gradedrange,
    with_scalar_indexing, ×
using TensorAlgebra: matricize
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

    # Each form prints a spelling that reads back in: the infix product for two or more
    # positional factors, the bare `NamedTuple` for a named one, and the explicit `Sector` call
    # for what neither can spell.
    s = (A = U1(1),) × (B = SU2(2),)
    @test sprint(show, s) == "(A = U1(1), B = SU2(2))"
    s = Trivial() × U1(3) × SU2(1 / 2)
    @test sprint(show, s) == "(U1(3) × SU2(1/2))"
    @test sprint(show, Sector(; A = U1(1))) == "(A = U1(1),)"
    @test sprint(show, Sector((U1(3),))) == "Sector((U1(3),))"
    @test sprint(show, Sector(())) == "Sector(())"
    @test sprint(show, Sector((;))) == "(;)"
end

@testset "compact display of Z, FermionParity, and product sectors" begin
    @test sprint(show, Z{3}(1)) == "Z{3}(1)"
    @test sprint(show, Z2(1)) == "Z2(1)"
    @test sprint(show, fZ2(true)) == "fZ2(1)"

    fn = fU1(2)
    @test sprint(show, fn) == "fU1(2)"
    @test sprint(show, dual(fn)) == "dual(fU1(2))"
    fs = fSU2(1 // 2)
    @test sprint(show, fs) == "fSU2(1/2)"
    @test sprint(show, dual(fs)) == "dual(fSU2(1/2))"

    # `Sector` unwraps a product into the native one, which is not the wrapped sector and does
    # not print as it: the wrapper is what tells the two apart.
    @test sprint(show, Sector(U1Irrep(1) ⊠ U1Irrep(2))) == "(U1(1) × U1(2))"
    @test startswith(
        sprint(show, TensorKitSector(U1Irrep(1) ⊠ U1Irrep(2))), "TensorKitSector("
    )
    @test Sector(U1Irrep(1) ⊠ U1Irrep(2)) != TensorKitSector(U1Irrep(1) ⊠ U1Irrep(2))
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
    g = gradedrange([U1(0) => 1, U1(1) => 2])
    for (a, spelling) in (
            (m, "FusedGradedMatrix{Float64, U1, Vector{Float64}, …}"),
            (g, "GradedOneTo{U1, …}"),
            (dual(g), "GradedOneTo{U1, …}"),
        )
        # Every parameter but the TensorKitSectors sector type, which shows as `…` so the
        # spelling does not read as the whole concrete type.
        s = sprint(show, MIME("text/plain"), a; context = :module => GradedArrays)
        @test occursin(spelling, s)
        @test !occursin("GradedArrays.", s)
        @test !occursin("TensorKitSectors.", s)
        # Reading from somewhere without the names in scope qualifies them, as it does for
        # any other type display. `Base` stands in for such a module.
        s_qualified = sprint(show, MIME("text/plain"), a; context = :module => Base)
        @test occursin("GradedArrays.", s_qualified)
        @test !occursin("TensorKitSectors.", s_qualified)
    end
end

@testset "product sector types in the display header" begin
    # A product type shows as the product of its factors, which gives the type back, rather than
    # as the struct and its parameter tuple.
    g = gradedrange([(charge = U1(0), spin = SU2(0)) => 1])
    s = sprint(show, MIME("text/plain"), g; context = :module => GradedArrays)
    @test occursin("GradedOneTo{(; charge = U1) × (; spin = SU2), …}", s)
    @test occursin("gradedrange([(charge = U1(0), spin = SU2(0)) => 1])", s)

    g = gradedrange([(U1(0) × SU2(0)) => 1])
    s = sprint(show, MIME("text/plain"), g; context = :module => GradedArrays)
    @test occursin("GradedOneTo{U1 × SU2, …}", s)

    # Fewer than two factors takes the container spelling, since the infix form would read as
    # the bare container rather than as the type.
    g = gradedrange([Sector((U1(0),)) => 1])
    s = sprint(show, MIME("text/plain"), g; context = :module => GradedArrays)
    @test occursin("GradedOneTo{×((U1,)), …}", s)

    g = gradedrange([Sector(; charge = U1(0)) => 1])
    s = sprint(show, MIME("text/plain"), g; context = :module => GradedArrays)
    @test occursin("GradedOneTo{×((; charge = U1)), …}", s)

    # An aliased product keeps its alias, which is the symmetry's own name.
    g = gradedrange([fU1(0) => 1])
    s = sprint(show, MIME("text/plain"), g; context = :module => GradedArrays)
    @test occursin("GradedOneTo{fU1, …}", s)
end

# The display edges: a group with no axes, a block set with nothing stored, and the adjoint,
# whose codomain and domain are its parent's swapped. Asserted on the labels and the line
# structure rather than on whole strings, since the type spelling depends on what the reader
# has in scope.
@testset "axis lines on the display edges" begin
    g1 = gradedrange([U1(0) => 1, U1(1) => 2])
    g2 = gradedrange([U1(0) => 2, U1(1) => 1, U1(2) => 1])

    # A vector has no domain, so the empty group prints nothing at all.
    v = fusedgradedvector([U1(0) => [1.0], U1(1) => [2.0, 3.0]])
    s = sprint(show, MIME("text/plain"), v)
    @test occursin("Codomain Dim 1: ", s)
    @test !occursin("Domain Dim", s)
    @test !endswith(s, "\n")

    # Nothing stored: the last axis line is the last line.
    for empty in (
            fusedgradedvector(Pair{U1, Vector{Float64}}[]),
            fusedgradedmatrix(Pair{U1, Matrix{Float64}}[]),
        )
        s = sprint(show, MIME("text/plain"), empty)
        @test occursin("Codomain Dim 1: ", s)
        @test !endswith(s, "\n")
        @test !occursin("\n\n", s)
    end

    # The adjoint shows axis lines of its own, swapped against its parent's.
    m = matricize(randn(Float64, (g1,), (g2,)))
    sm = sprint(show, MIME("text/plain"), m)
    sa = sprint(show, MIME("text/plain"), m')
    # The label is padded to the group's own width, so compare the axes without it.
    axis_of(str, label) = only(
        strip(split(l, ':'; limit = 2)[2])
            for l in split(str, '\n') if occursin(label, l)
    )
    @test axis_of(sa, "Codomain Dim 1:") == axis_of(sm, "Domain Dim 1:")
    @test axis_of(sa, "Domain Dim 1:") == axis_of(sm, "Codomain Dim 1:")
    @test !endswith(sa, "\n")

    # No legs at all, so no axis lines and no blank line before the matricized form.
    z = GradedArrays.GradedArray{Float64, U1}(undef, (), ())
    s = sprint(show, MIME("text/plain"), z)
    @test occursin("0-dimensional GradedArray (codomain 0, domain 0)", s)
    @test !occursin("\n\n", s)

    # The axis values line up. Each display block pads its own labels, and a graded array's
    # display nests the matricized form's block inside it, so this is the outer block alone,
    # which ends where that nested header begins.
    a = randn(Float64, (g1,), (g2,))
    lines = split(sprint(show, MIME("text/plain"), a), '\n')
    outer = lines[1:(findfirst(contains("FusedGradedMatrix"), lines) - 1)]
    columns = [
        first(findfirst("gradedrange(", l)) for l in outer if occursin(" Dim ", l)
    ]
    @test length(columns) == 2
    @test allequal(columns)
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
    # The axis values line up, so the narrower `Domain` label carries two spaces of padding.
    @test occursin("Domain Dim 1:   gradedrange([U1(0) => 2, U1(1) => 2])", s)
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
