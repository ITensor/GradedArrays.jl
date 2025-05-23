using GradedArrays:
  O2,
  SU,
  SU2,
  TrivialSector,
  U1,
  Z,
  dual,
  flip,
  gradedrange,
  nsymbol,
  quantum_dimension,
  space_isequal,
  trivial,
  unmerged_tensor_product
using TensorProducts: ⊗, tensor_product
using Test: @test, @testset, @test_throws
using TestExtras: @constinferred
using BlockArrays: blocklengths

@testset "Simple SymmetrySector fusion rules" begin
  @testset "Z{2} fusion rules" begin
    z0 = Z{2}(0)
    z1 = Z{2}(1)

    @test z0 ⊗ z0 == z0
    @test z0 ⊗ z1 == z1
    @test z1 ⊗ z1 == z0
    @test (@constinferred z0 ⊗ z0) == z0

    q = TrivialSector()
    @test (@constinferred q ⊗ q) == q
    @test (@constinferred q ⊗ z0) == z0
    @test (@constinferred z1 ⊗ q) == z1
    @test nsymbol(q, q, q) == 1

    # test different input number
    @test tensor_product(z0) == z0
    @test tensor_product(z0, z0, z0) == z0
    @test tensor_product(z0, z0, z0, z0) == z0

    @test (@constinferred quantum_dimension(gradedrange([z1 => 1]))) == 1
  end
  @testset "U(1) fusion rules" begin
    q1 = U1(1)
    q2 = U1(2)
    q3 = U1(3)

    @test q1 ⊗ q1 == U1(2)
    @test q1 ⊗ q2 == U1(3)
    @test q2 ⊗ q1 == U1(3)
    @test (@constinferred q1 ⊗ q2) == q3
    @test nsymbol(q1, q2, q3) == 1
    @test nsymbol(q1, q1, q3) == 0
  end

  @testset "O2 fusion rules" begin
    s0e = O2(0)
    s0o = O2(-1)
    s12 = O2(1//2)
    s1 = O2(1)

    q = TrivialSector()
    @test space_isequal((@constinferred s0e ⊗ q), gradedrange([s0e => 1]))
    @test space_isequal((@constinferred q ⊗ s0o), gradedrange([s0o => 1]))

    @test space_isequal((@constinferred s0e ⊗ s0e), gradedrange([s0e => 1]))
    @test space_isequal((@constinferred s0o ⊗ s0e), gradedrange([s0o => 1]))
    @test space_isequal((@constinferred s0o ⊗ s0e), gradedrange([s0o => 1]))
    @test space_isequal((@constinferred s0o ⊗ s0o), gradedrange([s0e => 1]))

    @test space_isequal((@constinferred s0e ⊗ s12), gradedrange([s12 => 1]))
    @test space_isequal((@constinferred s0o ⊗ s12), gradedrange([s12 => 1]))
    @test space_isequal((@constinferred s12 ⊗ s0e), gradedrange([s12 => 1]))
    @test space_isequal((@constinferred s12 ⊗ s0o), gradedrange([s12 => 1]))
    @test space_isequal((@constinferred s12 ⊗ s1), gradedrange([s12 => 1, O2(3//2) => 1]))
    @test space_isequal(
      (@constinferred s12 ⊗ s12), gradedrange([s0o => 1, s0e => 1, s1 => 1])
    )

    @test (@constinferred quantum_dimension(s0o ⊗ s1)) == 2
  end

  @testset "SU2 fusion rules" begin
    j1 = SU2(0)
    j2 = SU2(1//2)
    j3 = SU2(1)
    j4 = SU2(3//2)
    j5 = SU2(2)

    @test space_isequal(j1 ⊗ j2, gradedrange([j2 => 1]))
    @test space_isequal(j2 ⊗ j2, gradedrange([j1 => 1, j3 => 1]))
    @test space_isequal(j2 ⊗ j3, gradedrange([j2 => 1, j4 => 1]))
    @test space_isequal(j3 ⊗ j3, gradedrange([j1 => 1, j3 => 1, j5 => 1]))
    @test space_isequal((@constinferred j1 ⊗ j2), gradedrange([j2 => 1]))
    @test (@constinferred quantum_dimension(j1 ⊗ j2)) == 2

    @test tensor_product(j2) == j2
    @test space_isequal(tensor_product(j2, j1), gradedrange([j2 => 1]))
    @test space_isequal(tensor_product(j2, j1, j1), gradedrange([j2 => 1]))
  end
end

@testset "Gradedrange fusion rules" begin
  @testset "Trivial GradedUnitRange" begin
    g1 = gradedrange([U1(0) => 1])
    g2 = gradedrange([SU2(0) => 1])
    @test space_isequal(trivial(g1), g1)
    @test space_isequal(trivial(dual(g1)), g1)  # trivial returns nondual
    @test space_isequal(trivial(typeof(g2)), g2)
  end
  @testset "GradedUnitRange abelian tensor product" begin
    g1 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 2])
    g2 = gradedrange([U1(-2) => 2, U1(0) => 1, U1(1) => 2])

    @test space_isequal(flip(dual(g1)), gradedrange([U1(1) => 1, U1(0) => 1, U1(-1) => 2]))
    @test (@constinferred blocklengths(g1)) == [1, 1, 2]

    gt = gradedrange([
      U1(-3) => 2,
      U1(-2) => 2,
      U1(-1) => 4,
      U1(-1) => 1,
      U1(0) => 1,
      U1(1) => 2,
      U1(0) => 2,
      U1(1) => 2,
      U1(2) => 4,
    ])
    gf = gradedrange([
      U1(-3) => 2, U1(-2) => 2, U1(-1) => 5, U1(0) => 3, U1(1) => 4, U1(2) => 4
    ])
    @test space_isequal((@constinferred unmerged_tensor_product(g1, g2)), gt)
    @test space_isequal((@constinferred tensor_product(g1, g2)), gf)

    gtd1 = gradedrange([
      U1(-1) => 2,
      U1(-2) => 2,
      U1(-3) => 4,
      U1(1) => 1,
      U1(0) => 1,
      U1(-1) => 2,
      U1(2) => 2,
      U1(1) => 2,
      U1(0) => 4,
    ])
    gfd1 = gradedrange([
      U1(-3) => 4, U1(-2) => 2, U1(-1) => 4, U1(0) => 5, U1(1) => 3, U1(2) => 2
    ])
    @test space_isequal((@constinferred unmerged_tensor_product(dual(g1), g2)), gtd1)
    @test space_isequal((@constinferred tensor_product(dual(g1), g2)), gfd1)

    gtd2 = gradedrange([
      U1(1) => 2,
      U1(2) => 2,
      U1(3) => 4,
      U1(-1) => 1,
      U1(0) => 1,
      U1(1) => 2,
      U1(-2) => 2,
      U1(-1) => 2,
      U1(0) => 4,
    ])
    gfd2 = gradedrange([
      U1(-2) => 2, U1(-1) => 3, U1(0) => 5, U1(1) => 4, U1(2) => 2, U1(3) => 4
    ])
    @test space_isequal((@constinferred unmerged_tensor_product(g1, dual(g2))), gtd2)
    @test space_isequal((@constinferred tensor_product(g1, dual(g2))), gfd2)

    gtd = gradedrange([
      U1(3) => 2,
      U1(2) => 2,
      U1(1) => 4,
      U1(1) => 1,
      U1(0) => 1,
      U1(-1) => 2,
      U1(0) => 2,
      U1(-1) => 2,
      U1(-2) => 4,
    ])
    gfd = gradedrange([
      U1(-2) => 4, U1(-1) => 4, U1(0) => 3, U1(1) => 5, U1(2) => 2, U1(3) => 2
    ])
    @test space_isequal((@constinferred unmerged_tensor_product(dual(g1), dual(g2))), gtd)
    @test space_isequal((@constinferred tensor_product(dual(g1), dual(g2))), gfd)

    # test different (non-product) sectors cannot be fused
    @test_throws MethodError tensor_product(gradedrange([Z{2}(0) => 1]), g1)
    @test_throws MethodError tensor_product(gradedrange([Z{2}(0) => 1]), g2)
  end

  @testset "GradedUnitRange non-abelian fusion rules" begin
    g3 = gradedrange([SU2(0) => 1, SU2(1//2) => 2, SU2(1) => 1])
    g4 = gradedrange([SU2(1//2) => 1, SU2(1) => 2])
    g34 = gradedrange([
      SU2(1//2) => 1,
      SU2(0) => 2,
      SU2(1) => 2,
      SU2(1//2) => 1,
      SU2(3//2) => 1,
      SU2(1) => 2,
      SU2(1//2) => 4,
      SU2(3//2) => 4,
      SU2(0) => 2,
      SU2(1) => 2,
      SU2(2) => 2,
    ])

    @test space_isequal(unmerged_tensor_product(g3, g4), g34)

    @test space_isequal(dual(flip(g3)), g3)  # trivial for SU(2)
    @test space_isequal(
      (@constinferred tensor_product(g3, g4)),
      gradedrange([SU2(0) => 4, SU2(1//2) => 6, SU2(1) => 6, SU2(3//2) => 5, SU2(2) => 2]),
    )
    @test (@constinferred blocklengths(g3)) == [1, 4, 3]

    # test dual on non self-conjugate non-abelian representations
    s1 = SU{3}((0, 0))
    f3 = SU{3}((1, 0))
    c3 = SU{3}((1, 1))
    ad8 = SU{3}((2, 1))

    g5 = gradedrange([s1 => 1, f3 => 1])
    g6 = gradedrange([s1 => 1, c3 => 1])
    @test space_isequal(dual(flip(g5)), g6)
    @test space_isequal(
      tensor_product(g5, g6), gradedrange([s1 => 2, f3 => 1, c3 => 1, ad8 => 1])
    )
    @test space_isequal(
      tensor_product(dual(g5), g6),
      gradedrange([s1 => 1, f3 => 1, c3 => 2, SU{3}((2, 2)) => 1]),
    )
    @test space_isequal(
      tensor_product(g5, dual(g6)),
      gradedrange([s1 => 1, f3 => 2, c3 => 1, SU{3}((2, 0)) => 1]),
    )
    @test space_isequal(
      tensor_product(dual(g5), dual(g6)), gradedrange([s1 => 2, f3 => 1, c3 => 1, ad8 => 1])
    )

    @test nsymbol(ad8, ad8, ad8) == 2
  end

  @testset "Mixed GradedUnitRange - Sector fusion rules" begin
    g1 = gradedrange([U1(1) => 1, U1(2) => 2])
    g2 = gradedrange([U1(2) => 1, U1(3) => 2])
    @test space_isequal((@constinferred tensor_product(g1, U1(1))), g2)
    @test space_isequal((@constinferred tensor_product(U1(1), g1)), g2)

    g3 = gradedrange([SU2(0) => 1, SU2(1//2) => 2])
    g4 = gradedrange([SU2(0) => 2, SU2(1//2) => 1, SU2(1) => 2])
    @test space_isequal((@constinferred tensor_product(g3, SU2(1//2))), g4)
    @test space_isequal((@constinferred tensor_product(SU2(1//2), g3)), g4)

    # test different simple sectors cannot be fused
    @test_throws MethodError Z{2}(0) ⊗ U1(1)
    @test_throws MethodError SU2(1) ⊗ U1(1)
    @test_throws MethodError tensor_product(g1, SU2(1))
    @test_throws MethodError tensor_product(U1(1), g3)
  end
end
