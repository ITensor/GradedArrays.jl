using BlockArrays: blocklength, blocklengths
using GradedArrays:
  GradedArrays,
  GradedOneTo,
  NotAbelianStyle,
  U1,
  blocklabels,
  dual,
  unmerged_tensor_product,
  flip,
  gradedrange,
  space_isequal,
  isdual
using TensorProducts: OneToOne, tensor_product
using Test: @test, @testset

GradedArrays.SymmetryStyle(::Type{<:String}) = NotAbelianStyle()
GradedArrays.tensor_product(s1::String, s2::String) = gradedrange([s1 * s2 => 1])

@testset "unmerged_tensor_product" begin
  @test unmerged_tensor_product() isa OneToOne
  @test unmerged_tensor_product(OneToOne(), OneToOne()) isa OneToOne

  a = gradedrange(["x" => 2, "y" => 3])
  @test space_isequal(unmerged_tensor_product(a), a)

  b = unmerged_tensor_product(a, a)
  @test b isa GradedOneTo
  @test length(b) == 50
  @test blocklength(b) == 4
  @test blocklengths(b) == [8, 12, 12, 18]
  @test space_isequal(b, gradedrange(["xx" => 4, "yx" => 6, "xy" => 6, "yy" => 9]))

  c = unmerged_tensor_product(a, a, a)
  @test c isa GradedOneTo
  @test length(c) == 375
  @test blocklength(c) == 8
  @test blocklabels(c) == ["xxx", "yxx", "xyx", "yyx", "xxy", "yxy", "xyy", "yyy"]

  a = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])
  @test space_isequal(
    unmerged_tensor_product(a, a),
    gradedrange([
      U1(2) => 1,
      U1(3) => 3,
      U1(2) => 1,
      U1(3) => 3,
      U1(4) => 9,
      U1(3) => 3,
      U1(2) => 1,
      U1(3) => 3,
      U1(2) => 1,
    ]),
  )
  @test space_isequal(unmerged_tensor_product(a), a)
  @test space_isequal(unmerged_tensor_product(a, OneToOne()), a)
  @test space_isequal(unmerged_tensor_product(OneToOne(), a), a)
  @test space_isequal(tensor_product(a), gradedrange([U1(1) => 2, U1(2) => 3]))

  @test space_isequal(
    tensor_product(a, a), gradedrange([U1(2) => 4, U1(3) => 12, U1(4) => 9])
  )
  @test space_isequal(tensor_product(a, OneToOne()), gradedrange([U1(1) => 2, U1(2) => 3]))
  @test space_isequal(tensor_product(OneToOne(), a), gradedrange([U1(1) => 2, U1(2) => 3]))

  d = tensor_product(a, a, a)
  @test space_isequal(d, gradedrange([U1(3) => 8, U1(4) => 36, U1(5) => 54, U1(6) => 27]))
end

@testset "dual and tensor_product" begin
  a = gradedrange([U1(1) => 1, U1(2) => 3, U1(1) => 1])
  ad = dual(a)

  b = unmerged_tensor_product(ad)
  @test isdual(b)
  @test space_isequal(b, ad)
  @test space_isequal(unmerged_tensor_product(ad, OneToOne()), ad)
  @test space_isequal(unmerged_tensor_product(OneToOne(), ad), ad)

  b = tensor_product(ad)
  @test b isa GradedOneTo
  @test !isdual(b)
  @test space_isequal(b, gradedrange([U1(-2) => 3, U1(-1) => 2]))

  c = tensor_product(ad, ad)
  @test c isa GradedOneTo
  @test !isdual(c)
  @test space_isequal(c, gradedrange([U1(-4) => 9, U1(-3) => 12, U1(-2) => 4]))

  d = tensor_product(ad, a)
  @test !isdual(d)
  @test space_isequal(d, gradedrange([U1(-1) => 6, U1(0) => 13, U1(1) => 6]))

  e = tensor_product(a, ad)
  @test !isdual(d)
  @test space_isequal(e, d)
end
