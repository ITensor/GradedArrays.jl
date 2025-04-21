using BlockArrays: BlockedOneTo, blockisequal

using GradedArrays: NoLabel, blocklabels, dag, dual, flip, isdual, space_isequal
using Test: @test, @testset
using TensorProducts: OneToOne

@testset "GradedUnitRange interface for AbstractUnitRange" begin
  a0 = OneToOne()
  @test !isdual(a0)
  @test dual(a0) isa OneToOne
  @test space_isequal(a0, a0)
  @test space_isequal(a0, dual(a0))
  @test only(blocklabels(a0)) == NoLabel()

  a = 1:3
  ad = dual(a)
  af = flip(a)
  @test !isdual(a)
  @test !isdual(ad)
  @test !isdual(dag(a))
  @test !isdual(af)
  @test ad isa UnitRange
  @test af isa UnitRange
  @test space_isequal(ad, a)
  @test space_isequal(af, a)
  @test only(blocklabels(a)) == NoLabel()

  a = blockedrange([2, 3])
  ad = dual(a)
  af = flip(a)
  @test !isdual(a)
  @test !isdual(ad)
  @test ad isa BlockedOneTo
  @test af isa BlockedOneTo
  @test blockisequal(ad, a)
  @test blockisequal(af, a)
  @test blocklabels(a) == [NoLabel(), NoLabel()]
end
