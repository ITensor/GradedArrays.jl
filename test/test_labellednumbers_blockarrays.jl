using GradedArrays.LabelledNumbers: LabelledUnitRange, label, label_type, labelled
using BlockArrays: Block, blockaxes, blocklength, blocklengths

@testset "LabelledNumbersBlockArraysExt" begin
  x = labelled(1:2, "x")
  @test blockaxes(x) == (Block.(1:1),)
  @test blocklength(x) == 1
  @test blocklengths(x) == [2]
  a = x[Block(1)]
  @test a == 1:2
  @test a isa LabelledUnitRange
  @test label(a) == "x"
  @test label_type(a) === String
  @test label_type(typeof(a)) === String
end
