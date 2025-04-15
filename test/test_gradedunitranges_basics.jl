using BlockArrays:
  Block,
  BlockRange,
  BlockSlice,
  BlockVector,
  blockedrange,
  blockfirsts,
  blocklasts,
  blocklength,
  blocklengths,
  blocks,
  combine_blockaxes,
  mortar
using GradedArrays:
  GradedOneTo,
  GradedUnitRange,
  SectorUnitRange,
  blocklabels,
  gradedrange,
  nondual_sector,
  sector_type,
  sectorunitrange,
  space_isequal
using Test: @test, @test_broken, @testset

@testset "GradedUnitRanges basics" begin
  a0 = Base.OneTo(1)
  a = gradedrange(["x" => 2, "y" => 3])
  @test a isa GradedOneTo
  @test a isa GradedUnitRange
  @test sector_type(a) === String
  @test space_isequal(a, a)
  @test !space_isequal(a0, a)
  @test !space_isequal(a, a0)
  @test !space_isequal(a, 1:5)
  for x in iterate(a)
    @test x == 1
  end
  for x in iterate(a, 1)
    @test x == 2
  end
  for x in iterate(a, 2)
    @test x == 3
  end
  for x in iterate(a, 3)
    @test x == 4
  end
  for x in iterate(a, 4)
    @test x == 5
  end
  @test isnothing(iterate(a, 5))
  @test length(a) == 5
  @test step(a) == 1
  @test length(blocks(a)) == 2
  @test blocks(a)[1] == 1:2
  @test nondual_sector(blocks(a)[1]) == "x"
  @test blocks(a)[2] == 3:5
  @test nondual_sector(blocks(a)[2]) == "y"

  @test a[Block(2)] isa SectorUnitRange
  @test space_isequal(a[Block(2)], sectorunitrange("y", 3:5))

  @test a[4] == 4
  @test blocklengths(a) == [2, 3]
  @test blocklabels(a) == ["x", "y"]
  @test blockfirsts(a) == [1, 3]
  @test first(a) == 1
  @test blocklasts(a) == [2, 5]
  @test last(a) == 5
  @test blocklengths(only(axes(a))) == blocklengths(a)
  @test blocklabels(only(axes(a))) == blocklabels(a)

  @test axes(Base.Slice(a)) isa Tuple{typeof(a)}
  @test AbstractUnitRange{Int}(a) == 1:5
  b = combine_blockaxes(a, a)  # TODO
  @test b isa GradedOneTo
  @test b == 1:5
  @test space_isequal(b, a)

  # Slicing operations
  g = gradedrange(["x" => 2, "y" => 3])
  a = g[2:4]
  @test a isa BlockedUnitRange
  @test blockisequal(a, blockedrange([1, 1, 2])[Block.(2:3)])
  @test g[[2, 4]] == [2, 4]

  # Regression test for ambiguity error.
  g = gradedrange(["x" => 2, "y" => 3])
  a = g[BlockSlice(Block(1), Base.OneTo(2))]
  @test length(a) == 2
  @test a == 1:2
  @test blocklength(a) == 1
  @test a isa SectorUnitRange
  @test space_isequal(a, sectorunitrange("x" => 2))

  g = gradedrange(["x" => 2, "y" => 3])
  a = g[3:4]
  @test a isa BlockedUnitRange
  @test blockisequal(a, blockedrange([2, 2])[Block.(2:2)])

  g = gradedrange(["x" => 2, "y" => 3])
  a = g[Block(2)[2:3]]
  @test a isa UnitRange
  @test a == 4:5

  g = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = g[Block(2):Block(3)]
  @test a isa GradedUnitRange
  @test length(a) == 7
  @test blocklength(a) == 2
  @test blocklengths(a) == [3, 4]
  @test blocklabels(a) == ["y", "z"]
  @test a[Block(1)] == 3:5
  @test a[Block(2)] == 6:9
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  @test blocklengths(ax) == blocklengths(a)
  @test blocklabels(ax) == blocklabels(a)

  g = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = g[[Block(3), Block(2)]]
  @test a isa BlockVector
  @test length(a) == 7
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # `blocklengths(::BlockVector)`, unbrake this test
  # once it does.
  @test_broken blocklengths(a) == [4, 3]
  @test blocklabels(a) == ["z", "y"]
  @test a[Block(1)] == 6:9
  @test a[Block(2)] == 3:5
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  # TODO: Change to:
  # @test blocklengths(ax) == blocklengths(a)
  # once `blocklengths(::BlockVector)` is defined.
  @test blocklengths(ax) == [4, 3]
  @test blocklabels(ax) == blocklabels(a)

  x = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = x[[Block(3)[2:3], Block(2)[2:3]]]
  @test a isa BlockVector
  @test length(a) == 4
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # for `BlockVector`, should it?
  @test_broken blocklengths(a) == [2, 2]
  @test blocklabels(a) == ["z", "y"]
  @test a[Block(1)] == 7:8
  @test a[Block(2)] == 4:5
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  # TODO: Change to:
  # @test blocklengths(ax) == blocklengths(a)
  # once `blocklengths(::BlockVector)` is defined.
  @test blocklengths(ax) == [2, 2]
  @test blocklabels(ax) == blocklabels(a)

  g = gradedrange(["x" => 2, "y" => 3])
  I = mortar([Block(1)[1:1]])
  a = g[I]
  @test length(a) == 1
  @test label(first(a)) == "x"

  g = gradedrange(["x" => 2, "y" => 3])[1:5]
  I = mortar([Block(1)[1:1]])
  a = g[I]
  @test length(a) == 1
  @test label(first(a)) == "x"
end
