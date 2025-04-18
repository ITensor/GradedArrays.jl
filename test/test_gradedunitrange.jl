using BlockArrays:
  Block,
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
  GradedArrays,
  GradedOneTo,
  GradedUnitRange,
  SectorUnitRange,
  blocklabels,
  dual,
  flip,
  gradedrange,
  isdual,
  nondual_sector,
  sector_type,
  sectorrange,
  space_isequal
using Test: @test, @test_broken, @testset

@testset "GradedUnitRanges basics" begin
  r0 = Base.OneTo(1)
  g = gradedrange(["x" => 2, "y" => 3])
  @test g isa GradedOneTo
  @test g isa GradedUnitRange
  @test sector_type(g) === String
  @test space_isequal(g, g)
  @test !space_isequal(r0, g)
  @test !space_isequal(g, r0)
  @test !space_isequal(g, 1:5)
  for x in iterate(g)
    @test x == 1
  end
  for x in iterate(g, 1)
    @test x == 2
  end
  for x in iterate(g, 2)
    @test x == 3
  end
  for x in iterate(g, 3)
    @test x == 4
  end
  for x in iterate(g, 4)
    @test x == 5
  end
  @test isnothing(iterate(g, 5))
  @test length(g) == 5
  @test step(g) == 1
  @test length(blocks(g)) == 2
  @test blocks(g)[1] == 1:2
  @test nondual_sector(blocks(g)[1]) == "x"
  @test blocks(g)[2] == 3:5
  @test nondual_sector(blocks(g)[2]) == "y"

  @test g[Block(2)] isa SectorUnitRange
  @test space_isequal(g[Block(2)], sectorrange("y", 3:5))

  @test g[4] == 4
  @test blocklengths(g) == [2, 3]
  @test blocklabels(g) == ["x", "y"]
  @test blockfirsts(g) == [1, 3]
  @test first(g) == 1
  @test blocklasts(g) == [2, 5]
  @test last(g) == 5
  @test blocklengths(only(axes(g))) == blocklengths(g)
  @test blocklabels(only(axes(g))) == blocklabels(g)
  @test !isdual(g)

  @test axes(Base.Slice(g)) isa Tuple{typeof(g)}
  @test AbstractUnitRange{Int}(g) == 1:5
  @test_broken b = combine_blockaxes(g, g)  # TODO
  #=
  @test b isa GradedOneTo
  @test b == 1:5
  @test space_isequal(b, g)
  =#

  # Slicing operations
  g = gradedrange(["x" => 2, "y" => 3])
  a = g[2:4]
  @test a isa BlockedUnitRange
  @test blockisequal(a, blockedrange([1, 1, 2])[Block.(2:3)])
  @test g[[2, 4]] == [2, 4]

  # Regression test for ambiguity errors.
  g = gradedrange(["x" => 2, "y" => 3])
  a = g[BlockSlice(Block(1), Base.OneTo(2))]
  @test length(a) == 2
  @test a == 1:2
  @test blocklength(a) == 1
  @test a isa SectorUnitRange
  @test space_isequal(a, sectorrange("x" => 2))
  @test space_isequal(g, g[:])

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

  g = gradedrange(["x" => 2, "y" => 3, "z" => 4])
  a = g[[Block(3)[2:3], Block(2)[2:3]]]  # drop labels
  @test a isa BlockVector
  @test length(a) == 4
  @test blocklength(a) == 2
  # TODO: `BlockArrays` doesn't define `blocklengths`
  # for `BlockVector`, should it?
  @test_broken blocklengths(a) == [2, 2]
  @test a[Block(1)] == 7:8
  @test a[Block(2)] == 4:5
  ax = only(axes(a))
  @test ax == 1:length(a)
  @test length(ax) == length(a)
  # TODO: Change to:
  # @test blocklengths(ax) == blocklengths(a)
  # once `blocklengths(::BlockVector)` is defined.
  @test blocklengths(ax) == [2, 2]

  g = gradedrange(["x" => 2, "y" => 3])
  I = mortar([Block(1)[1:1]])
  a = g[I]
  @test length(a) == 1

  gd = gradedrange(["x" => 2, "y" => 3], dual=true)
  @test isdual(gd)
  @test gd[Block(2)] isa SectorUnitRange
  @test space_isequal(gd[Block(2)], sectorrange("y", 3:5, true))
  @test blocklabels(g) == blocklabels(gd)  # string is self-dual
  @test !space_isequal(gd, g)
  @test space_isequal(gd, dual(g))
  @test space_isequal(gd, flip(g))
  @test space_isequal(flip(gd), g)
  @test space_isequal(dual(gd), g)

  # label length > 1
  g = gradedrange(["x" => 2, "yy" => 3])
  @test length(g) == 8
  @test blocklengths(g) == [2, 6]
  @test space_isequal(g[Block(2)], sectorrange("yy", 3:8))
end
