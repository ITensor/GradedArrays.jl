using BlockArrays:
  Block,
  BlockSlice,
  BlockedOneTo,
  BlockVector,
  BlockedUnitRange,
  blockedrange,
  blockfirsts,
  blockisequal,
  blocklasts,
  blocklength,
  blocklengths,
  blocks,
  combine_blockaxes,
  findblock,
  findblockindex,
  mortar
using GradedArrays:
  GradedOneTo,
  GradedUnitRange,
  SectorOneTo,
  SectorUnitRange,
  SU,
  checkspaces,
  checkspaces_dual,
  dual,
  findfirstblock,
  findfirstblock_sector,
  flip,
  gradedrange,
  isdual,
  mortar_axis,
  sector_multiplicities,
  sector_type,
  sectors,
  sectorrange,
  space_isequal
using Test: @test, @test_throws, @testset

@testset "GradedUnitRanges basics" begin
  r0 = Base.OneTo(1)
  b0 = blockedrange([2, 3, 2])
  g1 = gradedrange(["x" => 2, "y" => 3, "z" => 2])
  @test g1 isa GradedOneTo
  @test !isdual(g1)

  # Base.Slice
  @test axes(Base.Slice(g1)) === (g1,)
  @test Base.axes1(Base.Slice(g1)) === g1
  @test Base.unsafe_indices(Base.Slice(g1)) === (g1,)

  g2 = gradedrange(1, ["x" => 2, "y" => 3, "z" => 2])
  @test !(g2 isa GradedOneTo)
  @test !isdual(g2)

  g1d = gradedrange(["x" => 2, "y" => 3, "z" => 2]; isdual=true)
  @test g1d isa GradedOneTo
  @test isdual(g1d)

  # Base.Slice
  @test axes(Base.Slice(g1d)) === (g1d,)
  @test Base.axes1(Base.Slice(g1d)) === g1d
  @test Base.unsafe_indices(Base.Slice(g1d)) === (g1d,)

  for g in (g1, g2, g1d)
    @test g isa GradedUnitRange
    @test sector_type(g) === String
    @test blockisequal(g, b0)
    @test space_isequal(g, g)
    @test space_isequal(copy(g), g)
    @test !space_isequal(r0, g)
    @test !space_isequal(g, r0)
    @test !space_isequal(g, b0)
    @test !space_isequal(g, 1:7)
    @test !space_isequal(g, dual(g))
    @test space_isequal(combine_blockaxes(g, g), g)
    @test g == 1:7
    for x in iterate(g)
      @test x == 1
    end
    for i in 1:6
      for x in iterate(g, i)
        @test x == i + 1
      end
    end
    @test isnothing(iterate(g, 7))
    @test length(g) == 7
    @test step(g) == 1
    @test blocklength(g) == 3
    @test length(blocks(g)) == 3

    @test g[Block(1)] isa SectorUnitRange
    @test !(g[Block(1)] isa SectorOneTo)
    @test space_isequal(g[Block(1)], sectorrange("x", 1:2, isdual(g)))
    @test space_isequal(g[Block(2)], sectorrange("y", 3:5, isdual(g)))
    @test space_isequal(g[Block(3)], sectorrange("z", 6:7, isdual(g)))

    @test g[4] == 4
    @test blocklengths(g) == [2, 3, 2]
    @test sectors(g) == ["x", "y", "z"]
    @test sector_multiplicities(g) == [2, 3, 2]
    @test blockfirsts(g) == [1, 3, 6]
    @test first(g) == 1
    @test blocklasts(g) == [2, 5, 7]
    @test last(g) == 7
    @test blocklengths(only(axes(g))) == blocklengths(g)
    @test sectors(only(axes(g))) == sectors(g)
    @test findblock(g, 2) == Block(1)
    @test findblock(g, 3) == Block(2)
    @test findblockindex(g, 3) == Block(2)[1]
    @test findfirstblock_sector(g, "y") == Block(2)
    @test isnothing(findfirstblock_sector(g, "a"))

    @test axes(Base.Slice(g)) isa Tuple{typeof(g)}
    @test AbstractUnitRange{Int}(g) == 1:7
    ge = eachindex(g)
    @test ge isa GradedOneTo
    @test space_isequal(g, ge)

    @test g[Block(1)[1]] == 1
    @test g[Block(2)[1]] == 3

    # Abelian slicing operations
    a = g[2:4]
    @test a isa GradedUnitRange
    @test !(a isa GradedOneTo)
    @test sectors(a) == ["x", "y"]
    @test blocklength(a) == 2
    @test space_isequal(first(blocks(a)), sectorrange("x", 2:2, isdual(g)))
    @test space_isequal(last(blocks(a)), sectorrange("y", 3:4, isdual(g)))
    @test g[[2, 4]] == [2, 4]

    # Regression test for ambiguity errors.
    a = g[BlockSlice(Block(1), Base.OneTo(2))]
    @test length(a) == 2
    @test a == 1:2
    @test blocklength(a) == 1
    @test a isa SectorUnitRange
    @test space_isequal(a, sectorrange("x" => 2, isdual(g)))
    @test space_isequal(g, g[:])
    @test typeof(g[:]) === typeof(g)

    a = g[Block(2)[2:3]]
    @test a isa SectorUnitRange
    @test space_isequal(a, sectorrange("y", 4:5, isdual(g)))

    a = g[Block(2):Block(3)]
    @test a isa GradedUnitRange
    @test !(a isa GradedOneTo)
    @test blocklength(a) == 2
    @test space_isequal(a[Block(1)], sectorrange("y", 3:5, isdual(g)))
    @test space_isequal(a[Block(2)], sectorrange("z", 6:7, isdual(g)))
    ax = only(axes(a))
    @test ax isa GradedOneTo
    @test space_isequal(ax, gradedrange(["y" => 3, "z" => 2]; isdual=isdual(g)))
    @test ax == 1:length(a)
    @test length(ax) == length(a)
    @test blocklengths(ax) == blocklengths(a)
    @test sectors(ax) == sectors(a)

    a = g[Block.(Base.oneto(2))]
    @test (a isa GradedOneTo) == (g isa GradedOneTo)
    @test space_isequal(a, g[Block(1):Block(2)])

    a = g[[Block(3), Block(2)]]
    @test a isa BlockVector
    @test length(a) == 5
    @test blocklength(a) == 2
    @test length.(blocks(a)) == [2, 3]
    @test sectors(a) == ["z", "y"]
    @test a[Block(1)] == 6:7
    @test a[Block(2)] == 3:5
    ax = only(axes(a))
    @test ax isa GradedOneTo
    @test space_isequal(ax, gradedrange(["z" => 2, "y" => 3]; isdual=isdual(g)))

    # slice with one multiplicity
    # TODO use dedicated Kroneckrange
    sr = g[(:, 1)]
    @test sr isa SectorUnitRange
    @test !(sr isa SectorOneTo)
    @test space_isequal(sr, sectorrange("x", 1:1, isdual(g)))
    sr = g[(:, 3)]
    @test sr isa SectorUnitRange
    @test !(sr isa SectorOneTo)
    @test space_isequal(sr, sectorrange("y", 3:3, isdual(g)))

    # slice along multiplicities
    # TODO use dedicated Kroneckrange
    for i in 1:length(g), j in i:length(g)
      @test g[(:, i:j)] isa GradedUnitRange
      @test space_isequal(g[(:, i:j)], g[i:j])
    end

    a = g[[Block(3)[1:1], Block(2)[2:3]]]
    @test a isa BlockVector
    @test length(a) == 3
    @test blocklength(a) == 2
    @test length.(blocks(a)) == [1, 2]
    @test a[Block(1)] == 6:6
    @test a[Block(2)] == 4:5
    ax = only(axes(a))
    @test ax isa GradedOneTo
    @test space_isequal(ax, gradedrange(["z" => 1, "y" => 2]; isdual=isdual(g)))

    I = mortar([Block(1)[1:1], Block(1)[1:2], Block(2)[1:2]])
    a = g[I]
    @test a isa BlockVector
    @test length(a) == 5
    @test blocklength(a) == 3
    ax = only(axes(a))
    @test ax isa GradedOneTo
    @test space_isequal(ax, gradedrange(["x" => 1, "x" => 2, "y" => 2]; isdual=isdual(g)))

    v = mortar([[Block(2), Block(2)], [Block(1)]])
    a = g[v]
    @test a isa BlockVector
    @test only(axes(a)) isa GradedOneTo
    @test space_isequal(only(axes(a)), gradedrange(["y" => 6, "x" => 2]; isdual=isdual(g)))
  end

  @test checkspaces(Bool, (g1, g1d), (g1, g1d))

  @test space_isequal(g1d, dual(g1))
  @test space_isequal(dual(g1d), g1)
  @test checkspaces((g1, g1d), (g1, g1d))
  @test checkspaces_dual(Bool, (g1, g1d), (g1d, g1))
  @test checkspaces_dual((g1, g1d), (g1d, g1))
  @test_throws ArgumentError checkspaces((g1, g1), (g1, g1d))
  @test_throws ArgumentError checkspaces_dual((g1, g1), (g1, g1d))

  for a in (
    combine_blockaxes(g1, b0),
    combine_blockaxes(g1d, b0),
    combine_blockaxes(b0, g1),
    combine_blockaxes(b0, g1d),
  )
    @test a isa BlockedOneTo
    @test blockisequal(a, b0)
  end
  @test_throws ArgumentError combine_blockaxes(g1, g1d)

  a = combine_blockaxes(g2, b0)
  @test a isa BlockedUnitRange
  @test blockisequal(a, b0)

  a = combine_blockaxes(g2, g1)
  @test a isa GradedUnitRange
  @test !(a isa GradedOneTo)
  @test space_isequal(a, g2)

  g3 = gradedrange(["x" => 2, "a" => 1, "y" => 2, "z" => 2])
  @test_throws ArgumentError combine_blockaxes(g1, g3)

  g3 = gradedrange(["x" => 1, "x" => 1, "y" => 2, "z" => 2])
  g4 = gradedrange(["x" => 2, "y" => 1, "y" => 1, "z" => 2])
  a = combine_blockaxes(g3, g4)
  @test space_isequal(a, gradedrange(["x" => 1, "x" => 1, "y" => 1, "y" => 1, "z" => 2]))

  sr1 = sectorrange("x", 2)
  sr2 = sectorrange("y", 3)
  @test space_isequal(g1[Block(1):Block(2)], mortar_axis([sr1, sr2]))
  @test_throws ArgumentError mortar_axis([sr1, dual(sr2)])
end

@testset "Non abelian axis" begin
  b0 = blockedrange([2, 6])
  g = gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 2])

  @test g isa GradedOneTo
  @test length(g) == 8
  @test !isdual(g)
  @test blockisequal(g, b0)
  @test sectors(g) == [SU((0, 0)), SU((1, 0))]
  @test sector_multiplicities(g) == [2, 2]
  @test space_isequal(g[Block(1)], sectorrange(SU((0, 0)), 2))
  @test space_isequal(g[Block(2)], sectorrange(SU((1, 0)), 3:8))

  @test sector_type(g) === SU{3,2}
  @test space_isequal(g, g)
  @test g == 1:8
  @test space_isequal(dual(g), gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 2]; isdual=true))
  @test !space_isequal(dual(g), g)
  @test space_isequal(flip(g), gradedrange([SU((0, 0)) => 2, SU((1, 1)) => 2]; isdual=true))

  @test iterate(g) == (1, 1)
  for i in 1:7
    @test iterate(g, i) == (i + 1, i + 1)
  end
  @test isnothing(iterate(g, 8))
  @test step(g) == 1

  @test g[4] == 4
  @test g[Block(1)[1]] == 1
  @test g[Block(2)[1]] == 3
  @test findfirstblock(g, SU((1, 0))) == Block(2)
  @test isnothing(findfirstblock(g, SU((2, 0))))

  # Non-abelian slicing operations
  a = g[2:4]
  @test a isa BlockedUnitRange
  @test blockisequal(a, blockedrange([1, 1, 2])[Block(2):Block(3)])

  # Regression test for ambiguity errors.
  a = g[BlockSlice(Block(1), Base.OneTo(2))]
  @test a isa SectorUnitRange
  @test space_isequal(a, sectorrange(SU((0, 0)) => 2))
  @test space_isequal(g, g[:])
  @test typeof(g[:]) === typeof(g)

  a = g[Block(2)[2:3]]
  @test a isa UnitRange
  @test a == 4:5

  a = g[Block(2):Block(2)]
  @test a isa GradedUnitRange
  @test !(a isa GradedOneTo)
  @test space_isequal(only(blocks(a)), sectorrange(SU((1, 0)), 3:8))
  ax = only(axes(a))
  @test ax isa GradedOneTo
  @test space_isequal(ax, gradedrange([SU((1, 0)) => 2]))

  a = g[Block.(Base.oneto(2))]
  @test a isa GradedOneTo
  @test space_isequal(a, g)

  a = g[[Block(2), Block(1)]]
  @test a isa BlockVector
  @test length(a) == 8
  @test blocklength(a) == 2
  @test sectors(a) == [SU((1, 0)), SU((0, 0))]
  @test length.(blocks(g)) == [2, 6]

  @test space_isequal(a[Block(1)], sectorrange(SU((1, 0)), 3:8))
  @test space_isequal(a[Block(2)], sectorrange(SU((0, 0)), 2))
  ax = only(axes(a))
  @test ax isa GradedOneTo
  @test space_isequal(ax, gradedrange([SU((1, 0)) => 2, SU((0, 0)) => 2]))

  # slice with one multiplicity
  # TODO use dedicated Kroneckrange
  sr = g[(:, 1)]
  @test sr isa SectorUnitRange
  @test !(sr isa SectorOneTo)
  @test space_isequal(sr, sectorrange(SU((0, 0)), 1:1))
  sr = g[(:, 3)]
  @test sr isa SectorUnitRange
  @test !(sr isa SectorOneTo)
  @test space_isequal(sr, sectorrange(SU((1, 0)), 3:5))

  g3 = gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 2, SU((2, 1)) => 2]; isdual=true)
  @test g3[(:, 1:1)] isa GradedUnitRange
  @test space_isequal(g3[(:, 1:1)], gradedrange([SU((0, 0)) => 1]; isdual=true))
  @test space_isequal(g3[(:, 1:2)], gradedrange([SU((0, 0)) => 2]; isdual=true))
  @test space_isequal(
    g3[(:, 1:3)], gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 1]; isdual=true)
  )
  @test space_isequal(
    g3[(:, 1:4)], gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 2]; isdual=true)
  )
  @test space_isequal(
    g3[(:, 1:5)],
    gradedrange([SU((0, 0)) => 2, SU((1, 0)) => 2, SU((2, 1)) => 1]; isdual=true),
  )
  @test space_isequal(
    g3[(:, 2:5)],
    gradedrange(2, [SU((0, 0)) => 1, SU((1, 0)) => 2, SU((2, 1)) => 1]; isdual=true),
  )
  @test space_isequal(
    g3[(:, 3:5)], gradedrange(3, [SU((1, 0)) => 2, SU((2, 1)) => 1]; isdual=true)
  )
  @test space_isequal(
    g3[(:, 4:5)], gradedrange(6, [SU((1, 0)) => 1, SU((2, 1)) => 1]; isdual=true)
  )

  a = g[[Block(2)[1:3], Block(1)[2:2]]]
  @test a isa BlockVector
  @test length(a) == 4
  @test blocklength(a) == 2
  @test a[Block(1)] == 3:5
  @test a[Block(2)] == 2:2
  ax = only(axes(a))
  @test ax isa BlockedOneTo
  @test blockisequal(ax, blockedrange([3, 1]))

  I = mortar([Block(1)[1:1]])
  a = g[I]
  @test a isa BlockVector
  @test length(a) == 1
  @test blocklength(a) == 1
  ax = only(axes(a))
  @test ax isa BlockedOneTo
  @test blockisequal(ax, blockedrange([1]))
end
