using BlockArrays:
  Block,
  BlockBoundsError,
  BlockRange,
  blockaxes,
  blocklasts,
  blocklength,
  blocklengths,
  blockisequal,
  blocks,
  findblock
using GradedArrays:
  U1,
  SU,
  SectorOneTo,
  SectorUnitRange,
  SectorVector,
  dual,
  flip,
  isdual,
  quantum_dimension,
  sector,
  sector_multiplicities,
  sector_multiplicity,
  sector_type,
  sectorrange,
  sectors,
  space_isequal,
  ungrade
using Test: @test, @test_throws, @testset
using TestExtras: @constinferred

@testset "SectorUnitRange" begin
  sr = sectorrange(SU((1, 0)), 2)
  @test sr isa SectorUnitRange

  # accessors
  @test sector(sr) == SU((1, 0))
  @test ungrade(sr) isa Base.OneTo
  @test ungrade(sr) == 1:6
  @test !isdual(sr)

  # Base interface
  @test first(sr) == 1
  @test last(sr) == 6
  @test length(sr) == 6
  @test firstindex(sr) == 1
  @test lastindex(sr) == 6
  @test eltype(sr) === Int
  @test step(sr) == 1
  @test eachindex(sr) == Base.oneto(6)
  @test only(axes(sr)) isa SectorOneTo
  @test sector(only(axes(sr))) == sector(sr)
  @test only(axes(sr)) == 1:6
  @test iterate(sr) == (1, 1)
  for i in 1:5
    @test iterate(sr, i) == (i + 1, i + 1)
  end
  @test isnothing(iterate(sr, 6))

  # Base.Slice
  @test axes(Base.Slice(sr)) === (sr,)
  @test Base.axes1(Base.Slice(sr)) === sr
  @test Base.unsafe_indices(Base.Slice(sr)) === (sr,)

  @test sr == 1:6
  @test sr == sr
  @test space_isequal(sr, sr)

  sr = sectorrange(SU((1, 0)) => 2)
  @test sr isa SectorUnitRange
  @test sector(sr) == SU((1, 0))
  @test ungrade(sr) isa Base.OneTo
  @test ungrade(sr) == 1:6
  @test !isdual(sr)

  sr = sectorrange(SU((1, 0)) => 2, true)
  @test sr isa SectorUnitRange
  @test sector(sr) == SU((1, 0))
  @test ungrade(sr) isa Base.OneTo
  @test ungrade(sr) == 1:6
  @test isdual(sr)

  sr = sectorrange(SU((1, 0)), 4:10, true)
  @test sr isa SectorUnitRange
  @test sector(sr) == SU((1, 0))
  @test ungrade(sr) isa UnitRange
  @test ungrade(sr) == 4:10
  @test isdual(sr)

  sr = sectorrange(SU((1, 0)), 2)
  @test !space_isequal(sr, sectorrange(SU((1, 1)), 2))
  @test !space_isequal(sr, sectorrange(SU((1, 0)), 2:7))
  @test !space_isequal(sr, sectorrange(SU((1, 1)), 2, true))
  @test !space_isequal(sr, sectorrange(SU((1, 0)), 2, true))

  sr2 = copy(sr)
  @test sr2 isa SectorUnitRange
  @test space_isequal(sr, sr2)
  sr3 = deepcopy(sr)
  @test sr3 isa SectorUnitRange
  @test space_isequal(sr, sr3)

  # BlockArrays interface
  @test blockaxes(sr) isa Tuple{BlockRange{1,<:Tuple{Base.OneTo}}}
  @test space_isequal(sr[Block(1)], sr)
  @test only(blocklasts(sr)) == 6
  @test findblock(sr, 2) == Block(1)

  @test blocklength(sr) == 1
  @test blocklengths(sr) == [6]
  @test only(blocks(sr)) == 1:6
  @test blockisequal(sr, sr)

  # GradedUnitRanges interface
  @test sector_type(sr) === SU{3,2}
  @test sector_type(typeof(sr)) === SU{3,2}
  @test sectors(sr) == [SU((1, 0))]
  @test sector_multiplicity(sr) == 2
  @test sector_multiplicities(sr) == [2]
  @test quantum_dimension(sr) == 6

  srd = dual(sr)
  @test sector(srd) == SU((1, 0))
  @test space_isequal(srd, sectorrange(SU((1, 0)), 2, true))
  @test sectors(srd) == [SU((1, 0))]

  srf = flip(sr)
  @test sector(srf) == SU((1, 1))
  @test space_isequal(srf, sectorrange(SU((1, 1)), 2, true))

  # getindex
  @test_throws BoundsError sr[0]
  @test_throws BoundsError sr[7]
  @test (@constinferred getindex(sr, 1)) isa Int64
  for i in 1:6
    @test sr[i] == i
  end
  @test sr[2:3] == 2:3
  @test (@constinferred getindex(sr, 2:3)) isa UnitRange
  @test sr[Block(1)] === sr
  @test_throws BlockBoundsError sr[Block(2)]

  sr2 = (@constinferred getindex(sr, (:, 2)))
  @test sr2 isa SectorUnitRange
  @test space_isequal(sr2, sectorrange(SU((1, 0)), 4:6))
  sr3 = (@constinferred getindex(sr, (:, 1:2)))
  @test sr3 isa SectorUnitRange
  @test space_isequal(sr3, sectorrange(SU((1, 0)), 1:6))

  # Abelian slicing
  srab = sectorrange(U1(1), 3)
  @test (@constinferred getindex(srab, 2:2)) isa SectorUnitRange
  @test space_isequal(srab[2:2], sectorrange(U1(1), 2:2))
  @test space_isequal(dual(srab)[2:2], sectorrange(U1(1), 2:2, true))
  @test srab[[1, 3]] isa SectorVector{Int}
  @test sector(srab[[1, 3]]) == sector(srab)
  @test ungrade(srab[[1, 3]]) == [1, 3]
  @test length(srab[[1, 3]]) == 2
  @test space_isequal(only(axes(srab[[1, 3]])), sectorrange(U1(1), 2))

  # Slice sector range with sector range
  sr1 = sectorrange(U1(1), 4)
  sr2 = sectorrange(U1(1), 3)
  @test sr1[sr2] === sr2
end
