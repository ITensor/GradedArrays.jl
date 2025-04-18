using Test: @test, @test_throws, @testset

using BlockArrays: Block, blocklength, blocklengths, blockisequal, blocks

using GradedArrays:
  AbstractSector,
  SU,
  SectorUnitRange,
  blocklabels,
  dual,
  flip,
  full_range,
  isdual,
  nondual_sector,
  quantum_dimension,
  sector_multiplicities,
  sector_multiplicity,
  sector_type,
  sectorrange,
  space_isequal

@testset "SectorUnitRange" begin
  sr = sectorrange(SU((1, 0)), 2)
  @test sr isa SectorUnitRange

  # accessors
  @test nondual_sector(sr) == SU((1, 0))
  @test full_range(sr) isa Base.OneTo
  @test full_range(sr) == 1:6
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
  @test only(axes(sr)) isa Base.OneTo
  @test only(axes(sr)) == 1:6

  @test sr == 1:6
  @test sr == sr
  @test space_isequal(sr, sr)

  sr = sectorrange(SU((1, 0)) => 2)
  @test sr isa SectorUnitRange
  @test nondual_sector(sr) == SU((1, 0))
  @test full_range(sr) isa Base.OneTo
  @test full_range(sr) == 1:6
  @test !isdual(sr)

  sr = sectorrange(SU((1, 0)) => 2, true)
  @test sr isa SectorUnitRange
  @test nondual_sector(sr) == SU((1, 0))
  @test full_range(sr) isa Base.OneTo
  @test full_range(sr) == 1:6
  @test isdual(sr)

  sr = sectorrange(SU((1, 0)), 4:10, true)
  @test sr isa SectorUnitRange
  @test nondual_sector(sr) == SU((1, 0))
  @test full_range(sr) isa UnitRange
  @test full_range(sr) == 4:10
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
  @test blocklabels(sr) == [SU((1, 0))]
  @test sector_multiplicity(sr) == 2
  @test sector_multiplicities(sr) == [2]

  srd = dual(sr)
  @test nondual_sector(srd) == SU((1, 0))
  @test space_isequal(srd, sectorrange(SU((1, 0)), 2, true))

  srf = flip(sr)
  @test nondual_sector(srf) == SU((1, 1))
  @test space_isequal(srf, sectorrange(SU((1, 1)), 2, true))

  # getindex
  @test_throws BoundsError sr[0]
  @test_throws BoundsError sr[7]
  for i in 1:6
    @test sr[i] == i
  end
  @test sr[2:3] == 2:3
  @test sr[Block(1)] === sr
  @test_throws BlockBoundsError sr[Block(2)]

  sr2 = sr[(:, 2)]
  @test sr2 isa SectorUnitRange
  @test space_isequal(sr2, sectorrange(SU((1, 0)), 4:6))
  sr3 = sr[(:, 1:2)]
  @test sr3 isa SectorUnitRange
  @test space_isequal(sr3, sectorrange(SU((1, 0)), 1:6))
end
