using BlockArrays:
    Block, BlockBoundsError, BlockRange,
    blockaxes, blocklasts, blocklength, blocklengths,
    blockisequal, blocks, findblock
using GradedArrays:
    U1, SU2,
    SectorOneTo, SectorUnitRange, SectorVector,
    dual, flip, isdual,
    quantum_dimension,
    sector, sector_multiplicities, sector_multiplicity, sector_type,
    sectorrange, ungrade, sectors, space_isequal
using Test: @test, @test_throws, @testset, @test_broken
using TestExtras: @constinferred

@testset "SectorUnitRange" begin
    sr = sectorrange(SU2(1 / 2), 2)
    @test sr isa SectorUnitRange

    # accessors
    @test sector(sr) == SU2(1 / 2)
    @test ungrade(sr) isa Base.OneTo
    @test ungrade(sr) == 1:4
    @test !isdual(sr)

    # Base interface
    @test first(sr) == 1
    @test last(sr) == 4
    @test length(sr) == 4
    @test firstindex(sr) == 1
    @test lastindex(sr) == 4
    @test eltype(sr) ≡ Int
    @test step(sr) == 1
    @test eachindex(sr) == Base.oneto(4)
    @test only(axes(sr)) isa SectorOneTo
    @test sector(only(axes(sr))) == sector(sr)
    @test only(axes(sr)) == 1:4
    @test iterate(sr) == (1, 1)
    for i in 1:3
        @test iterate(sr, i) == (i + 1, i + 1)
    end
    @test isnothing(iterate(sr, 4))

    # Base.Slice
    @test axes(Base.Slice(sr)) ≡ (sr,)
    @test Base.axes1(Base.Slice(sr)) ≡ sr
    @test Base.unsafe_indices(Base.Slice(sr)) ≡ (sr,)

    @test sr == 1:4
    @test sr == sr
    @test space_isequal(sr, sr)

    sr = sectorrange(SU2(1 / 2) => 2)
    @test sr isa SectorUnitRange
    @test sector(sr) == SU2(1 / 2)
    @test ungrade(sr) isa Base.OneTo
    @test ungrade(sr) == 1:4
    @test !isdual(sr)

    sr = sectorrange(SU2(1 / 2) => 2, true)
    @test sr isa SectorUnitRange
    @test sector(sr) == SU2(1 / 2)'
    @test ungrade(sr) isa Base.OneTo
    @test ungrade(sr) == 1:4
    @test isdual(sr)

    sr = sectorrange(SU2(1 / 2), 4:10, true)
    @test sr isa SectorUnitRange
    @test sector(sr) == SU2(1 / 2)'
    # TODO: what should ungrade return?
    @test_broken ungrade(sr) isa UnitRange
    @test_broken ungrade(sr) == 4:10
    @test isdual(sr)

    sr = sectorrange(SU2(1 / 2), 2)
    @test !space_isequal(sr, sectorrange(SU2(1), 2))
    @test !space_isequal(sr, sectorrange(SU2(1 / 2), 2:7))
    @test !space_isequal(sr, sectorrange(SU2(1), 2, true))
    @test !space_isequal(sr, sectorrange(SU2(1 / 2), 2, true))

    sr2 = copy(sr)
    @test sr2 isa SectorUnitRange
    @test space_isequal(sr, sr2)
    sr3 = deepcopy(sr)
    @test sr3 isa SectorUnitRange
    @test space_isequal(sr, sr3)

    # BlockArrays interface
    @test blockaxes(sr) isa Tuple{BlockRange{1, <:Tuple{Base.OneTo}}}
    @test space_isequal(sr[Block(1)], sr)
    @test only(blocklasts(sr)) == 4
    @test findblock(sr, 2) == Block(1)

    @test blocklength(sr) == 1
    @test blocklengths(sr) == [4]
    @test_broken only(blocks(sr)) == 1:4
    @test blockisequal(sr, sr)

    # GradedUnitRanges interface
    @test sector_type(sr) ≡ SU2
    @test sector_type(typeof(sr)) ≡ SU2
    @test sectors(sr) == [SU2(1 / 2)]
    @test sector_multiplicity(sr) == 2
    @test sector_multiplicities(sr) == [2]
    @test quantum_dimension(sr) == 4

    srd = dual(sr)
    @test sector(srd) == dual(sector(sr))
    @test space_isequal(srd, sectorrange(SU2(1 / 2), 2, true))
    @test sectors(srd) == dual.(sectors(sr))

    srf = flip(sr)
    @test sector(srf) == flip(sector(sr))
    @test isdual(srf) == isdual(sr)
    @test space_isequal(srf, sectorrange(flip(SU2(1 / 2)), 2))

    # getindex
    @test_throws BoundsError sr[0]
    @test_throws BoundsError sr[7]
    @test (@constinferred getindex(sr, 1)) isa Int64
    for i in 1:4
        @test sr[i] == i
    end
    @test_broken sr[2:3] == 2:3
    @test_broken (@constinferred getindex(sr, 2:3)) isa UnitRange
    @test sr[Block(1)] ≡ sr
    @test_throws BlockBoundsError sr[Block(2)]

    # TODO: do we want to reinstate this syntax?
    # sr2 = (@constinferred getindex(sr, (:, 2)))
    # @test sr2 isa SectorUnitRange
    # @test space_isequal(sr2, sectorrange(SU2(1 / 2), 3:4))
    # sr3 = (@constinferred getindex(sr, (:, 1:2)))
    # @test sr3 isa SectorUnitRange
    # @test space_isequal(sr3, sectorrange(SU2(1 / 2), 1:4))

    # Abelian slicing
    srab = sectorrange(U1(1), 3)
    @test (@constinferred getindex(srab, 2:2)) isa SectorUnitRange
    @test_broken space_isequal(srab[2:2], sectorrange(U1(1), 2:2))
    @test_broken space_isequal(dual(srab)[2:2], sectorrange(U1(1), 2:2, true))
    # TODO: do we need to add SectorVector?
    @test_broken srab[[1, 3]] isa SectorVector{Int}
    @test_broken sector(srab[[1, 3]]) == sector(srab)
    @test_broken ungrade(srab[[1, 3]]) == [1, 3]
    @test length(srab[[1, 3]]) == 2
    @test_broken space_isequal(only(axes(srab[[1, 3]])), sectorrange(U1(1), 2))

    # Slice sector range with sector range
    sr1 = sectorrange(U1(1), 4)
    sr2 = sectorrange(U1(1), 3)
    @test_broken sr1[sr2] ≡ sr2

    sr = sectorrange(U1(1), 4)
    r = Base.OneTo(4)
    @test Broadcast.axistype(sr, sr) ≡ sr
    @test_broken Broadcast.axistype(sr, r) ≡ r
    @test_broken Broadcast.axistype(r, sr) ≡ r
end
