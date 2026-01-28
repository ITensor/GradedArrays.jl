using BlockArrays: BlockedOneTo, blockedrange, blockisequal

using GradedArrays:
    NoSector, dag, dual, flip, isdual, map_sectors, sectors, space_isequal, ungrade
using Test: @test, @testset

@testset "GradedUnitRange interface for AbstractUnitRange" begin
    a = 1:3
    ad = dual(a)
    af = flip(a)
    @test !isdual(a)
    @test !isdual(ad)
    @test !isdual(af)
    @test ad isa UnitRange
    @test af isa UnitRange
    @test space_isequal(ad, a)
    @test space_isequal(af, a)
    @test only(sectors(a)) == NoSector()
    @test ungrade(a) === a
    @test map_sectors(identity, a) === a
    @test dag(a) === a

    a = blockedrange([2, 3])
    ad = dual(a)
    af = flip(a)
    @test !isdual(a)
    @test !isdual(ad)
    @test ad isa BlockedOneTo
    @test af isa BlockedOneTo
    @test blockisequal(ad, a)
    @test blockisequal(af, a)
    @test sectors(a) == [NoSector(), NoSector()]
    @test ungrade(a) === a
    @test map_sectors(identity, a) === a
    @test dag(a) === a
end
