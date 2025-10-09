using BlockArrays: BlockedOneTo, blockedrange, blockisequal

using GradedArrays:
    NoSector, dag, dual, flip, isdual, map_sectors, sectors, space_isequal, ungrade
using Test: @test, @testset
using TensorProducts: OneToOne

@testset "GradedUnitRange interface for AbstractUnitRange" begin
    a0 = OneToOne()
    @test !isdual(a0)
    @test dual(a0) isa OneToOne
    @test space_isequal(a0, a0)
    @test space_isequal(a0, dual(a0))
    @test only(sectors(a0)) == NoSector()
    @test ungrade(a0) === a0
    @test map_sectors(identity, a0) === a0
    @test dag(a0) === a0

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
