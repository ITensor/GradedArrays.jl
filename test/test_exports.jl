using GradedArrays: GradedArrays

using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :Fib,
        :GradedArrays,
        :GradedArray,
        :GradedOneTo,
        :GradedUnitRange,
        :Ising,
        :O2,
        :SectorArray,
        :SectorDelta,
        :SectorMatrix,
        :SectorOneTo,
        :SectorRange,
        :SectorUnitRange,
        :SU2,
        :TrivialSector,
        :U1,
        :Z,
        :Z2,
        :dual,
        :flip,
        :gradedrange,
        :isdual,
        :sector,
        :sector_multiplicities,
        :sector_multiplicity,
        :sectorrange,
        :sectors,
        :sector_type,
        :space_isequal,
        :ungrade,
    ]
    @test issetequal(names(GradedArrays), exports)
end
