using GradedArrays: GradedArrays
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :AbstractGradedArray,
        :AbelianArray,
        :Fib,
        :GradedArrays,
        :GradedIndices,
        :Ising,
        :labels,
        :O2,
        :SectorArray,
        :SectorDelta,
        :SectorIndices,
        :SectorMatrix,
        :SectorRange,
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
    ]
    @test issetequal(names(GradedArrays), exports)
end
