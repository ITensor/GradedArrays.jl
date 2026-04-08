using GradedArrays: GradedArrays
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :AbstractGradedArray,
        :AbstractGradedMatrix,
        :AbelianArray,
        :AbelianMatrix,
        :AbelianVector,
        :Fib,
        :GradedArrays,
        :GradedUnitRange,
        :Ising,
        :labels,
        :O2,
        :SectorArray,
        :SectorDelta,
        :SectorUnitRange,
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
