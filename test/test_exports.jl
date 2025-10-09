using GradedArrays: GradedArrays

using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :GradedArrays,
        :SU2,
        :U1,
        :Z,
        :dag,
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
