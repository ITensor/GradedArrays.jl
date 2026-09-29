using GradedArrays: GradedArrays
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :GradedArrays,
        :Sector,
        :TrivialSector,
        :U1,
        :SU2,
        :CU1,
        :SUN,
        :Z,
        :Z2,
        :fZ2,
        :fU1,
        :fSU2,
        :sectorproduct,
        :×,
        :GradedArray,
        :gradedrange,
        :dual,
        :isdual,
    ]
    if VERSION >= v"1.11"
        # Marked `public` (not exported); `public` names appear in `names(...)` on Julia 1.11+.
        append!(
            exports,
            [
                :GradedContract,
                :OrientedSector,
                :TensorKitSector,
                :sectors,
                :with_scalar_indexing,
                :with_block_indexing,
            ]
        )
    end
    @test issetequal(names(GradedArrays), exports)
end
