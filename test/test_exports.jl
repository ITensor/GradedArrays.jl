using GradedArrays: GradedArrays
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :GradedArrays,
        :TrivialSector,
        :U1,
        :SU2,
        :Z,
        :Z2,
        :GradedArray,
        :gradedrange,
        :dual,
        :isdual,
    ]
    if VERSION >= v"1.11"
        # Marked `public` (not exported); `public` names appear in `names(...)` on Julia 1.11+.
        append!(
            exports,
            [:SectorRange, :sectors, :with_scalar_indexing, :with_block_indexing]
        )
    end
    @test issetequal(names(GradedArrays), exports)
end
