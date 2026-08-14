using GradedArrays: GradedArrays
using Test: @test, @testset
@testset "Test exports" begin
    exports = [
        :AbstractGradedOneTo,
        :AbstractSectorArray,
        :AbstractSectorDelta,
        :UniqueSectorArray,
        :UniqueSectorDelta,
        :UniqueSectorMatrix,
        :UniqueSectorVector,
        :Data,
        :FusedGradedDiagonal,
        :FusedGradedMatrix,
        :FusedGradedOneTo,
        :FusedGradedVector,
        :GradedArray,
        :GradedArrays,
        :GradedBlockAlgorithm,
        :GradedOneTo,
        :codomain,
        :data,
        :domain,
        :dataaxes,
        :dataaxes1,
        :datalength,
        :datalengths,
        :eachdataaxis,
        :eachsectoraxis,
        :SectorIdentity,
        :FusedSectorMatrix,
        :FusedSectorVector,
        :SectorOneTo,
        :SectorRange,
        :SU2,
        :TrivialSector,
        :U1,
        :Z,
        :Z2,
        :dual,
        :flip,
        :fusedgradedrange,
        :gradedrange,
        :isdual,
        :sector,
        :sectoraxes,
        :sectoraxes1,
        :sectorlength,
        :sectorlengths,
        :sectors,
        :sectortype,
    ]
    if VERSION >= v"1.11"
        # Marked `public` (not exported); `public` names appear in `names(...)` on Julia 1.11+.
        append!(exports, [:with_scalar_indexing, :with_block_indexing])
    end
    @test issetequal(names(GradedArrays), exports)
end
