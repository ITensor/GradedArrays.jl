using GradedArrays: GradedArrays

using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :GradedArrays,
    :U1,
    :Z,
    :blocklabels,
    :dag,
    :dual,
    :flip,
    :gradedrange,
    :isdual,
    :sectorrange,
    :sector_type,
    :space_isequal,
  ]
  @test issetequal(names(GradedArrays), exports)
end
