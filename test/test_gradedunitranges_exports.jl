using GradedArrays.GradedUnitRanges: GradedUnitRanges
using Test: @test, @testset
@testset "Test exports" begin
  exports = [:GradedUnitRanges, :gradedrange]
  @test issetequal(names(GradedUnitRanges), exports)
end
