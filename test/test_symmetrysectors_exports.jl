using GradedArrays.SymmetrySectors: SymmetrySectors
using Test: @test, @testset
@testset "Test exports" begin
  exports = [:SymmetrySectors, :U1, :Z, :dual]
  @test issetequal(names(SymmetrySectors), exports)
end
