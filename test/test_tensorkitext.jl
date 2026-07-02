using GradedArrays: GradedOneTo, SU2, U1, dual, gradedrange
using TensorAlgebra: to_range
using TensorKit: TensorKit, GradedSpace, SU2Irrep, Vect, dim
using Test: @test, @test_throws, @testset

# `to_range` on `sector => multiplicity` pairs is routed by symmetry. Abelian sectors keep
# the block-sparse `GradedOneTo` backend; non-abelian sectors have no block-sparse
# representation and build a native TensorKit `GradedSpace` (this extension). This lets
# `Index([SU2(0) => 1, …])` reach the same `TensorMap`-backed `ITensor` as passing a native
# TensorKit space to `Index`.
@testset "GradedArraysTensorKitExt" begin
    # Abelian sectors are untouched by the extension: still a block-sparse `GradedOneTo`.
    @testset "abelian stays block-sparse" begin
        r = to_range([U1(0) => 2, U1(1) => 3])
        @test r isa GradedOneTo
        @test !(r isa GradedSpace)
        @test r == gradedrange([U1(0) => 2, U1(1) => 3])
    end

    # Non-abelian sectors build a native TensorKit `GradedSpace` equal to the one produced
    # by TensorKit's own constructor.
    @testset "non-abelian builds a native GradedSpace" begin
        r = to_range([SU2(0) => 1, SU2(1) => 2])
        @test r isa GradedSpace
        @test r == Vect[SU2Irrep](0 => 1, 1 => 2)
        @test dim(r) == 1 * 1 + 3 * 2
    end

    # A raw list of TensorKit sectors (no `SectorRange` wrapper) is a valid input and builds
    # the same non-dual space.
    @testset "raw TensorKit sectors" begin
        r = to_range([SU2Irrep(0) => 1, SU2Irrep(1) => 2])
        @test r isa GradedSpace
        @test r == Vect[SU2Irrep](0 => 1, 1 => 2)
    end

    # The sector arrow rides inside the space: a shared dual flag makes a dual space.
    @testset "dual arrow rides inside the space" begin
        r = to_range([dual(SU2(0)) => 1, dual(SU2(1)) => 2])
        @test r isa GradedSpace
        @test r == TensorKit.dual(Vect[SU2Irrep](0 => 1, 1 => 2))
    end

    # A native TensorKit space has a single arrow, so mixed arrows have no representation.
    @testset "mixed arrows error" begin
        @test_throws ArgumentError to_range([SU2(0) => 1, dual(SU2(1)) => 2])
    end
end
