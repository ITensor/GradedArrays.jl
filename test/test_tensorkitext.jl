using GradedArrays: GradedArrays, GradedOneTo, SU2, U1, dual, gradedrange
using TensorAlgebra: to_range
using TensorKit: TensorKit, GradedSpace, SU2Irrep, Vect, dim
using Test: @test, @test_throws, @testset

# `to_range` on `sector => multiplicity` pairs is routed by symmetry. Abelian sectors always keep the
# block-sparse `GradedOneTo` backend. Non-abelian sectors have no block-sparse representation on the
# abelian backend, so they build a native TensorKit `GradedSpace` (this extension); on the fusion
# backend `FusionArray` represents them via its coupled matrix, so they stay a `GradedOneTo`.
const FUSION_BACKEND = GradedArrays.graded_backend == "fusion"

@testset "GradedArraysTensorKitExt" begin
    # Abelian sectors are untouched by the extension: still a block-sparse `GradedOneTo`.
    @testset "abelian stays block-sparse" begin
        r = to_range([U1(0) => 2, U1(1) => 3])
        @test r isa GradedOneTo
        @test !(r isa GradedSpace)
        @test r == gradedrange([U1(0) => 2, U1(1) => 3])
    end

    @testset "non-abelian sectors" begin
        r = to_range([SU2(0) => 1, SU2(1) => 2])
        if FUSION_BACKEND
            # The fusion backend keeps non-abelian sectors as a block-sparse `GradedOneTo`.
            @test r isa GradedOneTo
            @test r == gradedrange([SU2(0) => 1, SU2(1) => 2])
        else
            # The abelian backend has no block-sparse representation, so it builds a native space.
            @test r isa GradedSpace
            @test r == Vect[SU2Irrep](0 => 1, 1 => 2)
            @test dim(r) == 1 * 1 + 3 * 2
        end
    end

    # The sector arrow rides inside the result: a shared dual flag makes a dual space/range.
    @testset "dual arrow" begin
        r = to_range([dual(SU2(0)) => 1, dual(SU2(1)) => 2])
        if FUSION_BACKEND
            @test r == dual(gradedrange([SU2(0) => 1, SU2(1) => 2]))
        else
            @test r isa GradedSpace
            @test r == TensorKit.dual(Vect[SU2Irrep](0 => 1, 1 => 2))
        end
    end

    # A single-arrow representation cannot hold mixed arrows, on either backend.
    @testset "mixed arrows error" begin
        @test_throws ArgumentError to_range([SU2(0) => 1, dual(SU2(1)) => 2])
    end
end
