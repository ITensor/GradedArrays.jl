using GradedArrays:
    GradedArrays, AbstractGradedArray, U1, dual, gradedrange, isdual, mortar_axis, sectors
using TensorAlgebra: TensorAlgebra, cat_axes, cat_similar, cat_style
using Test: @test, @test_broken, @test_throws, @testset

@testset "cat / directsum" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(1) => 2])

    @testset "cat_axis is concat-order" begin
        # Block-append preserving sector order: sectors of g1 then g2, not merged or sorted.
        ax = TensorAlgebra.cat_axis(g1, g2)
        @test ax == mortar_axis([g1, g2])
        @test sectors(ax) == [U1(0), U1(1), U1(0), U1(1)]
        @test length(ax) == length(g1) + length(g2)
        # n-ary folds pairwise, staying concat-order.
        @test TensorAlgebra.cat_axis(g1, g2, g1) == mortar_axis([g1, g2, g1])
    end

    @testset "cat_axis duality" begin
        @test isdual(TensorAlgebra.cat_axis(dual(g1), dual(g2)))
        @test !isdual(TensorAlgebra.cat_axis(g1, g2))
        # Mismatched arrows cannot be concatenated.
        @test_throws ArgumentError TensorAlgebra.cat_axis(dual(g1), g2)
    end

    @testset "cat_axes" begin
        a1 = randn(g1, g1)
        a2 = randn(g2, g2)
        @test cat_axes(Val((1, 2)), a1, a2) ==
            (mortar_axis([g1, g2]), mortar_axis([g1, g2]))
    end

    @testset "cat_similar allocates a graded array with concat axes" begin
        a1 = randn(g1, g1)
        a2 = randn(g2, g2)
        ax = cat_axes(Val((1, 2)), a1, a2)
        d = cat_similar(cat_style(Val((1, 2)), a1, a2), Float64, ax, a1, a2)
        @test d isa AbstractGradedArray
        @test eltype(d) == Float64
        @test axes(d) == (mortar_axis([g1, g2]), mortar_axis([g1, g2]))
    end

    @testset "cat data placement" begin
        a1 = randn(g1, g1)
        a2 = randn(g2, g2)
        @test let r = TensorAlgebra.concatenate((1, 2), a1, a2)
            axes(r) == (mortar_axis([g1, g2]), mortar_axis([g1, g2])) &&
                Array(r) == cat(Array(a1), Array(a2); dims = (1, 2))
        end

        b1 = randn(g1, g1)
        b2 = randn(g2, g1)
        @test let r = TensorAlgebra.concatenate(1, b1, b2)
            axes(r) == (mortar_axis([g1, g2]), g1) &&
                Array(r) == cat(Array(b1), Array(b2); dims = 1)
        end
    end

    @testset "directsum defaults to cat" begin
        a1 = randn(g1, g1)
        a2 = randn(g2, g2)
        @test let r = TensorAlgebra.directsum(a1, a2; dims = (1, 2))
            Array(r) == cat(Array(a1), Array(a2); dims = (1, 2))
        end
    end

    @testset "Base.cat routes through concatenate" begin
        a1 = randn(g1, g1)
        a2 = randn(g2, g2)
        @test let r = cat(a1, a2; dims = (1, 2))
            Array(r) == cat(Array(a1), Array(a2); dims = (1, 2))
        end
    end
end
