using BlockArrays: Block, blocks
using GradedArrays:
    U1, dual, gradedrange, storedvalues, with_block_indexing, with_scalar_indexing
using Test: @test, @test_throws, @testset

@testset "indexing guards" begin
    g = gradedrange([U1(0) => 2, U1(1) => 2])
    a = zeros(Float64, (g,), (dual(g),))
    B = Block(1, 1)

    @testset "scalar indexing is off by default" begin
        @test_throws ErrorException a[1, 1]
        @test_throws ErrorException (a[1, 1] = 1.0)
        with_scalar_indexing() do
            @test a[1, 1] == 0.0
            a[1, 1] = 2.0
            @test a[1, 1] == 2.0
        end
    end

    @testset "block indexing is off by default" begin
        @test_throws ErrorException a[B]
        @test_throws ErrorException view(a, B)
        with_block_indexing() do
            @test a[B] isa AbstractArray
        end
    end

    @testset "the two guards are independent" begin
        # Enabling scalar indexing does not enable block indexing.
        with_scalar_indexing() do
            @test_throws ErrorException a[B]
        end
        # Enabling block indexing does not enable scalar indexing.
        with_block_indexing() do
            @test_throws ErrorException a[1, 1]
        end
    end

    @testset "guards have dynamic extent" begin
        with_scalar_indexing() do
            @test a[1, 1] isa Float64
        end
        # Disabled again once the block exits.
        @test_throws ErrorException a[1, 1]
        # `allow = false` re-disables within an enabled scope, and the outer scope is restored.
        with_scalar_indexing() do
            with_scalar_indexing(false) do
                @test_throws ErrorException a[1, 1]
            end
            @test a[1, 1] isa Float64
        end
    end

    @testset "block-container interface is guarded" begin
        # `storedvalues`/`getstoredindex` on the block container reach through the guarded block
        # views, so they require `with_block_indexing` like any other block access.
        @test_throws ErrorException storedvalues(blocks(a))
        with_block_indexing() do
            @test storedvalues(blocks(a)) isa AbstractVector
        end
    end
end
