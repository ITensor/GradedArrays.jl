using BlockArrays: Block, blockedrange
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: GradedArrays, AbelianGradedArray, AbelianSectorArray, Data,
    FusedGradedMatrix, GradedOneTo, SectorMatrix, U1, data, dual, gradedrange, sectoraxes,
    sectors
using Test: @test, @test_throws, @testset

@testset "Data indexing" begin
    @testset "FusedGradedMatrix" begin
        sectors = [U1(0), U1(1)]
        m = FusedGradedMatrix(sectors, [ones(2, 3), 2 * ones(4, 5)])

        @testset "Data getindex returns copy of raw block" begin
            d = m[Data(1, 1)]
            @test d isa Matrix{Float64}
            @test d == ones(2, 3)
            # Verify it's a copy, not a view
            d[1, 1] = 999.0
            @test m.blocks[1][1, 1] == 1.0
        end

        @testset "Data getindex second block" begin
            d = m[Data(2, 2)]
            @test d == 2 * ones(4, 5)
        end

        @testset "Data setindex! copies into raw block" begin
            sectors2 = [U1(0), U1(1)]
            m2 = FusedGradedMatrix(sectors2, [zeros(2, 3), zeros(4, 5)])
            new_data = 7 * ones(2, 3)
            m2[Data(1, 1)] = new_data
            @test m2.blocks[1] == new_data
            # Verify it's a copy, not aliased
            new_data[1, 1] = 0.0
            @test m2.blocks[1][1, 1] == 7.0
        end

        @testset "Data setindex! size mismatch errors" begin
            m3 = FusedGradedMatrix([U1(0)], [zeros(2, 3)])
            @test_throws DimensionMismatch (m3[Data(1, 1)] = ones(5, 5))
        end

        @testset "Data off-diagonal errors" begin
            @test_throws ErrorException m[Data(1, 2)]
            @test_throws ErrorException (m[Data(1, 2)] = ones(2, 5))
        end

        @testset "Block setindex! with SectorMatrix" begin
            sectors2 = [U1(0), U1(1)]
            m4 = FusedGradedMatrix(sectors2, [zeros(2, 3), zeros(4, 5)])
            sm = SectorMatrix(U1(0), 7 * ones(2, 3))
            m4[Block(1, 1)] = sm
            @test m4.blocks[1] == 7 * ones(2, 3)
        end

        @testset "Block setindex! verifies sector" begin
            sectors2 = [U1(0), U1(1)]
            m5 = FusedGradedMatrix(sectors2, [zeros(2, 3), zeros(4, 5)])
            sm_wrong = SectorMatrix(U1(2), ones(2, 3))
            @test_throws DimensionMismatch (m5[Block(1, 1)] = sm_wrong)
        end

        @testset "Block setindex! off-diagonal errors" begin
            sectors2 = [U1(0), U1(1)]
            m6 = FusedGradedMatrix(sectors2, [zeros(2, 3), zeros(4, 5)])
            sm = SectorMatrix(U1(0), ones(2, 3))
            @test_throws Exception (m6[Block(1, 2)] = sm)
        end
    end

    @testset "AbelianGradedArray" begin
        g1 = gradedrange([U1(0) => 2, U1(1) => 3])
        g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        a[Block(2, 2)] = 2 * ones(3, 2)

        @testset "Data getindex returns copy of raw block" begin
            d = a[Data(1, 1)]
            @test d isa Matrix{Float64}
            @test d == ones(2, 1)
            # Verify it's a copy
            d[1, 1] = 999.0
            @test data(a[Block(1, 1)]) == ones(2, 1)
        end

        @testset "Data getindex second block" begin
            d = a[Data(2, 2)]
            @test d == 2 * ones(3, 2)
        end

        @testset "Data setindex! copies into raw block" begin
            a2 = AbelianGradedArray{Float64}(undef, g1, g2)
            a2[Block(1, 1)] = zeros(2, 1)
            new_data = 5 * ones(2, 1)
            a2[Data(1, 1)] = new_data
            @test data(a2[Block(1, 1)]) == 5 * ones(2, 1)
            # Verify it's a copy
            new_data[1, 1] = 0.0
            @test data(a2[Block(1, 1)])[1, 1] == 5.0
        end

        @testset "Data access on unstored block errors" begin
            @test_throws Exception a[Data(1, 2)]
            @test_throws Exception (a[Data(1, 2)] = ones(2, 2))
        end
    end
end
