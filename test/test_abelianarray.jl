using BlockArrays: BlockArrays, Block, blocklength
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: GradedArrays, AbelianArray, AbstractGradedArray, GradedOneTo,
    SectorArray, SectorRange, U1, dual, gradedrange, isdual, labels, sector_multiplicities,
    sector_type, sectors
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "AbelianArray" begin
    # Helper: build U1 axes
    g1 = gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])
    g2 = gradedrange([TKS.U1Irrep(0) => 1, TKS.U1Irrep(-1) => 2])

    @testset "Construction" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        @test a isa AbelianArray{Float64, 2, TKS.U1Irrep, Matrix{Float64}}
        @test a isa AbstractGradedArray{Float64, 2}
        @test a isa AbstractArray{Float64, 2}
        @test size(a) == (5, 3)
        @test ndims(a) == 2

        # Tuple form constructor
        a2 = AbelianArray{Float64}(undef, (g1, g2))
        @test size(a2) == (5, 3)
    end

    @testset "Empty array has no stored blocks" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        @test isempty(collect(eachblockstoredindex(a)))
    end

    @testset "Block setindex!/getindex" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        # Block (1,1): U1(0) with mult 2 × U1(0) with mult 1 → 2×1
        data11 = reshape([1.0, 3.0], 2, 1)
        a[Block(1, 1)] = data11

        blk = a[Block(1, 1)]
        @test blk isa SectorArray
        @test blk.data == data11
        @test labels(blk) == (TKS.U1Irrep(0), TKS.U1Irrep(0))
        @test blk.isdual == (false, false)
    end

    @testset "Block getindex returns correct labels and isdual" begin
        g1_dual = gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])'
        a = AbelianArray{Float64}(undef, g1_dual, g2)
        data = ones(2, 1)
        a[Block(1, 1)] = data

        blk = a[Block(1, 1)]
        @test labels(blk) == (TKS.U1Irrep(0), TKS.U1Irrep(0))
        @test blk.isdual == (true, false)
    end

    @testset "Block getindex for unstored block returns zeros" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        blk = a[Block(2, 1)]
        @test blk isa SectorArray
        @test all(iszero, blk.data)
        @test size(blk) == (3, 1)
    end

    @testset "Single Block{N} argument" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        blk = a[Block(1, 1)]
        @test blk isa SectorArray
        @test all(isone, blk.data)
    end

    @testset "SectorArray block setindex!" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        # Block (1,1): 2×1
        sa = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(0)),
            (false, false),
            reshape([5.0, 7.0], 2, 1)
        )
        a[Block(1, 1)] = sa
        @test a[Block(1, 1)].data == reshape([5.0, 7.0], 2, 1)
    end

    @testset "eachblockstoredindex" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        a[Block(2, 2)] = ones(3, 2)

        stored = Set(collect(eachblockstoredindex(a)))
        @test Block(1, 1) in stored
        @test Block(2, 2) in stored
        @test length(stored) == 2
    end

    @testset "Scalar indexing errors" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        @test_throws ErrorException a[1, 1]
        @test_throws ErrorException (a[1, 1] = 42.0)
    end

    @testset "Dual axes" begin
        g1_dual = gradedrange([TKS.U1Irrep(0) => 2, TKS.U1Irrep(1) => 3])'
        g2_dual = gradedrange([TKS.U1Irrep(0) => 1, TKS.U1Irrep(-1) => 2])'
        a = AbelianArray{Float64}(undef, g1_dual, g2_dual)

        @test isdual(a.axes[1]) == true
        @test isdual(a.axes[2]) == true
        @test size(a) == (5, 3)

        a[Block(1, 1)] = ones(2, 1)
        blk = a[Block(1, 1)]
        @test blk.isdual == (true, true)
    end

    @testset "similar" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)

        a2 = similar(a)
        @test a2 isa AbelianArray{Float64, 2}
        @test size(a2) == size(a)
        @test isempty(collect(eachblockstoredindex(a2)))

        a3 = similar(a, ComplexF64)
        @test a3 isa AbelianArray{ComplexF64, 2}
        @test size(a3) == size(a)
    end

    @testset "sector_type" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        @test sector_type(typeof(a)) == SectorRange{TKS.U1Irrep}
    end

    @testset "Multiple block insertions" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        a[Block(1, 2)] = 2.0 * ones(2, 2)
        a[Block(2, 1)] = 3.0 * ones(3, 1)
        a[Block(2, 2)] = 4.0 * ones(3, 2)

        @test length(collect(eachblockstoredindex(a))) == 4
        @test a[Block(1, 2)].data == 2.0 * ones(2, 2)
        @test a[Block(2, 1)].data == 3.0 * ones(3, 1)
    end

    @testset "SU2 (non-abelian dimensions)" begin
        # SU2 j=1/2 has dim=2, j=1 has dim=3
        g_su2 = GradedOneTo(
            [TKS.SU2Irrep(1 // 2), TKS.SU2Irrep(1)],
            [1, 1],
            false
        )
        # Block lengths: dim(1/2)*1 = 2, dim(1)*1 = 3 => total 5
        @test length(g_su2) == 5

        a = AbelianArray{Float64}(undef, g_su2, g_su2)
        @test size(a) == (5, 5)

        # Block (1,1) is 2x2
        a[Block(1, 1)] = [1.0 2.0; 3.0 4.0]
        blk = a[Block(1, 1)]
        @test size(blk) == (2, 2)
        @test labels(blk) == (TKS.SU2Irrep(1 // 2), TKS.SU2Irrep(1 // 2))

        # Block (2,2) is 3x3
        a[Block(2, 2)] = ones(3, 3)
        blk2 = a[Block(2, 2)]
        @test size(blk2) == (3, 3)
        @test labels(blk2) == (TKS.SU2Irrep(1), TKS.SU2Irrep(1))
    end

    @testset "show" begin
        a = AbelianArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        s = sprint(show, MIME("text/plain"), a)
        @test occursin("AbelianArray", s)
        @test occursin("2×2-blocked", s)
        @test occursin("5×3", s)
        @test occursin("1 stored block", s)
    end

    @testset "blocks accessor" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        a = AbelianArray{Float64}(undef, g, dual(g))
        a[Block(1, 1)] = SectorArray((U1(0), dual(U1(0))), ones(2, 2))
        a[Block(2, 2)] = SectorArray((U1(1), dual(U1(1))), 2 * ones(3, 3))

        b = BlockArrays.blocks(a)
        @test size(b) == (2, 2)

        # Stored blocks return SectorArray
        b11 = b[1, 1]
        @test b11 isa SectorArray
        @test b11.data ≈ ones(2, 2)

        # Unstored blocks error
        @test_throws ErrorException b[1, 2]

        # Writing through blocks
        b[1, 1] = SectorArray((U1(0), dual(U1(0))), 5 * ones(2, 2))
        @test a[Block(1, 1)].data ≈ 5 * ones(2, 2)
    end

    @testset "fill! and zero!" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        a = AbelianArray{Float64}(undef, g, dual(g))
        a[Block(1, 1)] = SectorArray((U1(0), dual(U1(0))), ones(2, 2))

        # fill!(a, 0) zeros stored blocks in place
        fill!(a, 0)
        @test !isempty(a.blockdata)
        @test all(iszero, a.blockdata[(1, 1)])

        # fill! with nonzero errors
        @test_throws ArgumentError fill!(a, 1.0)

        # zero! zeros stored blocks in place (blocks stay allocated)
        a[Block(1, 1)] = SectorArray((U1(0), dual(U1(0))), ones(2, 2))
        GradedArrays.FI.zero!(a)
        @test !isempty(a.blockdata)
        @test all(iszero, a.blockdata[(1, 1)])
    end
end
