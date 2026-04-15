using BlockArrays: BlockArrays, Block, blockedrange, blocklength
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: GradedArrays, AbelianGradedArray, AbelianSectorArray,
    AbstractGradedArray, FusedGradedMatrix, GradedOneTo, SU2, SectorRange, U1, data,
    datalengths, dual, gradedrange, isdual, sectoraxes, sectors, sectortype
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "AbelianGradedArray" begin
    # Helper: build U1 axes
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])

    @testset "Construction" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        @test a isa AbelianGradedArray{Float64, 2, Matrix{Float64}, U1}
        @test a isa AbstractGradedArray{Float64, 2}
        @test a isa AbstractArray{Float64, 2}
        @test size(a) == (5, 3)
        @test ndims(a) == 2

        # Tuple form constructor
        a2 = AbelianGradedArray{Float64}(undef, (g1, g2))
        @test size(a2) == (5, 3)
    end

    @testset "Constructor allocates allowed blocks" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        stored = Set(collect(eachblockstoredindex(a)))
        @test Block(1, 1) in stored  # U1(0) × U1(0): charge 0
        @test Block(2, 2) in stored  # U1(1) × U1(-1): charge 0
        @test length(stored) == 2
        # Blocks are allocated but uninitialized (undef)
        @test size(a[Block(1, 1)]) == (2, 1)
        @test size(a[Block(2, 2)]) == (3, 2)
    end

    @testset "Block setindex!/getindex" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        # Block (1,1): U1(0) with mult 2 × U1(0) with mult 1 → 2×1
        data11 = reshape([1.0, 3.0], 2, 1)
        a[Block(1, 1)] = data11

        blk = a[Block(1, 1)]
        @test blk isa AbelianSectorArray
        @test data(blk) == data11
        @test sectoraxes(blk) == (U1(0), U1(0))
    end

    @testset "Block getindex returns correct sectors" begin
        g1_dual = gradedrange([U1(0) => 2, U1(1) => 3])'
        a = AbelianGradedArray{Float64}(undef, g1_dual, g2)
        data = ones(2, 1)
        a[Block(1, 1)] = data

        blk = a[Block(1, 1)]
        @test sectoraxes(blk) == (U1(0)', U1(0))
    end

    @testset "Block getindex for unstored block errors" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        @test_throws ErrorException a[Block(2, 1)]
    end

    @testset "Single Block{N} argument" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        blk = a[Block(1, 1)]
        @test blk isa AbelianSectorArray
        @test all(isone, data(blk))
    end

    @testset "AbelianSectorArray block setindex!" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        # Block (1,1): 2×1
        sa = AbelianSectorArray((U1(0), U1(0)), reshape([5.0, 7.0], 2, 1))
        a[Block(1, 1)] = sa
        @test data(a[Block(1, 1)]) == reshape([5.0, 7.0], 2, 1)
    end

    @testset "eachblockstoredindex" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        a[Block(2, 2)] = ones(3, 2)

        stored = Set(collect(eachblockstoredindex(a)))
        @test Block(1, 1) in stored
        @test Block(2, 2) in stored
        @test length(stored) == 2
    end

    @testset "Scalar indexing errors" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        @test_throws ErrorException a[1, 1]
        @test_throws ErrorException (a[1, 1] = 42.0)
    end

    @testset "Dual axes" begin
        g1_dual = gradedrange([U1(0) => 2, U1(1) => 3])'
        g2_dual = gradedrange([U1(0) => 1, U1(-1) => 2])'
        a = AbelianGradedArray{Float64}(undef, g1_dual, g2_dual)

        @test isdual(axes(a, 1)) == true
        @test isdual(axes(a, 2)) == true
        @test size(a) == (5, 3)

        a[Block(1, 1)] = ones(2, 1)
        blk = a[Block(1, 1)]
        @test sectoraxes(blk) == (U1(0)', U1(0)')
    end

    @testset "similar" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)

        a2 = similar(a)
        @test a2 isa AbelianGradedArray{Float64, 2}
        @test size(a2) == size(a)
        # similar now allocates all allowed blocks (same as constructor)
        @test length(collect(eachblockstoredindex(a2))) == 2

        a3 = similar(a, ComplexF64)
        @test a3 isa AbelianGradedArray{ComplexF64, 2}
        @test size(a3) == size(a)
    end

    @testset "sectortype" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        @test sectortype(typeof(a)) == U1
    end

    @testset "Stored block insertions" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        a[Block(2, 2)] = 4.0 * ones(3, 2)

        @test length(collect(eachblockstoredindex(a))) == 2
        @test data(a[Block(1, 1)]) == ones(2, 1)
        @test data(a[Block(2, 2)]) == 4.0 * ones(3, 2)

        # Setting unstored (non-allowed) blocks errors
        @test_throws ErrorException (a[Block(1, 2)] = 2.0 * ones(2, 2))
    end

    @testset "SU2 (non-abelian dimensions)" begin
        # SU2 j=1/2 has quantum dim=2, j=1 has quantum dim=3.
        # FusedGradedMatrix blocks store multiplicity data (without quantum dim).
        su2_sectors = [SU2(1 // 2), SU2(1)]
        blocks_su2 = [[1.0 2.0; 3.0 4.0], ones(3, 3)]
        m = FusedGradedMatrix(su2_sectors, blocks_su2)
        # size = sum(quantum_dim * multiplicity) per side = 2*2 + 3*3 = 13
        @test size(m) == (13, 13)

        # Block (1,1): SU2(1/2) with mult=2, quantum dim=2 → size 2*2 = 4
        blk = m[Block(1, 1)]
        @test size(blk) == (4, 4)

        # Block (2,2): SU2(1) with mult=3, quantum dim=3 → size 3*3 = 9
        blk2 = m[Block(2, 2)]
        @test size(blk2) == (9, 9)
    end

    @testset "show" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        s = sprint(show, MIME("text/plain"), a)
        @test occursin("AbelianGradedArray", s)
        @test occursin("2×2-blocked", s)
        @test occursin("5×3", s)
        @test occursin("2 stored block", s)
    end

    @testset "blocks accessor" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        a = AbelianGradedArray{Float64}(undef, g, dual(g))
        a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), ones(2, 2))
        a[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), 2 * ones(3, 3))

        b = BlockArrays.blocks(a)
        @test size(b) == (2, 2)

        # Stored blocks return AbelianSectorArray
        b11 = b[1, 1]
        @test b11 isa AbelianSectorArray
        @test data(b11) ≈ ones(2, 2)

        # Unstored blocks error
        @test_throws ErrorException b[1, 2]

        # Writing through blocks
        b[1, 1] = AbelianSectorArray((U1(0), dual(U1(0))), 5 * ones(2, 2))
        @test data(a[Block(1, 1)]) ≈ 5 * ones(2, 2)
    end

    @testset "fill! and zero!" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        a = AbelianGradedArray{Float64}(undef, g, dual(g))
        a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), ones(2, 2))

        # fill!(a, 0) zeros stored blocks in place
        fill!(a, 0)
        @test !isempty(a.blockdata)
        @test all(iszero, a.blockdata[(1, 1)])

        # fill! with nonzero errors
        @test_throws ArgumentError fill!(a, 1.0)

        # zero! zeros stored blocks in place (blocks stay allocated)
        a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), ones(2, 2))
        GradedArrays.FI.zero!(a)
        @test !isempty(a.blockdata)
        @test all(iszero, a.blockdata[(1, 1)])
    end
end

@testset "FusedGradedMatrix Pair constructor" begin
    m = FusedGradedMatrix([U1(0) => [1.0 2.0; 3.0 4.0], U1(1) => ones(3, 3)])
    @test m isa FusedGradedMatrix{Float64}
    @test data(m[Block(1, 1)]) == [1.0 2.0; 3.0 4.0]
    @test data(m[Block(2, 2)]) == ones(3, 3)
end

@testset "FusedGradedMatrix undef constructor" begin
    sectors = [U1(0), U1(1)]
    cod_bls = [2, 3]
    dom_bls = [1, 2]

    @testset "Convenience constructor (defaults D = Matrix{T})" begin
        m = FusedGradedMatrix{Float64}(undef, sectors, cod_bls, dom_bls)
        @test m isa FusedGradedMatrix{Float64, Matrix{Float64}, U1}
        @test length(m.sectors) == 2
        @test m.sectors == sectors
        @test size(m.blocks[1]) == (2, 1)
        @test size(m.blocks[2]) == (3, 2)
    end

    @testset "Fully parameterized constructor" begin
        m = FusedGradedMatrix{Float64, Matrix{Float64}, U1}(
            undef, sectors, (blockedrange(cod_bls), blockedrange(dom_bls))
        )
        @test m isa FusedGradedMatrix{Float64, Matrix{Float64}, U1}
        @test size(m.blocks[1]) == (2, 1)
    end

    @testset "Tuple BlockedOneTo form" begin
        m = FusedGradedMatrix{Float64}(
            undef, sectors, (blockedrange([2, 3]), blockedrange([1, 2]))
        )
        @test m isa FusedGradedMatrix{Float64, Matrix{Float64}, U1}
        @test size(m.blocks[1]) == (2, 1)
        @test size(m.blocks[2]) == (3, 2)
    end

    @testset "Rejects mismatched lengths" begin
        @test_throws Exception FusedGradedMatrix{Float64}(
            undef, sectors, cod_bls, [1]
        )
    end

    @testset "Rejects unsorted sectors" begin
        @test_throws ArgumentError FusedGradedMatrix{Float64}(
            undef, [U1(1), U1(0)], cod_bls, dom_bls
        )
    end

    @testset "Rejects non-unique sectors" begin
        @test_throws ArgumentError FusedGradedMatrix{Float64}(
            undef, [U1(0), U1(0)], [2, 3], [1, 2]
        )
    end
end
