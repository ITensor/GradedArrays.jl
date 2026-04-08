import GradedArrays
using BlockArrays: Block, blocklength
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: AbelianArray, FusedSectorMatrix, GradedIndices, SectorArray,
    SectorDelta, SectorRange, U1, dual, flip, gradedrange, isdual, label, sector,
    sector_multiplicities, sector_type, sectormergesort, sectorrange, sectors,
    tensor_product, ⊗
using Random: randn!
using TensorAlgebra: TensorAlgebra, contract, linearbroadcasted, matricize
using Test: @test, @test_throws, @testset

@testset "SectorArray linear broadcasting" begin
    s = SectorArray(
        (U1(0), dual(U1(0))),
        randn!(Matrix{ComplexF64}(undef, 2, 2))
    )
    t = SectorArray(
        (U1(0), dual(U1(0))),
        randn!(Matrix{ComplexF64}(undef, 2, 2))
    )
    @test s isa SectorArray
    @test t isa SectorArray

    α = 2.0
    β = -3.0

    st = α .* s .+ β .* t
    @test st isa SectorArray
    @test st.data isa Matrix
    @test Array(st) ≈ α .* Array(s) .+ β .* Array(t)
    @test axes(st) == axes(s)

    conjdiff = conj.(s) .- t ./ β
    @test conjdiff isa SectorArray
    @test conjdiff.data isa Matrix
    @test Array(conjdiff) ≈ conj.(Array(s)) .- Array(t) ./ β
    @test axes(conjdiff) == axes(s)

    @test_throws ArgumentError s .* t
    @test_throws ArgumentError exp.(s)
end

@testset "SectorArray scalar multiplication materializes on broadcast" begin
    s = SectorArray(
        (U1(0), dual(U1(0))),
        randn!(Matrix{Float64}(undef, 2, 2))
    )

    materialized = 2 .* s
    @test materialized isa SectorArray
    @test materialized.data isa Matrix
    @test materialized[1, 1] == 2 * s[1, 1]
    @test Array(materialized) ≈ 2 .* Array(s)

    scaled_mul = 2 * s
    @test scaled_mul isa SectorArray
    @test scaled_mul.data isa Matrix
    @test scaled_mul[1, 1] == 2 * s[1, 1]
    @test Array(scaled_mul) ≈ 2 .* Array(s)
end

@testset "SectorArray permutedims (bosonic)" begin
    data = randn!(Matrix{Float64}(undef, 3, 2))
    s = SectorArray((U1(0), dual(U1(1))), data)
    sp = permutedims(s, (2, 1))
    @test sp isa SectorArray
    @test label(sp, 1) == label(U1(1))
    @test label(sp, 2) == label(U1(0))
    @test Array(sp) ≈ permutedims(data)
end

@testset "SectorArray permutedims (3D bosonic)" begin
    data = randn!(Array{Float64}(undef, 2, 3, 4))
    s = SectorArray((U1(0), U1(1), U1(2)), data)
    sp = permutedims(s, (3, 1, 2))
    @test sp isa SectorArray
    @test sector(sp, 1) == U1(2)
    @test sector(sp, 2) == U1(0)
    @test sector(sp, 3) == U1(1)
    @test Array(sp) ≈ permutedims(data, (3, 1, 2))
end

@testset "AbelianArray permutedims" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianArray{Float64}(undef, g1, g2)

    # Set a block
    block_data = randn!(Matrix{Float64}(undef, 2, 2))
    a[Block(1, 2)] = SectorArray((U1(0), U1(-1)), block_data)

    ap = permutedims(a, (2, 1))
    @test ap isa AbelianArray
    @test axes(ap, 1) == g2
    @test axes(ap, 2) == g1

    # The block (1,2) in a should map to block (2,1) in ap
    ap_block = ap[Block(2, 1)]
    @test Array(ap_block) ≈ permutedims(block_data)
end

@testset "AbelianArray linear broadcasting" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianArray{Float64}(undef, g1, g2)
    b = AbelianArray{Float64}(undef, g1, g2)

    block_a = randn!(Matrix{Float64}(undef, 2, 2))
    block_b = randn!(Matrix{Float64}(undef, 2, 2))
    a[Block(1, 2)] = SectorArray((U1(0), U1(-1)), block_a)
    b[Block(1, 2)] = SectorArray((U1(0), U1(-1)), block_b)

    α = 2.0
    β = -3.0
    c = α .* a .+ β .* b
    @test c isa AbelianArray
    c_block = c[Block(1, 2)]
    @test Array(c_block) ≈ α .* block_a .+ β .* block_b
end

@testset "sectormergesort on AbelianArray" begin
    # Axis with repeated sectors: U1(1) appears at blocks 1 and 3
    g1 = gradedrange([U1(1) => 2, U1(0) => 1, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianArray{Float64}(undef, g1, g2)

    a[Block(1, 2)] = SectorArray((U1(1), U1(-1)), ones(2, 2))
    a[Block(3, 2)] = SectorArray((U1(1), U1(-1)), 2 * ones(3, 2))

    a_merged = sectormergesort(a)

    # Sectors should be sorted and unique after merge
    @test sectors(a_merged.axes[1]) == [U1(0), U1(1)]
    @test sector_multiplicities(a_merged.axes[1]) == [1, 5]
    @test sectors(a_merged.axes[2]) == [U1(-1), U1(0)]
    @test sector_multiplicities(a_merged.axes[2]) == [2, 1]

    # The merged U1(1) block should stack the two source blocks (2×2 + 3×2 → 5×2)
    merged_block = a_merged[Block(2, 1)]
    @test size(merged_block) == (5, 2)
    @test merged_block.data[1:2, :] ≈ ones(2, 2)
    @test merged_block.data[3:5, :] ≈ 2 * ones(3, 2)

    # U1(0) block should be empty (no stored data)
    empty_block = a_merged[Block(1, 2)]
    @test size(empty_block) == (1, 1)
    @test all(iszero, empty_block.data)
end

@testset "matricize 2D AbelianArray" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianArray{Float64}(undef, g1, g2)

    block_11 = randn!(Matrix{Float64}(undef, 2, 1))
    block_22 = randn!(Matrix{Float64}(undef, 3, 2))
    a[Block(1, 1)] = SectorArray((U1(0), U1(0)), block_11)
    a[Block(2, 2)] = SectorArray((U1(1), U1(-1)), block_22)

    m = matricize(a, (1,), (2,))
    @test m isa AbelianArray{Float64, 2}

    # Row axis should be sorted: [U1(0), U1(1)]
    @test sectors(m.axes[1]) == [U1(0), U1(1)]
    @test sector_multiplicities(m.axes[1]) == [2, 3]

    # Column axis should be sorted: [U1(-1), U1(0)] (non-dual, since g2 is non-dual)
    @test sectors(m.axes[2]) == [U1(-1), U1(0)]
    @test sector_multiplicities(m.axes[2]) == [2, 1]

    # Data should be preserved in zero-flux blocks
    # U1(0) row × U1(0) col → flux 0
    # U1(1) row × U1(-1) col → flux 0
    @test haskey(m.blockdata, (1, 2))  # U1(0) × U1(0)
    @test haskey(m.blockdata, (2, 1))  # U1(1) × U1(-1)
    @test m.blockdata[(1, 2)] ≈ block_11
    @test m.blockdata[(2, 1)] ≈ block_22
end

@testset "matricize 4D AbelianArray" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    a = AbelianArray{Float64}(undef, g, g, dual(g), dual(g))

    a[Block(1, 1, 1, 1)] = SectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), ones(1, 1, 1, 1)
    )
    a[Block(2, 2, 2, 2)] = SectorArray(
        (U1(1), U1(1), dual(U1(1)), dual(U1(1))), 2 * ones(1, 1, 1, 1)
    )

    m = matricize(a, (1, 2), (3, 4))
    @test m isa AbelianArray{Float64, 2}

    # Fused codomain: [U1(0)=>1, U1(1)=>2, U1(2)=>1]
    @test sectors(m.axes[1]) == [U1(0), U1(1), U1(2)]
    @test sector_multiplicities(m.axes[1]) == [1, 2, 1]

    # All stored blocks should have zero flux
    for bk in keys(m.blockdata)
        row_s = sectors(m.axes[1])[bk[1]]
        col_s = sectors(m.axes[2])[bk[2]]
        flux = row_s ⊗ flip(col_s)
        @test GradedArrays.istrivial(flux)
    end

    # Check data values: block (1,1) and (3,3) should be on the diagonal
    @test m.blockdata[(1, 1)] ≈ ones(1, 1)
    @test m.blockdata[(3, 3)] ≈ 2 * ones(1, 1)
end

@testset "FusedSectorMatrix from 2D matricized AbelianArray" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianArray{Float64}(undef, g1, g2)

    block_11 = ones(2, 1)
    block_22 = 2 * ones(3, 2)
    a[Block(1, 1)] = SectorArray((U1(0), U1(0)), block_11)
    a[Block(2, 2)] = SectorArray((U1(1), U1(-1)), block_22)

    m = matricize(a, (1,), (2,))
    fsm = FusedSectorMatrix(m)

    @test fsm isa FusedSectorMatrix{Float64}
    @test fsm.sectors == [label(U1(0)), label(U1(1))]
    @test length(fsm.blocks) == 2
    @test fsm.blocks[1] ≈ block_11
    @test fsm.blocks[2] ≈ block_22
end

@testset "FusedSectorMatrix from 4D matricized AbelianArray" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    a = AbelianArray{Float64}(undef, g, g, dual(g), dual(g))

    a[Block(1, 1, 1, 1)] = SectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), ones(1, 1, 1, 1)
    )
    a[Block(2, 2, 2, 2)] = SectorArray(
        (U1(1), U1(1), dual(U1(1)), dual(U1(1))), 2 * ones(1, 1, 1, 1)
    )

    m = matricize(a, (1, 2), (3, 4))
    fsm = FusedSectorMatrix(m)

    @test fsm isa FusedSectorMatrix{Float64}
    @test fsm.sectors == [label(U1(0)), label(U1(1)), label(U1(2))]
    @test length(fsm.blocks) == 3

    # U1(0): 1×1 block with data
    @test fsm.blocks[1] ≈ ones(1, 1)
    # U1(1): 2×2 zeros (no stored data for this sector)
    @test fsm.blocks[2] ≈ zeros(2, 2)
    # U1(2): 1×1 block with data
    @test fsm.blocks[3] ≈ 2 * ones(1, 1)
end

@testset "contract 2D AbelianArray (matrix-matrix)" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianArray{Float64}(undef, g, dual(g))
    b = AbelianArray{Float64}(undef, g, dual(g))

    a_11 = randn!(Matrix{Float64}(undef, 2, 2))
    a_22 = randn!(Matrix{Float64}(undef, 3, 3))
    a[Block(1, 1)] = SectorArray((U1(0), dual(U1(0))), a_11)
    a[Block(2, 2)] = SectorArray((U1(1), dual(U1(1))), a_22)

    b_11 = randn!(Matrix{Float64}(undef, 2, 2))
    b_22 = randn!(Matrix{Float64}(undef, 3, 3))
    b[Block(1, 1)] = SectorArray((U1(0), dual(U1(0))), b_11)
    b[Block(2, 2)] = SectorArray((U1(1), dual(U1(1))), b_22)

    result, dimnames = contract(a, (1, -1), b, (-1, 2))
    @test result isa AbelianArray{Float64, 2}
    @test result[Block(1, 1)].data ≈ a_11 * b_11
    @test result[Block(2, 2)].data ≈ a_22 * b_22
end

@testset "contract 4D AbelianArray" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianArray{Float64}(undef, g, g, dual(g), dual(g))
    b = AbelianArray{Float64}(undef, g, g, dual(g), dual(g))

    a_data = randn!(Array{Float64}(undef, 2, 2, 2, 2))
    a[Block(1, 1, 1, 1)] = SectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), a_data
    )
    b_data = randn!(Array{Float64}(undef, 2, 2, 2, 2))
    b[Block(1, 1, 1, 1)] = SectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), b_data
    )

    # Contract: a[1, -1, 2, -2] * b[2, -3, 1, -4]
    # This permutes + contracts
    result, dimnames = contract(a, (1, -1, 2, -2), b, (2, -3, 1, -4))
    @test result isa AbelianArray

    # Verify against dense
    a_dense = zeros(5, 5, 5, 5)
    a_dense[1:2, 1:2, 1:2, 1:2] = a_data
    b_dense = zeros(5, 5, 5, 5)
    b_dense[1:2, 1:2, 1:2, 1:2] = b_data
    result_dense, _ = contract(a_dense, (1, -1, 2, -2), b_dense, (2, -3, 1, -4))
    @test size(result) == size(result_dense)
end
