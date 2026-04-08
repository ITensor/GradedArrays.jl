import GradedArrays
using BlockArrays: Block, blocklength
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: AbelianArray, GradedIndices, SectorArray, SectorDelta, SectorRange, U1,
    dual, flip, gradedrange, isdual, label, sector, sector_type, sectorrange, sectors,
    tensor_product, ⊗
using Random: randn!
using TensorAlgebra: TensorAlgebra, linearbroadcasted
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
