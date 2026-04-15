import GradedArrays
using BlockArrays: Block, blocklength
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: AbelianGradedArray, AbelianGradedMatrix, AbelianSectorArray,
    AbelianSectorDelta, FusedGradedMatrix, GradedOneTo, SectorMatrix, SectorOneTo,
    SectorRange, U1, data, datalengths, dual, flip, gradedrange, isdual, sector, sectoraxes,
    sectormergesort, sectors, sectortype, tensor_product
using Random: randn!
using TensorAlgebra:
    TensorAlgebra, FusionStyle, contract, linearbroadcasted, matricize, unmatricize
using Test: @test, @test_throws, @testset

@testset "AbelianSectorArray linear broadcasting" begin
    s = AbelianSectorArray(
        (U1(0), dual(U1(0))),
        randn!(Matrix{ComplexF64}(undef, 2, 2))
    )
    t = AbelianSectorArray(
        (U1(0), dual(U1(0))),
        randn!(Matrix{ComplexF64}(undef, 2, 2))
    )
    @test s isa AbelianSectorArray
    @test t isa AbelianSectorArray

    α = 2.0
    β = -3.0

    st = α .* s .+ β .* t
    @test st isa AbelianSectorArray
    @test data(st) isa Matrix
    @test Array(st) ≈ α .* Array(s) .+ β .* Array(t)
    @test axes(st) == axes(s)

    conjdiff = conj.(s) .- t ./ β
    @test conjdiff isa AbelianSectorArray
    @test data(conjdiff) isa Matrix
    @test Array(conjdiff) ≈ conj.(Array(s)) .- Array(t) ./ β
    @test axes(conjdiff) == axes(s)

    @test_throws ArgumentError s .* t
    @test_throws ArgumentError exp.(s)
end

@testset "AbelianSectorArray scalar multiplication materializes on broadcast" begin
    s = AbelianSectorArray(
        (U1(0), dual(U1(0))),
        randn!(Matrix{Float64}(undef, 2, 2))
    )

    materialized = 2 .* s
    @test materialized isa AbelianSectorArray
    @test data(materialized) isa Matrix
    @test materialized[1, 1] == 2 * s[1, 1]
    @test Array(materialized) ≈ 2 .* Array(s)

    scaled_mul = 2 * s
    @test scaled_mul isa AbelianSectorArray
    @test data(scaled_mul) isa Matrix
    @test scaled_mul[1, 1] == 2 * s[1, 1]
    @test Array(scaled_mul) ≈ 2 .* Array(s)
end

@testset "AbelianSectorArray permutedims (bosonic)" begin
    data = randn!(Matrix{Float64}(undef, 3, 2))
    s = AbelianSectorArray((U1(0), dual(U1(1))), data)
    sp = permutedims(s, (2, 1))
    @test sp isa AbelianSectorArray
    @test sectoraxes(sp, 1) == dual(U1(1))
    @test sectoraxes(sp, 2) == U1(0)
    @test Array(sp) ≈ permutedims(data)
end

@testset "AbelianSectorArray permutedims (3D bosonic)" begin
    data = randn!(Array{Float64}(undef, 2, 3, 4))
    s = AbelianSectorArray((U1(0), U1(1), U1(2)), data)
    sp = permutedims(s, (3, 1, 2))
    @test sp isa AbelianSectorArray
    @test sectoraxes(sp, 1) == U1(2)
    @test sectoraxes(sp, 2) == U1(0)
    @test sectoraxes(sp, 3) == U1(1)
    @test Array(sp) ≈ permutedims(data, (3, 1, 2))
end

@testset "AbelianGradedArray permutedims" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianGradedArray{Float64}(undef, g1, g2)

    # Set allowed block (2,2): U1(1) × U1(-1) = 0
    block_data = randn!(Matrix{Float64}(undef, 3, 2))
    a[Block(2, 2)] = AbelianSectorArray((U1(1), U1(-1)), block_data)

    ap = permutedims(a, (2, 1))
    @test ap isa AbelianGradedArray
    @test axes(ap, 1) == g2
    @test axes(ap, 2) == g1

    # The block (2,2) in a should map to block (2,2) in ap
    ap_block = ap[Block(2, 2)]
    @test Array(ap_block) ≈ permutedims(block_data)
end

@testset "AbelianGradedArray linear broadcasting" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = zeros(Float64, g1, g2)
    b = zeros(Float64, g1, g2)

    # Use allowed block (2,2): U1(1) × U1(-1) = 0
    block_a = randn!(Matrix{Float64}(undef, 3, 2))
    block_b = randn!(Matrix{Float64}(undef, 3, 2))
    a[Block(2, 2)] = AbelianSectorArray((U1(1), U1(-1)), block_a)
    b[Block(2, 2)] = AbelianSectorArray((U1(1), U1(-1)), block_b)

    α = 2.0
    β = -3.0
    c = α .* a .+ β .* b
    @test c isa AbelianGradedArray
    c_block = c[Block(2, 2)]
    @test Array(c_block) ≈ α .* block_a .+ β .* block_b
end

@testset "sectormergesort on AbelianGradedArray" begin
    # Axis with repeated sectors: U1(1) appears at blocks 1 and 3
    g1 = gradedrange([U1(1) => 2, U1(0) => 1, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = zeros(Float64, g1, g2)

    a[Block(1, 2)] = AbelianSectorArray((U1(1), U1(-1)), ones(2, 2))
    a[Block(3, 2)] = AbelianSectorArray((U1(1), U1(-1)), 2 * ones(3, 2))

    a_merged = sectormergesort(a)

    # Sectors should be sorted and unique after merge
    @test sectors(axes(a_merged, 1)) == [U1(0), U1(1)]
    @test datalengths(axes(a_merged, 1)) == [1, 5]
    @test sectors(axes(a_merged, 2)) == [U1(-1), U1(0)]
    @test datalengths(axes(a_merged, 2)) == [2, 1]

    # The merged U1(1) block should stack the two source blocks (2×2 + 3×2 → 5×2)
    merged_block = a_merged[Block(2, 1)]
    @test size(merged_block) == (5, 2)
    @test data(merged_block)[1:2, :] ≈ ones(2, 2)
    @test data(merged_block)[3:5, :] ≈ 2 * ones(3, 2)

    # U1(0) block should be empty (no stored data)
    empty_block = a_merged[Block(1, 2)]
    @test size(empty_block) == (1, 1)
    @test all(iszero, data(empty_block))
end

@testset "matricize 2D AbelianGradedArray → FusedGradedMatrix" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianGradedArray{Float64}(undef, g1, g2)

    block_11 = randn!(Matrix{Float64}(undef, 2, 1))
    block_22 = randn!(Matrix{Float64}(undef, 3, 2))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), U1(0)), block_11)
    a[Block(2, 2)] = AbelianSectorArray((U1(1), U1(-1)), block_22)

    fsm = matricize(a, (1,), (2,))
    @test fsm isa FusedGradedMatrix{Float64}
    @test fsm.sectors == [U1(0), U1(1)]
    @test blocklength(fsm) == 2
    @test data(fsm[Block(1, 1)]) ≈ block_11
    @test data(fsm[Block(2, 2)]) ≈ block_22

    a_matrix = AbelianGradedArray(fsm)
    @test a_matrix isa AbelianGradedMatrix
    @test sectors(axes(a_matrix, 1)) == [U1(0), U1(1)]
    @test sectors(axes(a_matrix, 1)) == dual.(sectors(axes(a_matrix, 2)))
end

@testset "sectormergesort on reshaped AbelianGradedMatrix preserves canonical pairing" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianGradedArray{Float64}(undef, g1, g2)

    block_11 = randn!(Matrix{Float64}(undef, 2, 1))
    block_22 = randn!(Matrix{Float64}(undef, 3, 2))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), U1(0)), block_11)
    a[Block(2, 2)] = AbelianSectorArray((U1(1), U1(-1)), block_22)

    a_reshaped = matricize(GradedArrays.BlockReshapeFusion(), a, Val(1))
    a_merged = sectormergesort(a_reshaped)

    @test sectors(axes(a_merged, 1)) == [U1(0), U1(1)]
    @test sectors(axes(a_merged, 1)) == dual.(sectors(axes(a_merged, 2)))
    @test collect(eachblockstoredindex(a_merged)) == [Block(1, 1), Block(2, 2)]
    @test Array(a_merged[Block(1, 1)]) ≈ block_11
    @test Array(a_merged[Block(2, 2)]) ≈ block_22
end

@testset "matricize 4D AbelianGradedArray → FusedGradedMatrix" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    a = zeros(Float64, g, g, dual(g), dual(g))

    a[Block(1, 1, 1, 1)] = AbelianSectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), ones(1, 1, 1, 1)
    )
    a[Block(2, 2, 2, 2)] = AbelianSectorArray(
        (U1(1), U1(1), dual(U1(1)), dual(U1(1))), 2 * ones(1, 1, 1, 1)
    )

    fsm = matricize(a, (1, 2), (3, 4))
    @test fsm isa FusedGradedMatrix{Float64}
    @test fsm.sectors == [U1(0), U1(1), U1(2)]
    @test blocklength(fsm) == 3

    @test data(fsm[Block(1, 1)]) ≈ ones(1, 1)
    @test data(fsm[Block(2, 2)]) ≈ zeros(2, 2)
    @test data(fsm[Block(3, 3)]) ≈ 2 * ones(1, 1)
end

@testset "FusedGradedMatrix(::AbelianGradedMatrix) requires canonical fused axes" begin
    row_ax = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, row_ax, dual(row_ax))
    block_11 = randn!(Matrix{Float64}(undef, 2, 2))
    block_22 = randn!(Matrix{Float64}(undef, 3, 3))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), block_11)
    a[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), block_22)

    fsm = FusedGradedMatrix(a)
    @test data(fsm[Block(1, 1)]) ≈ block_11
    @test data(fsm[Block(2, 2)]) ≈ block_22

    nonsorted_ax = gradedrange([U1(1) => 3, U1(0) => 2])
    a_nonsorted = AbelianGradedArray{Float64}(undef, nonsorted_ax, dual(nonsorted_ax))
    a_nonsorted[Block(1, 1)] = AbelianSectorArray((U1(1), dual(U1(1))), block_22)
    a_nonsorted[Block(2, 2)] = AbelianSectorArray((U1(0), dual(U1(0))), block_11)
    @test_throws ArgumentError FusedGradedMatrix(a_nonsorted)

    mismatched_col_ax = gradedrange([U1(-1) => 3, U1(0) => 2])
    a_mismatched = AbelianGradedArray{Float64}(undef, row_ax, mismatched_col_ax)
    @test_throws ArgumentError FusedGradedMatrix(a_mismatched)
end

@testset "Off-diagonal block setindex! errors" begin
    ax = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, ax, dual(ax))
    @test_throws ErrorException (
        a[Block(1, 2)] =
            AbelianSectorArray((U1(0), dual(U1(1))), randn!(Matrix{Float64}(undef, 2, 3)))
    )
end

@testset "contract 2D AbelianGradedArray (matrix-matrix)" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, g, dual(g))
    b = AbelianGradedArray{Float64}(undef, g, dual(g))

    a_11 = randn!(Matrix{Float64}(undef, 2, 2))
    a_22 = randn!(Matrix{Float64}(undef, 3, 3))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), a_11)
    a[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), a_22)

    b_11 = randn!(Matrix{Float64}(undef, 2, 2))
    b_22 = randn!(Matrix{Float64}(undef, 3, 3))
    b[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), b_11)
    b[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), b_22)

    result, dimnames = contract(a, (1, -1), b, (-1, 2))
    @test result isa AbelianGradedArray{Float64, 2}
    @test data(result[Block(1, 1)]) ≈ a_11 * b_11
    @test data(result[Block(2, 2)]) ≈ a_22 * b_22
end

@testset "unmatricize AbelianSectorMatrix with SectorOneTo axes" begin
    # Create a 3D AbelianSectorArray, matricize it, then unmatricize and verify roundtrip
    codomain_ax = SectorOneTo(U1(0), 2)
    domain_ax1 = SectorOneTo(U1(0)', 3)
    domain_ax2 = SectorOneTo(U1(1)', 4)

    data_3d = randn!(Array{Float64}(undef, 2, 3, 4))
    s = AbelianSectorArray(
        (sector(codomain_ax), sector(domain_ax1), sector(domain_ax2)),
        data_3d
    )

    # Matricize with 1 codomain leg
    sm = matricize(s, Val(1))
    @test sm isa SectorMatrix
    @test ndims(sm) == 2

    # Unmatricize back to 3D
    s_back = unmatricize(sm, (codomain_ax,), (domain_ax1, domain_ax2))
    @test s_back isa AbelianSectorArray
    @test ndims(s_back) == 3
    @test size(s_back) == size(s)

    # For bosonic (U1) sectors, no fermionic phase, data should match
    @test Array(s_back) ≈ data_3d
end

@testset "contract 4D AbelianGradedArray" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, g, g, dual(g), dual(g))
    b = AbelianGradedArray{Float64}(undef, g, g, dual(g), dual(g))

    a_data = randn!(Array{Float64}(undef, 2, 2, 2, 2))
    a[Block(1, 1, 1, 1)] = AbelianSectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), a_data
    )
    b_data = randn!(Array{Float64}(undef, 2, 2, 2, 2))
    b[Block(1, 1, 1, 1)] = AbelianSectorArray(
        (U1(0), U1(0), dual(U1(0)), dual(U1(0))), b_data
    )

    # Contract: a[1, -1, 2, -2] * b[2, -3, 1, -4]
    # This permutes + contracts
    result, dimnames = contract(a, (1, -1, 2, -2), b, (2, -3, 1, -4))
    @test result isa AbelianGradedArray

    # Verify against dense
    a_dense = zeros(5, 5, 5, 5)
    a_dense[1:2, 1:2, 1:2, 1:2] = a_data
    b_dense = zeros(5, 5, 5, 5)
    b_dense[1:2, 1:2, 1:2, 1:2] = b_data
    result_dense, _ = contract(a_dense, (1, -1, 2, -2), b_dense, (2, -3, 1, -4))
    @test size(result) == size(result_dense)
end

@testset "scale! with β=0 zeros uninitialized blocks" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, g, dual(g))
    GradedArrays.scale!(a, false)
    @test all(iszero, data(a[Block(1, 1)]))
    @test all(iszero, data(a[Block(2, 2)]))
end

@testset "allocating broadcast produces correct results" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(Float64, (g, dual(g)))
    a[Block(1, 1)] = [1.0 0.0; 0.0 1.0]
    a[Block(2, 2)] = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

    b = 3 .* a
    @test data(b[Block(1, 1)]) == [3.0 0.0; 0.0 3.0]
    @test data(b[Block(2, 2)]) == [3.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 3.0]

    c = a - a
    @test all(iszero, data(c[Block(1, 1)]))
    @test all(iszero, data(c[Block(2, 2)]))
end

@testset "FusedGradedMatrix broadcasting" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [[1.0 2.0; 3.0 4.0], [5.0 0.0; 0.0 6.0]])

    m2 = 3 * m
    @test data(m2[Block(1, 1)]) == [3.0 6.0; 9.0 12.0]
    @test data(m2[Block(2, 2)]) == [15.0 0.0; 0.0 18.0]

    n = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), ones(2, 2)])
    s = m + n
    @test data(s[Block(1, 1)]) == [2.0 3.0; 4.0 5.0]
    @test data(s[Block(2, 2)]) == [6.0 1.0; 1.0 7.0]

    c = similar(m, Float64)
    c .= 3 .* m .+ 2 .* n
    @test data(c[Block(1, 1)]) == [5.0 8.0; 11.0 14.0]
    @test data(c[Block(2, 2)]) == [17.0 2.0; 2.0 20.0]
end
