using BlockArrays: Block, blocksize
using BlockSparseArrays: BlockSparseArray, mortar_axis
using GradedArrays: GradedArray, GradedMatrix, SU2, SectorArray, SectorDelta, U1, dual,
    flip, gradedrange, isdual, sector, sector_type, sectorrange, trivial,
    trivial_gradedrange, ⊗
using Random: randn!
using TensorAlgebra: TensorAlgebra, FusionStyle, contract,
    matricize, tensor_product_axis, trivial_axis, unmatricize
using Test: @test, @test_throws, @testset

function randn_blockdiagonal(elt::Type, axes::Tuple)
    a = BlockSparseArray{elt}(undef, axes)
    blockdiaglength = minimum(blocksize(a))
    for i in 1:blockdiaglength
        b = Block(ntuple(Returns(i), ndims(a)))
        a[b] = randn!(a[b])
    end
    return a
end

@testset "trivial_axis" begin
    g1 = gradedrange([U1(1) => 1, U1(2) => 1])
    g2 = gradedrange([U1(-1) => 2, U1(2) => 1])
    @test trivial_gradedrange((g1, g2)) == gradedrange([U1(0) => 1])
    @test trivial_gradedrange(sector_type(g1)) == gradedrange([U1(0) => 1])

    gN = gradedrange([(; N = U1(1)) => 1])
    gS = gradedrange([(; S = SU2(1 // 2)) => 1])
    gNS = gradedrange([(; N = U1(0), S = SU2(0)) => 1])
    @test trivial_gradedrange(sector_type(gN)) == gradedrange([(; N = U1(0)) => 1])
    @test trivial_gradedrange((gN, gS)) == gNS
end

@testset "SectorDelta domain axis tensor product uses flip" begin
    r1 = sectorrange(U1(1), 2)
    r2 = sectorrange(U1(2), 3)
    rdomain = tensor_product_axis(FusionStyle(SectorDelta), Val(:domain), r1, r2)
    @test rdomain == flip(r1 ⊗ r2)
    @test isdual(rdomain)
    @test sector(rdomain) == flip(U1(3))
end

@testset "SectorDelta matricize handles empty codomain/domain" begin
    s1 = U1(1)
    s2 = U1(2)
    s3 = U1(-1)
    d = SectorDelta{Float64}((s1, s2, s3))

    m_left_empty = matricize(d, (), (1, 2, 3))
    @test axes(m_left_empty, 1) == trivial(sector_type(d))
    @test axes(m_left_empty, 2) == flip(s1 ⊗ s2 ⊗ s3)

    m_right_empty = matricize(d, (1, 2, 3), ())
    @test axes(m_right_empty, 1) == s1 ⊗ s2 ⊗ s3
    @test axes(m_right_empty, 2) == trivial(sector_type(d))
end

@testset "SectorDelta unmatricize preserves provided axes" begin
    m = SectorDelta{Float64}((U1(3), flip(U1(3))))
    @test unmatricize(m, (U1(1), U1(2)), (U1(-2), U1(-1))) ==
        SectorDelta{Float64}((U1(1), U1(2), U1(-2), U1(-1)))
    @test unmatricize(m, (U1(1), U1(1)), (U1(-2), U1(-1))) isa SectorDelta
end

@testset "SectorArray linear broadcasting" begin
    s = SectorArray((U1(0), dual(U1(0))), randn!(Matrix{ComplexF64}(undef, 2, 2)))
    t = SectorArray((U1(0), dual(U1(0))), randn!(Matrix{ComplexF64}(undef, 2, 2)))
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
    s = SectorArray((U1(0), dual(U1(0))), randn!(Matrix{Float64}(undef, 2, 2)))

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

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "`contract` `GradedArray` (eltype=$elt)" for elt in elts
    @testset "matricize" begin
        d1 = gradedrange([U1(0) => 1, U1(1) => 1])
        d2 = gradedrange([U1(0) => 1, U1(1) => 1])
        a = randn_blockdiagonal(elt, (d1, d2, dual(d1), dual(d2)))
        m = matricize(a, (1, 2), (3, 4))
        @test m isa GradedMatrix
        @test axes(m, 1) == gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 1])
        @test axes(m, 2) == flip(gradedrange([U1(0) => 1, U1(-1) => 2, U1(-2) => 1]))

        for I in CartesianIndices(m)
            if I ∈ CartesianIndex.([(1, 1), (4, 4)])
                @test !iszero(m[I])
            else
                @test iszero(m[I])
            end
        end
        @test a[1, 1, 1, 1] == m[1, 1]
        @test a[2, 2, 2, 2] == m[4, 4]
        @test blocksize(m) == (3, 3)
        @test a == unmatricize(m, (d1, d2), (dual(d1), dual(d2)))

        # check block fusing and splitting
        d = gradedrange([U1(0) => 2, U1(1) => 1])
        b = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
        @test unmatricize(
            matricize(b, (1, 2), (3, 4)), (axes(b, 1), axes(b, 2)),
            (axes(b, 3), axes(b, 4))
        ) == b

        d1234 =
            gradedrange([U1(-2) => 1, U1(-1) => 4, U1(0) => 6, U1(1) => 4, U1(2) => 1])
        m = matricize(a, (1, 2, 3, 4), ())
        @test m isa GradedMatrix
        @test axes(m, 1) == d1234
        @test axes(m, 2) == flip(gradedrange([U1(0) => 1]))
        @test a == unmatricize(m, (d1, d2, dual(d1), dual(d2)), ())

        m = matricize(a, (), (1, 2, 3, 4))
        @test m isa GradedMatrix
        @test axes(m, 1) == gradedrange([U1(0) => 1])
        @test axes(m, 2) == dual(d1234)
        @test a == unmatricize(m, (), (d1, d2, dual(d1), dual(d2)))
    end

    @testset "contract with U(1)" begin
        d = gradedrange([U1(0) => 2, U1(1) => 3])
        a1 = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
        a2 = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
        a3 = randn_blockdiagonal(elt, (d, dual(d)))
        a1_dense = convert(Array, a1)
        a2_dense = convert(Array, a2)
        a3_dense = convert(Array, a3)

        # matrix matrix
        a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
        a_dest_dense, dimnames_dest_dense = contract(
            a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
        )
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa GradedArray
        @test Array(a_dest) ≈ a_dest_dense

        # matrix vector
        a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
        a_dest_dense, dimnames_dest_dense =
            contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa GradedArray
        @test Array(a_dest) ≈ a_dest_dense

        # vector matrix
        a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
        a_dest_dense, dimnames_dest_dense =
            contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa GradedArray
        @test Array(a_dest) ≈ a_dest_dense

        # vector vector
        a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
        a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa BlockSparseArray{elt, 0}
        @test Array(a_dest) ≈ a_dest_dense

        # outer product
        a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
        a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
        @test dimnames_dest == dimnames_dest_dense
        @test size(a_dest) == size(a_dest_dense)
        @test a_dest isa GradedArray
        @test Array(a_dest) ≈ a_dest_dense
    end
end
