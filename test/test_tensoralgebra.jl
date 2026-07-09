import GradedArrays
using BlockArrays: Block, blocklength
using GradedArrays: AbelianGradedArray, AbelianGradedMatrix, AbelianSectorArray,
    AbelianSectorDelta, FusedGradedMatrix, GradedOneTo, SectorMatrix, SectorOneTo,
    SectorRange, U1, data, datalengths, dual, eachblockstoredindex, eachsectoraxis, flip,
    gradedrange, isdual, sector, sectoraxes, sectormergesort, sectors, sectortype,
    tensor_product
using LinearAlgebra: tr
using Random: randn!
using TensorAlgebra: TensorAlgebra, FusionStyle, contract, linearbroadcasted, matricize,
    matricizeperm, unmatricize
using Test: @test, @test_broken, @test_throws, @testset

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

    # `conj.` lowers each operand to a `ConjArray` whose axes are dualized, so a
    # fully-conjugated broadcast lines up and matches the eager result (bosonic here, so no
    # fermion sign).
    cst = conj.(s) .- conj.(t) ./ β
    @test cst isa AbelianSectorArray
    @test Array(cst) ≈ conj.(Array(s)) .- conj.(Array(t)) ./ β
    @test sectoraxes(cst) == sectoraxes(conj(s))
    @test Array(conj.(s)) ≈ conj(Array(s))

    # Conjugating only some operands leaves dualized axes against non-dual ones: rejected.
    @test_throws DimensionMismatch conj.(s) .- t

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

    fsm = matricizeperm(a, (1,), (2,))
    @test fsm isa FusedGradedMatrix{Float64}
    @test collect(keys(fsm.blocks)) == [U1(0), U1(1)]
    @test blocklength(fsm, 1) == 2
    @test blocklength(fsm, 2) == 2
    @test data(fsm[Block(1, 1)]) ≈ block_11
    @test data(fsm[Block(2, 2)]) ≈ block_22

    a_matrix = AbelianGradedArray(fsm)
    @test a_matrix isa AbelianGradedMatrix
    @test sectors(axes(a_matrix, 1)) == [U1(0), U1(1)]
    # Dual-resolved sectors of the row pair with the dual of the column's.
    @test eachsectoraxis(axes(a_matrix, 1)) == dual.(eachsectoraxis(axes(a_matrix, 2)))
end

@testset "matricize AbelianGradedMatrix preserves canonical sector pairing" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = AbelianGradedArray{Float64}(undef, g1, g2)

    block_11 = randn!(Matrix{Float64}(undef, 2, 1))
    block_22 = randn!(Matrix{Float64}(undef, 3, 2))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), U1(0)), block_11)
    a[Block(2, 2)] = AbelianSectorArray((U1(1), U1(-1)), block_22)

    fsm = matricizeperm(a, (1,), (2,))
    @test fsm isa FusedGradedMatrix{Float64}
    # Each stored N-D block lands in the coupled sector pairing its row charge with
    # the dual of its column charge: (U1(0), U1(0)) → U1(0), (U1(1), U1(-1)) → U1(1).
    @test collect(keys(fsm.blocks)) == [U1(0), U1(1)]
    @test data(fsm[Block(1, 1)]) ≈ block_11
    @test data(fsm[Block(2, 2)]) ≈ block_22
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

    fsm = matricizeperm(a, (1, 2), (3, 4))
    @test fsm isa FusedGradedMatrix{Float64}
    @test collect(keys(fsm.blocks)) == [U1(0), U1(1), U1(2)]
    @test blocklength(fsm, 1) == 3
    @test blocklength(fsm, 2) == 3

    @test data(fsm[Block(1, 1)]) ≈ ones(1, 1)
    @test data(fsm[Block(2, 2)]) ≈ zeros(2, 2)
    @test data(fsm[Block(3, 3)]) ≈ 2 * ones(1, 1)
end

@testset "tr of a matricized AbelianGradedArray" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    a = ones(Float64, g, g, dual(g), dual(g))
    a[Block(2, 2, 2, 2)] .*= 2  # give the blocks distinct traces

    # `tr` on the matricized graded matrix sums the diagonal blocks and matches the dense trace.
    fsm = matricizeperm(a, (1, 2), (3, 4))
    @test tr(fsm) ≈ tr(Array(fsm))
    # `TensorAlgebra.tr` over the (1, 2) | (3, 4) bipartition routes through the same path.
    @test TensorAlgebra.tr(a, (1, 2, 3, 4), (1, 2), (3, 4)) ≈ tr(Array(fsm))
end

@testset "matricize 3D AbelianGradedArray and unmatricize round-trip" begin
    # 3D case where the merged codomain (tensor product of two `r`s) has
    # sectors absent from the domain — the asymmetric `FusedGradedMatrix`
    # natively handles this (codomain has U1(2), domain has only U1(0) and
    # U1(1)).
    r = gradedrange([U1(0) => 1, U1(1) => 2])
    a = zeros(Float64, (r, r, dual(r)))
    a[Block(1, 1, 1)] = fill(1.0, 1, 1, 1)
    a[Block(1, 2, 2)] = fill(2.0, 1, 2, 2)
    a[Block(2, 1, 2)] = fill(3.0, 2, 1, 2)

    fsm = matricizeperm(a, (1, 2), (3,))
    @test fsm isa FusedGradedMatrix{Float64}
    # Codomain carries all three sectors, domain only the two that exist on
    # the contracted leg — the new asymmetric design.
    @test collect(keys(fsm.codomain)) == [U1(0), U1(1), U1(2)]
    @test collect(keys(fsm.domain)) == [U1(0), U1(1)]
    @test collect(keys(fsm.blocks)) == [U1(0), U1(1)]

    # Round-trip through `unmatricize` recovers the original blocks. The domain axes are
    # passed codomain-facing (un-dualized), so the original `dual(r)` domain axis is given
    # as `r`.
    a_back = unmatricize(fsm, (r, r), (r,))
    @test a_back isa AbelianGradedArray
    @test ndims(a_back) == 3
    for I in eachblockstoredindex(a)
        @test data(a[I]) ≈ data(a_back[I])
    end
end

@testset "FusedGradedMatrix(::AbelianGradedMatrix) accepts asymmetric axes" begin
    row_ax = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, row_ax, dual(row_ax))
    block_11 = randn!(Matrix{Float64}(undef, 2, 2))
    block_22 = randn!(Matrix{Float64}(undef, 3, 3))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), block_11)
    a[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), block_22)

    fsm = FusedGradedMatrix(a)
    @test data(fsm[Block(1, 1)]) ≈ block_11
    @test data(fsm[Block(2, 2)]) ≈ block_22

    # Unsorted axes still rejected (they violate the sorted-keys invariant).
    nonsorted_ax = gradedrange([U1(1) => 3, U1(0) => 2])
    a_nonsorted = AbelianGradedArray{Float64}(undef, nonsorted_ax, dual(nonsorted_ax))
    a_nonsorted[Block(1, 1)] = AbelianSectorArray((U1(1), dual(U1(1))), block_22)
    a_nonsorted[Block(2, 2)] = AbelianSectorArray((U1(0), dual(U1(0))), block_11)
    @test_throws ArgumentError FusedGradedMatrix(a_nonsorted)

    # Asymmetric axes (codomain and dual(domain) sector sets differ) are
    # now accepted: cod-only and dom-only sectors land as one-sided blocks
    # of size 0 on the absent axis (no stored block).
    asym_col_ax = dual(gradedrange([U1(2) => 3, U1(3) => 2]))
    a_asym = AbelianGradedArray{Float64}(undef, row_ax, asym_col_ax)
    fsm_asym = FusedGradedMatrix(a_asym)
    @test fsm_asym isa FusedGradedMatrix
    @test collect(keys(fsm_asym.codomain)) == [U1(0), U1(1)]
    @test collect(keys(fsm_asym.domain)) == [U1(2), U1(3)]
    # No stored blocks since we didn't write any data and the sectors don't overlap.
    @test isempty(fsm_asym.blocks)
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
    @test result isa AbelianGradedArray{Float64, <:Any, 2}
    @test data(result[Block(1, 1)]) ≈ a_11 * b_11
    @test data(result[Block(2, 2)]) ≈ a_22 * b_22
end

@testset "contract AbelianGradedArray to a scalar (elt=$elt)" for elt in
    (Float64, ComplexF64)
    # A full contraction over every index collapses to a rank-0 result. The
    # destination is allocated as a rank-0 graded array (trivial sector), so the
    # whole matricize/mul!/unmatricize path stays in graded land; the result reads
    # back as a scalar via `result[]`.
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn(elt, (g, dual(g)))
    b = randn(elt, (dual(g), g))

    result = contract((), a, (1, 2), b, (1, 2))
    @test result isa AbelianGradedArray{elt, <:Any, 0}
    @test ndims(result) == 0
    @test sectortype(result) === U1
    @test result[] ≈ sum(Array(a) .* Array(b))
end

@testset "matricize/unmatricize a rank-0 graded array" begin
    # The rank-0 limit of the matricize path, exercised directly. With no axes, the
    # codomain/domain groups fuse to the trivial sector, so the unmerged axes are a
    # single trivial block (the sector type is supplied explicitly).
    row, col = GradedArrays.unmerged_matricize_axes(U1, (), ())
    @test sectors(row) == [U1(0)]
    @test sectors(col) == [U1(0)]
    @test isdual(col)

    # A rank-0 graded array matricizes to a 1×1 trivial-sector `FusedGradedMatrix`,
    # and unmatricizing back recovers the scalar as a rank-0 graded array.
    a = AbelianGradedArray{Float64, U1, 0, Array{Float64, 0}}(undef, ())
    a[] = 4.0
    m = matricize(GradedArrays.SectorFusion(), a, Val(0))
    @test m isa FusedGradedMatrix{Float64}
    @test size(m) == (1, 1)
    @test data(m[Block(1, 1)]) == fill(4.0, 1, 1)

    back = unmatricize(GradedArrays.SectorFusion(), m, (), ())
    @test back isa AbelianGradedArray{Float64, <:Any, 0}
    @test back[] == 4.0
end

@testset "unmatricize AbelianSectorMatrix with SectorOneTo axes" begin
    # Create a 3D AbelianSectorArray, matricize it, then unmatricize and verify roundtrip
    codomain_ax = SectorOneTo(U1(0), 2)
    domain_ax1 = SectorOneTo(conj(U1(0)), 3)
    domain_ax2 = SectorOneTo(conj(U1(1)), 4)

    data_3d = randn!(Array{Float64}(undef, 2, 3, 4))
    s = AbelianSectorArray(
        (sector(codomain_ax), sector(domain_ax1), sector(domain_ax2)),
        data_3d
    )

    # Matricize with 1 codomain leg
    sm = matricize(s, Val(1))
    @test sm isa SectorMatrix
    @test ndims(sm) == 2

    # Unmatricize back to 3D. The domain axes are passed codomain-facing (un-dualized),
    # so the stored `conj`-ed domain axes are given as their un-dualized counterparts.
    s_back = unmatricize(sm, (codomain_ax,), (conj(domain_ax1), conj(domain_ax2)))
    @test s_back isa AbelianSectorArray
    @test ndims(s_back) == 3
    @test size(s_back) == size(s)

    # For bosonic (U1) sectors, no fermionic phase, data should match
    @test Array(s_back) ≈ data_3d
end

@testset "contract 4D AbelianGradedArray" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn(g, g, dual(g), dual(g))
    b = randn(g, g, dual(g), dual(g))

    # Contract: a[1, -1, 2, -2] * b[2, -3, 1, -4] (permutes + contracts).
    result, dimnames = contract(a, (1, -1, 2, -2), b, (2, -3, 1, -4))
    @test result isa AbelianGradedArray

    # Verify numerics against the dense contraction of the same data.
    result_dense, _ = contract(Array(a), (1, -1, 2, -2), Array(b), (2, -3, 1, -4))
    @test Array(result) ≈ result_dense
end

@testset "scale! with β=0 zeros uninitialized blocks" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, g, dual(g))
    TensorAlgebra.scale!(a, false)
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

@testset "FusedGradedMatrix block-wise arithmetic" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [[1.0 2.0; 3.0 4.0], [5.0 0.0; 0.0 6.0]])

    m2 = 3 * m
    @test data(m2[Block(1, 1)]) == [3.0 6.0; 9.0 12.0]
    @test data(m2[Block(2, 2)]) == [15.0 0.0; 0.0 18.0]

    n = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), ones(2, 2)])
    s = m + n
    @test data(s[Block(1, 1)]) == [2.0 3.0; 4.0 5.0]
    @test data(s[Block(2, 2)]) == [6.0 1.0; 1.0 7.0]
end

@testset "FusedGradedMatrix broadcasting errors" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [[1.0 2.0; 3.0 4.0], [5.0 0.0; 0.0 6.0]])
    n = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), ones(2, 2)])
    @test_throws ArgumentError m .+ n
    @test_throws ArgumentError 3 .* m
    @test_throws ArgumentError conj.(m)
    c = similar(m, Float64)
    @test_throws ArgumentError c .= 3 .* m .+ 2 .* n
end

# Regression coverage for TensorAlgebra-level unmatricize-axis bugs on graded
# operators: a factor's reconstructed axes must respect the conj/dual pairing
# between contracted bonds rather than reuse the factor's own axes.
@testset "TA.svd_compact round-trip on AbelianGradedArray (axes_S regression)" begin
    s = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
    A = AbelianGradedArray{Float64}(undef, s, dual(s))
    randn!(A)
    U, S, Vᴴ = TensorAlgebra.svd_compact(A, (1,), (2,))
    US = contract((:a, :r), U, (:a, :i), S, (:i, :r))
    USV = contract((:a, :b), US, (:a, :r), Vᴴ, (:r, :b))
    @test A ≈ USV
    # `*` on `AbelianGradedMatrix` routes through the block-wise `contract`.
    @test A ≈ U * S * Vᴴ
end

@testset "TA.gram_eigh_full_with_pinv on AbelianGradedMatrix (axes_Y regression)" begin
    s = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
    B = AbelianGradedArray{Float64}(undef, s, dual(s))
    randn!(B)
    # PSD by construction. Build `A = B * B'` block-wise via `contract`
    # so we stay on the graded matmul path; the natural `*` form is broken
    # against the same scalar-indexing path as the SVD round-trip above.
    A = contract((:a, :b), B, (:a, :r), conj(B), (:b, :r))
    # `*` on two `AbelianGradedMatrix` works, but the adjoint forms (`B * B'`,
    # `X * X'` below) still need a block-aware `adjoint`; `B'` is an `Adjoint`
    # wrapper that falls through to LinearAlgebra's scalar-indexing path.
    @test_broken A ≈ B * B'
    X, Y = TensorAlgebra.gram_eigh_full_with_pinv(A, (1,), (2,))
    # X · conj(X) ≈ A on the rank subspace.
    @test A ≈ contract((:a, :b), X, (:a, :r), conj(X), (:b, :r))
    @test_broken A ≈ X * X'
    # Y is a left inverse of X on the rank subspace.
    YX = contract((:r, :s), Y, (:r, :a), X, (:a, :s))
    @test YX ≈ TensorAlgebra.one(YX, (:r, :s), (:r,), (:s,))
end

@testset "contract rejects mismatched contracted-axis duality (bosonic)" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, g, dual(g))
    randn!(a)

    # The contracted leg of `a` is `dual(g)`; here `b`'s contracted leg is also
    # `dual(g)`, which is neither the canonical dual pairing nor (for bosonic
    # U1) an accepted same-`isdual` pair, so the contraction is rejected.
    b = AbelianGradedArray{Float64}(undef, dual(g), dual(g))
    @test_throws ArgumentError contract(a, (1, -1), b, (-1, 2))

    # Sanity: the canonically dual-paired contraction is accepted.
    b_ok = AbelianGradedArray{Float64}(undef, g, dual(g))
    randn!(b_ok)
    result, = contract(a, (1, -1), b_ok, (-1, 2))
    @test result isa AbelianGradedArray
end
