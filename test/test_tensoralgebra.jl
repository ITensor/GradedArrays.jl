import GradedArrays
using BlockArrays: Block, blocklength
using GradedArrays: FusedGradedMatrix, FusedGradedVector, FusedSectorMatrix, FusionArray,
    GradedOneTo, SU2, SectorOneTo, SectorOnesVector, U1, UniqueSectorArray,
    UniqueSectorDelta, data, datalengths, dual, eachblockstoredindex, eachsectoraxis, flip,
    gradedrange, isdual, sector, sectoraxes, sectormergesort, sectors, sectortype,
    tensor_product, with_block_indexing, with_scalar_indexing
using LinearAlgebra: tr
using Random: randn!
using TensorAlgebra: TensorAlgebra, MatricizeStyle, contract, linearbroadcasted, matricize,
    matricizeperm, unmatricize
using TensorKitSectors: FermionNumber
using Test: @test, @test_broken, @test_throws, @testset

@testset "UniqueSectorArray linear broadcasting" begin
    s = UniqueSectorArray(randn!(Matrix{ComplexF64}(undef, 2, 2)), (U1(0), dual(U1(0))))
    t = UniqueSectorArray(randn!(Matrix{ComplexF64}(undef, 2, 2)), (U1(0), dual(U1(0))))
    @test s isa UniqueSectorArray
    @test t isa UniqueSectorArray

    α = 2.0
    β = -3.0

    st = α .* s .+ β .* t
    @test st isa UniqueSectorArray
    @test data(st) isa Matrix
    @test Array(st) ≈ α .* Array(s) .+ β .* Array(t)
    @test axes(st) == axes(s)

    # `conj.` lowers each operand to a `ConjArray` whose axes are dualized, so a
    # fully-conjugated broadcast lines up and matches the eager result (bosonic here, so no
    # fermion sign).
    cst = conj.(s) .- conj.(t) ./ β
    @test cst isa UniqueSectorArray
    @test Array(cst) ≈ conj.(Array(s)) .- conj.(Array(t)) ./ β
    @test sectoraxes(cst) == sectoraxes(conj(s))
    @test Array(conj.(s)) ≈ conj(Array(s))

    # Conjugating only some operands leaves dualized axes against non-dual ones: rejected.
    @test_throws DimensionMismatch conj.(s) .- t

    @test_throws ArgumentError s .* t
    @test_throws ArgumentError exp.(s)
end

@testset "UniqueSectorArray scalar multiplication materializes on broadcast" begin
    s = UniqueSectorArray(randn!(Matrix{Float64}(undef, 2, 2)), (U1(0), dual(U1(0))))

    materialized = 2 .* s
    @test materialized isa UniqueSectorArray
    @test data(materialized) isa Matrix
    with_scalar_indexing() do
        @test materialized[1, 1] == 2 * s[1, 1]
    end
    @test Array(materialized) ≈ 2 .* Array(s)

    scaled_mul = 2 * s
    @test scaled_mul isa UniqueSectorArray
    @test data(scaled_mul) isa Matrix
    with_scalar_indexing() do
        @test scaled_mul[1, 1] == 2 * s[1, 1]
    end
    @test Array(scaled_mul) ≈ 2 .* Array(s)
end

@testset "UniqueSectorArray permutedims (bosonic)" begin
    data = randn!(Matrix{Float64}(undef, 3, 2))
    s = UniqueSectorArray(data, (U1(0), dual(U1(1))))
    sp = permutedims(s, (2, 1))
    @test sp isa UniqueSectorArray
    @test sectoraxes(sp, 1) == dual(U1(1))
    @test sectoraxes(sp, 2) == U1(0)
    @test Array(sp) ≈ permutedims(data)
end

@testset "UniqueSectorArray permutedims (3D bosonic)" begin
    data = randn!(Array{Float64}(undef, 2, 3, 4))
    s = UniqueSectorArray(data, (U1(0), U1(1), U1(2)))
    sp = permutedims(s, (3, 1, 2))
    @test sp isa UniqueSectorArray
    @test sectoraxes(sp, 1) == U1(2)
    @test sectoraxes(sp, 2) == U1(0)
    @test sectoraxes(sp, 3) == U1(1)
    @test Array(sp) ≈ permutedims(data, (3, 1, 2))
end

@testset "graded array permutedims" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = zeros(Float64, g1, g2)

    # Set allowed block (2,2): U1(1) × U1(-1) = 0
    block_data = randn!(Matrix{Float64}(undef, 3, 2))
    with_block_indexing() do
        return a[Block(2, 2)] = UniqueSectorArray(block_data, (U1(1), U1(-1)))
    end

    ap = permutedims(a, (2, 1))
    @test ap isa FusionArray
    @test axes(ap, 1) == g2
    @test axes(ap, 2) == g1

    # The block (2,2) in a should map to block (2,2) in ap
    with_block_indexing() do
        ap_block = ap[Block(2, 2)]
        @test Array(ap_block) ≈ permutedims(block_data)
    end
end

@testset "graded array linear broadcasting" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = zeros(Float64, g1, g2)
    b = zeros(Float64, g1, g2)

    # Use allowed block (2,2): U1(1) × U1(-1) = 0
    block_a = randn!(Matrix{Float64}(undef, 3, 2))
    block_b = randn!(Matrix{Float64}(undef, 3, 2))
    with_block_indexing() do
        a[Block(2, 2)] = UniqueSectorArray(block_a, (U1(1), U1(-1)))
        return b[Block(2, 2)] = UniqueSectorArray(block_b, (U1(1), U1(-1)))
    end

    α = 2.0
    β = -3.0
    c = α .* a .+ β .* b
    @test c isa FusionArray
    with_block_indexing() do
        c_block = c[Block(2, 2)]
        @test Array(c_block) ≈ α .* block_a .+ β .* block_b
    end
end

@testset "sectormergesort on a graded array" begin
    # `FusionArray` represents unfused (unsorted, repeated-sector) external axes directly (`U1(1)` at
    # blocks 1 and 3 here). `sectormergesort` sorts and merges them; since the fused storage is already
    # canonical, it is a pure external-axis relabel over the same data.
    g1 = gradedrange([U1(1) => 2, U1(0) => 1, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = zeros(Float64, g1, g2)

    with_block_indexing() do
        a[Block(1, 2)] = UniqueSectorArray(ones(2, 2), (U1(1), U1(-1)))
        return a[Block(3, 2)] = UniqueSectorArray(2 * ones(3, 2), (U1(1), U1(-1)))
    end

    a_merged = sectormergesort(a)

    # Sectors should be sorted and unique after merge
    @test sectors(axes(a_merged, 1)) == [U1(0), U1(1)]
    @test datalengths(axes(a_merged, 1)) == [1, 5]
    @test sectors(axes(a_merged, 2)) == [U1(0), U1(-1)]
    @test datalengths(axes(a_merged, 2)) == [1, 2]

    # The merged U1(1) block should stack the two source blocks (2×2 + 3×2 → 5×2)
    with_block_indexing() do
        merged_block = a_merged[Block(2, 2)]
        @test size(merged_block) == (5, 2)
        @test data(merged_block)[1:2, :] ≈ ones(2, 2)
        @test data(merged_block)[3:5, :] ≈ 2 * ones(3, 2)

        # U1(0) × U1(0) block should be empty (no stored data)
        empty_block = a_merged[Block(1, 1)]
        @test size(empty_block) == (1, 1)
        @test all(iszero, data(empty_block))
    end
end

@testset "matricize 2D graded array → FusedGradedMatrix" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = zeros(Float64, g1, g2)

    block_11 = randn!(Matrix{Float64}(undef, 2, 1))
    block_22 = randn!(Matrix{Float64}(undef, 3, 2))
    with_block_indexing() do
        a[Block(1, 1)] = UniqueSectorArray(block_11, (U1(0), U1(0)))
        return a[Block(2, 2)] = UniqueSectorArray(block_22, (U1(1), U1(-1)))
    end

    fsm = matricizeperm(a, (1,), (2,))
    @test fsm isa FusedGradedMatrix{Float64}
    # Each stored N-D block lands in the coupled sector pairing its row charge with
    # the dual of its column charge: (U1(0), U1(0)) → U1(0), (U1(1), U1(-1)) → U1(1).
    @test collect(keys(fsm.blocks)) == [U1(0), U1(1)]
    @test blocklength(fsm, 1) == 2
    @test blocklength(fsm, 2) == 2
    @test data(fsm[Block(1, 1)]) ≈ block_11
    @test data(fsm[Block(2, 2)]) ≈ block_22
end

@testset "matricize 4D graded array → FusedGradedMatrix" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    a = zeros(Float64, g, g, dual(g), dual(g))

    with_block_indexing() do
        a[Block(1, 1, 1, 1)] =
            UniqueSectorArray(ones(1, 1, 1, 1), (U1(0), U1(0), dual(U1(0)), dual(U1(0))))
        return a[Block(2, 2, 2, 2)] =
            UniqueSectorArray(
            2 * ones(1, 1, 1, 1),
            (U1(1), U1(1), dual(U1(1)), dual(U1(1)))
        )
    end

    fsm = matricizeperm(a, (1, 2), (3, 4))
    @test fsm isa FusedGradedMatrix{Float64}
    @test collect(keys(fsm.blocks)) == [U1(0), U1(1), U1(2)]
    @test blocklength(fsm, 1) == 3
    @test blocklength(fsm, 2) == 3

    @test data(fsm[Block(1, 1)]) ≈ ones(1, 1)
    @test data(fsm[Block(2, 2)]) ≈ zeros(2, 2)
    @test data(fsm[Block(3, 3)]) ≈ 2 * ones(1, 1)
end

@testset "tr of a matricized graded array" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    a = ones(Float64, g, g, dual(g), dual(g))
    with_block_indexing() do
        return a[Block(2, 2, 2, 2)] .*= 2  # give the blocks distinct traces
    end

    # `tr` on the matricized graded matrix sums the diagonal blocks and matches the dense trace.
    fsm = matricizeperm(a, (1, 2), (3, 4))
    @test tr(fsm) ≈ tr(Array(fsm))
    # `TensorAlgebra.tr` over the (1, 2) | (3, 4) bipartition routes through the same path.
    @test TensorAlgebra.tr(a, (1, 2, 3, 4), (1, 2), (3, 4)) ≈ tr(Array(fsm))
end

@testset "matricize 3D graded array and unmatricize round-trip" begin
    # 3D case where the merged codomain (tensor product of two `r`s) has
    # sectors absent from the domain — the asymmetric `FusedGradedMatrix`
    # natively handles this (codomain has U1(2), domain has only U1(0) and
    # U1(1)).
    r = gradedrange([U1(0) => 1, U1(1) => 2])
    a = zeros(Float64, (r, r, dual(r)))
    with_block_indexing() do
        a[Block(1, 1, 1)] = fill(1.0, 1, 1, 1)
        a[Block(1, 2, 2)] = fill(2.0, 1, 2, 2)
        return a[Block(2, 1, 2)] = fill(3.0, 2, 1, 2)
    end

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
    @test a_back isa FusionArray
    @test ndims(a_back) == 3
    @test Array(a_back) ≈ Array(a)
end

@testset "Off-diagonal block setindex! errors" begin
    ax = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(Float64, ax, dual(ax))
    # A forbidden off-diagonal block is rejected (a `FusionArray` throws a TensorKit `SectorMismatch`).
    with_block_indexing() do
        @test_throws Exception (
            a[Block(1, 2)] =
                UniqueSectorArray(
                randn!(Matrix{Float64}(undef, 2, 3)),
                (U1(0), dual(U1(1)))
            )
        )
    end
end

@testset "contract 2D graded array (matrix-matrix)" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(Float64, g, dual(g))
    b = zeros(Float64, g, dual(g))

    a_11 = randn!(Matrix{Float64}(undef, 2, 2))
    a_22 = randn!(Matrix{Float64}(undef, 3, 3))
    b_11 = randn!(Matrix{Float64}(undef, 2, 2))
    b_22 = randn!(Matrix{Float64}(undef, 3, 3))
    with_block_indexing() do
        a[Block(1, 1)] = UniqueSectorArray(a_11, (U1(0), dual(U1(0))))
        a[Block(2, 2)] = UniqueSectorArray(a_22, (U1(1), dual(U1(1))))
        b[Block(1, 1)] = UniqueSectorArray(b_11, (U1(0), dual(U1(0))))
        return b[Block(2, 2)] = UniqueSectorArray(b_22, (U1(1), dual(U1(1))))
    end

    result, dimnames = contract(a, (1, -1), b, (-1, 2))
    @test result isa FusionArray{Float64, <:Any, 2}
    with_block_indexing() do
        @test data(result[Block(1, 1)]) ≈ a_11 * b_11
        @test data(result[Block(2, 2)]) ≈ a_22 * b_22
    end
end

@testset "contract graded array to a scalar (elt=$elt)" for elt in
    (Float64, ComplexF64)
    # A full contraction over every index collapses to a rank-0 result. The
    # destination is allocated as a rank-0 graded array (trivial sector), so the
    # whole matricize/mul!/unmatricize path stays in graded land; the result reads
    # back as a scalar via `result[]`.
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn(elt, (g, dual(g)))
    b = randn(elt, (dual(g), g))

    result = contract((), a, (1, 2), b, (1, 2))
    # A rank-0 contraction result is a graded array.
    @test result isa FusionArray{elt, <:Any, 0}
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
    a = FusionArray{Float64, U1}(undef, (), ())
    a[] = 4.0
    m = matricize(GradedArrays.SectorMatricize(), a, Val(0))
    @test m isa FusedGradedMatrix{Float64}
    @test size(m) == (1, 1)
    @test data(m[Block(1, 1)]) == fill(4.0, 1, 1)

    # `unmatricize` allocates a rank-0 graded array, like the higher-rank path.
    back = unmatricize(GradedArrays.SectorMatricize(), m, (), ())
    @test back isa FusionArray{Float64, <:Any, 0}
    @test back[] == 4.0
end

@testset "unmatricize UniqueSectorMatrix with SectorOneTo axes" begin
    # Create a 3D UniqueSectorArray, matricize it, then unmatricize and verify roundtrip
    codomain_ax = SectorOneTo(U1(0), 2)
    domain_ax1 = SectorOneTo(conj(U1(0)), 3)
    domain_ax2 = SectorOneTo(conj(U1(1)), 4)

    data_3d = randn!(Array{Float64}(undef, 2, 3, 4))
    s = UniqueSectorArray(
        data_3d,
        (sector(codomain_ax), sector(domain_ax1), sector(domain_ax2))
    )

    # Matricize with 1 codomain leg
    sm = matricize(s, Val(1))
    @test sm isa FusedSectorMatrix
    @test ndims(sm) == 2

    # Unmatricize back to 3D. The domain axes are passed codomain-facing (un-dualized),
    # so the stored `conj`-ed domain axes are given as their un-dualized counterparts.
    s_back = unmatricize(sm, (codomain_ax,), (conj(domain_ax1), conj(domain_ax2)))
    @test s_back isa UniqueSectorArray
    @test ndims(s_back) == 3
    @test size(s_back) == size(s)

    # For bosonic (U1) sectors, no fermionic phase, data should match
    @test Array(s_back) ≈ data_3d
end

@testset "contract 4D graded array" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn(g, g, dual(g), dual(g))
    b = randn(g, g, dual(g), dual(g))

    # Contract: a[1, -1, 2, -2] * b[2, -3, 1, -4] (permutes + contracts).
    result, dimnames = contract(a, (1, -1, 2, -2), b, (2, -3, 1, -4))
    @test result isa FusionArray

    # Verify numerics against the dense contraction of the same data.
    result_dense, _ = contract(Array(a), (1, -1, 2, -2), Array(b), (2, -3, 1, -4))
    @test Array(result) ≈ result_dense
end

@testset "scale! with β=0 zeros uninitialized blocks" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(Float64, g, dual(g))
    TensorAlgebra.scale!(a, false)
    with_block_indexing() do
        @test all(iszero, data(a[Block(1, 1)]))
        @test all(iszero, data(a[Block(2, 2)]))
    end
end

@testset "allocating broadcast produces correct results" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(Float64, (g, dual(g)))
    with_block_indexing() do
        a[Block(1, 1)] = [1.0 0.0; 0.0 1.0]
        return a[Block(2, 2)] = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    end

    b = 3 .* a
    c = a - a
    with_block_indexing() do
        @test data(b[Block(1, 1)]) == [3.0 0.0; 0.0 3.0]
        @test data(b[Block(2, 2)]) == [3.0 0.0 0.0; 0.0 3.0 0.0; 0.0 0.0 3.0]

        @test all(iszero, data(c[Block(1, 1)]))
        @test all(iszero, data(c[Block(2, 2)]))
    end
end

@testset "FusedGradedMatrix block-wise arithmetic" begin
    m = FusedGradedMatrix([[1.0 2.0; 3.0 4.0], [5.0 0.0; 0.0 6.0]], [U1(0), U1(1)])

    m2 = 3 * m
    @test data(m2[Block(1, 1)]) == [3.0 6.0; 9.0 12.0]
    @test data(m2[Block(2, 2)]) == [15.0 0.0; 0.0 18.0]

    n = FusedGradedMatrix([ones(2, 2), ones(2, 2)], [U1(0), U1(1)])
    s = m + n
    @test data(s[Block(1, 1)]) == [2.0 3.0; 4.0 5.0]
    @test data(s[Block(2, 2)]) == [6.0 1.0; 1.0 7.0]
end

@testset "FusedGradedMatrix linear broadcasting" begin
    m = FusedGradedMatrix([[1.0 2.0; 3.0 4.0], [5.0 0.0; 0.0 6.0]], [U1(0), U1(1)])
    n = FusedGradedMatrix([ones(2, 2), ones(2, 2)], [U1(0), U1(1)])

    s = m .+ n
    @test data(s[Block(1, 1)]) == [2.0 3.0; 4.0 5.0]
    @test data(s[Block(2, 2)]) == [6.0 1.0; 1.0 7.0]

    p = 3 .* m
    @test data(p[Block(1, 1)]) == [3.0 6.0; 9.0 12.0]
    @test data(p[Block(2, 2)]) == [15.0 0.0; 0.0 18.0]

    c = similar(m, Float64)
    c .= 3 .* m .+ 2 .* n
    @test data(c[Block(1, 1)]) == [5.0 8.0; 11.0 14.0]
    @test data(c[Block(2, 2)]) == [17.0 2.0; 2.0 20.0]

    # Nonlinear broadcasts are rejected rather than silently mishandled.
    @test_throws ArgumentError m .+ 1
    @test_throws ArgumentError m .^ 2
end

@testset "FusedGradedMatrix conj is disallowed ($label)" for (label, sectorpairs) in (
        "bosonic" => [U1(0) => 1, U1(1) => 2, U1(2) => 1],
        "fermionic" => [FermionNumber(0) => 1, FermionNumber(1) => 2, FermionNumber(2) => 1],
    )
    # `conj` would dualize each coupled sector, flipping the first axis to dual, which the fused
    # storage types disallow. The dotted form is rejected by the constructor invariant when it
    # allocates the dual-keyed result. Conjugate the `FusionArray` (or matricize) instead.
    m = randn!(FusedGradedMatrix{ComplexF64}(undef, sectorpairs))
    @test_throws ErrorException conj(m)
    @test_throws ArgumentError conj.(m)
end

@testset "FusedGradedMatrix non-abelian (SU2) broadcasting" begin
    # A non-abelian block's axis length is its reduced length times the sector's quantum dimension
    # (SU2(1) has dimension 3), so the broadcast allocator has to key on reduced lengths, and the
    # per-block phase must not require unique fusion.
    m = randn!(FusedGradedMatrix{ComplexF64}(undef, [SU2(0) => 2, SU2(1) => 3]))
    n = randn!(FusedGradedMatrix{ComplexF64}(undef, [SU2(0) => 2, SU2(1) => 3]))

    s = 2 .* m .- n
    @test s isa FusedGradedMatrix
    @test Array(s) ≈ 2 .* Array(m) .- Array(n)

    v = randn!(
        FusedGradedVector{ComplexF64}(undef, [SU2(0) => 1, SU2(1 // 2) => 2, SU2(1) => 1])
    )
    @test (3 .* v) isa FusedGradedVector
    @test Array(3 .* v) ≈ 3 .* Array(v)
end

@testset "FusedGradedVector conj is disallowed" begin
    # `conj` would dualize the sectors, flipping the first axis to dual, which the fused storage
    # types disallow. The dotted form is rejected by the constructor invariant when it allocates the
    # dual-keyed result. Conjugate the `FusionArray` (or matricize) instead.
    v = randn!(
        FusedGradedVector{ComplexF64}(
            undef,
            [FermionNumber(n) => l for (n, l) in zip(0:2, (1, 2, 1))]
        )
    )
    @test_throws ErrorException conj(v)
    @test_throws ArgumentError conj.(v)
end

@testset "FusedGradedVector non-abelian Array reconstruction" begin
    # The structural factor of a block is `SectorOnesVector`, the all-ones vector of length equal
    # to the sector's quantum dimension, so materializing repeats each reduced value over the
    # irrep (`SU2(j)` has quantum dimension `2j + 1`: 1, 2, 3 here).
    v = FusedGradedVector(
        [Float64[10.0], Float64[20.0], Float64[30.0]],
        [SU2(0), SU2(1 // 2), SU2(1)]
    )
    @test sector(view(v, Block(2))) isa SectorOnesVector
    @test Array(sector(view(v, Block(2)))) == ones(2)
    @test Array(v) == [10.0, 20.0, 20.0, 30.0, 30.0, 30.0]
end

# Regression coverage for TensorAlgebra-level unmatricize-axis bugs on graded
# operators: a factor's reconstructed axes must respect the conj/dual pairing
# between contracted bonds rather than reuse the factor's own axes.
@testset "TA.svd_compact round-trip on a graded array (axes_S regression)" begin
    s = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
    A = randn(Float64, (s, dual(s)))
    U, S, Vᴴ = TensorAlgebra.svd_compact(A, (1,), (2,))
    US = contract((:a, :r), U, (:a, :i), S, (:i, :r))
    USV = contract((:a, :b), US, (:a, :r), Vᴴ, (:r, :b))
    @test A ≈ USV
end

@testset "TA.gram_eigh_full_with_pinv (axes_Y regression)" begin
    s = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
    B = randn(Float64, (s,), (s,))
    # PSD by construction. Build `A = B * B'` block-wise via `contract` with an explicit `conj`.
    A = contract((:a, :b), B, (:a, :r), conj(B), (:b, :r))
    X, Y = TensorAlgebra.gram_eigh_full_with_pinv(A, (1,), (2,))
    # X · conj(X) ≈ A on the rank subspace.
    @test A ≈ contract((:a, :b), X, (:a, :r), conj(X), (:b, :r))
    # Y is a left inverse of X on the rank subspace.
    YX = contract((:r, :s), Y, (:r, :a), X, (:a, :s))
    @test YX ≈ TensorAlgebra.one(YX, (:r, :s), (:r,), (:s,))
end

@testset "contract rejects mismatched contracted-axis duality (bosonic)" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(Float64, g, dual(g))
    randn!(a)

    # The contracted leg of `a` is `dual(g)`; here `b`'s contracted leg is also
    # `dual(g)`, which is neither the canonical dual pairing nor (for bosonic
    # U1) an accepted same-`isdual` pair, so the contraction is rejected.
    b = zeros(Float64, dual(g), dual(g))
    @test_throws ArgumentError contract(a, (1, -1), b, (-1, 2))

    # Sanity: the canonically dual-paired contraction is accepted.
    b_ok = zeros(Float64, g, dual(g))
    randn!(b_ok)
    result, = contract(a, (1, -1), b_ok, (-1, 2))
    @test result isa FusionArray
end
