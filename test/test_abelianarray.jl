using BlockArrays: BlockArrays, Block, blocklength, blocklengths
using Dictionaries: Dictionary
using GradedArrays: GradedArrays, AbelianGradedArray, AbelianSectorArray,
    AbstractGradedArray, FusedGradedMatrix, FusedGradedVector, GradedOneTo, SU2,
    SectorRange, U1, blockstoredlength, data, datalengths, dual, eachblockstoredindex,
    gradedrange, isdual, sectoraxes, sectors, sectortype, to_gradedrange
using LinearAlgebra: LinearAlgebra
using Random: Random
using SparseArraysBase: isstored
using TensorAlgebra: TensorAlgebra, fill_map, matricize, ones_map, rand_map, randn_map,
    unmatricize, zeros_map
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_broken, @test_throws, @testset

# The graded constructors (`randn`/`rand`/`zeros`/…) route through the selected backend, so
# assertions on backend-routed results check the active backend's type. `FUSION_BACKEND` guards the
# `broken =` markers on the handful of behaviors not yet at parity on the fusion backend (tracked in
# the parity plan).
const FUSION_BACKEND = GradedArrays.graded_backend == "fusion"
const GradedArrayT = FUSION_BACKEND ? GradedArrays.FusionArray : AbelianGradedArray

@testset "AbelianGradedArray" begin
    # Helper: build U1 axes
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])

    @testset "Construction" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        @test a isa AbelianGradedArray{Float64, U1, 2, Matrix{Float64}}
        @test a isa AbstractGradedArray{Float64, <:Any, 2}
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
        g1_dual = conj(gradedrange([U1(0) => 2, U1(1) => 3]))
        a = AbelianGradedArray{Float64}(undef, g1_dual, g2)
        a[Block(1, 1)] = ones(2, 1)

        blk = a[Block(1, 1)]
        @test sectoraxes(blk) == (conj(U1(0)), U1(0))
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

    @testset "block storage interface (backend-routed)" begin
        # Exercises the active backend's array (FusionArray on the fusion backend): the stored blocks
        # are the symmetry-allowed external blocks, and `isstored` / `blockstoredlength` derive from
        # them through `blocks`.
        g = gradedrange([U1(0) => 2, U1(1) => 3])
        a = randn(g, dual(g))
        @test a isa GradedArrayT
        @test issetequal(eachblockstoredindex(a), GradedArrays.allowedblocks(axes(a)))
        @test blockstoredlength(a) == length(collect(eachblockstoredindex(a)))
        @test isstored(a, Block(1, 1))
        @test !isstored(a, Block(1, 2))  # symmetry-forbidden

        # A multi-leg array: several external blocks fuse to the same coupled sector.
        a3 = zeros(g, g, dual(g))
        @test issetequal(eachblockstoredindex(a3), GradedArrays.allowedblocks(axes(a3)))
        @test blockstoredlength(a3) == length(collect(eachblockstoredindex(a3)))
    end

    @testset "isstored(a, ::Block)" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        @test isstored(a, Block(1, 1))
        @test !isstored(a, Block(2, 1))  # symmetry-forbidden block
        m = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), 2 * ones(3, 3)])
        @test isstored(m, Block(1, 1))
        @test !isstored(m, Block(1, 2))  # off-diagonal in sector space
        @test !isstored(m, Block(3, 1))  # out of range
    end

    @testset "blocks accessor (fused)" begin
        m = FusedGradedMatrix([U1(0), U1(1)], [ones(2, 2), 2 * ones(3, 3)])
        b = BlockArrays.blocks(m)
        @test size(b) == (2, 2)
        @test data(b[1, 1]) ≈ ones(2, 2)     # stored diagonal block, shares data
        @test_throws ErrorException b[1, 2]  # off-diagonal is symmetry-forbidden
        v = FusedGradedVector([U1(0), U1(1)], [ones(2), 2 * ones(3)])
        bv = BlockArrays.blocks(v)
        @test size(bv) == (2,)
        @test data(bv[2]) ≈ 2 * ones(3)
    end

    @testset "Scalar getindex" begin
        # `zero!` so the other allowed block (2, 2) is initialized: the elementwise comparison
        # below reads every position, and an undef block could hold `NaN` (never `==` itself).
        a = TensorAlgebra.zero!(AbelianGradedArray{Float64}(undef, g1, g2))
        a[Block(1, 1)] = reshape([5.0, 7.0], 2, 1)
        dense = Array(a)
        # Scalar reads match the dense array elementwise, including forbidden positions
        # (unstored blocks), which read as a structural zero.
        @test all(a[i, j] == dense[i, j] for i in axes(a, 1), j in axes(a, 2))
        @test a[1, 1] === 5.0
        @test a[3, 1] === 0.0  # falls in an unstored (forbidden) block
    end

    @testset "Scalar setindex!" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = reshape([5.0, 7.0], 2, 1)
        # Writing into an allowed block updates the single element and reads back.
        a[1, 1] = 42.0
        @test a[1, 1] === 42.0
        @test a[2, 1] === 7.0  # neighboring element in the same block is untouched
        # A forbidden position has no valid target.
        @test_throws ErrorException (a[3, 1] = 1.0)
    end

    @testset "Scalar indexing requires unique fusion" begin
        # The fused representation can hold non-abelian sectors, where scalar indexing is
        # not well defined; it must error rather than read/write past the reduced block data.
        m = FusedGradedMatrix([SU2(0), SU2(1 // 2)], [ones(1, 1), ones(1, 1)])
        @test_throws ErrorException m[1, 1]
        @test_throws ErrorException (m[1, 1] = 1.0)
    end

    @testset "unsupported ops error gracefully" begin
        a = TensorAlgebra.zero!(AbelianGradedArray{Float64}(undef, g1, g2))
        # No block-aware `adjoint` for a general graded array (it would silently scalar-index).
        @test_throws ErrorException a'
        # Mixing a graded array with a plain dense array is rejected rather than recursing.
        dense = zeros(size(a))
        @test_throws ErrorException a ≈ dense
        @test_throws ErrorException a - dense
        # Broadcasting with a scalar still works.
        @test 2 .* a isa AbelianGradedArray
    end

    @testset "rank-0 (scalar) array" begin
        # A rank-0 graded array holds a single trivial-sector value. `S` can't be
        # inferred from the empty axes, so the undef constructor takes it explicitly.
        a = AbelianGradedArray{Float64, U1, 0, Array{Float64, 0}}(undef, ())
        @test ndims(a) == 0
        @test size(a) == ()
        @test axes(a) == ()
        @test sectortype(a) === U1
        @test collect(eachblockstoredindex(a)) == [Block()]

        # The convenience constructor infers `S` from the axes, which is impossible for
        # empty axes, so it requires at least one axis; a rank-0 array uses the
        # fully-parameterized form above.
        @test_throws MethodError AbelianGradedArray{Float64}(undef, ())

        # `a[]` is allowed (one element, no coordinates), unlike higher-rank scalar
        # indexing.
        a[] = 3.5
        @test a[] == 3.5

        # The block view shares data as a rank-0 `AbelianSectorArray`.
        v = view(a, Block())
        @test v isa AbelianSectorArray{Float64, <:Any, 0}
        @test v[] == 3.5

        # `similar` with empty axes builds a rank-0 graded array carrying the
        # prototype's sector type, even from a higher-rank prototype.
        s = similar(AbelianGradedArray{Float64}(undef, g1, g2), ComplexF64, ())
        @test s isa AbelianGradedArray{ComplexF64, <:Any, 0}
        @test sectortype(s) === U1
    end

    @testset "Dual axes" begin
        g1_dual = conj(gradedrange([U1(0) => 2, U1(1) => 3]))
        g2_dual = conj(gradedrange([U1(0) => 1, U1(-1) => 2]))
        a = AbelianGradedArray{Float64}(undef, g1_dual, g2_dual)

        @test isdual(axes(a, 1)) == true
        @test isdual(axes(a, 2)) == true
        @test size(a) == (5, 3)

        a[Block(1, 1)] = ones(2, 1)
        blk = a[Block(1, 1)]
        @test sectoraxes(blk) == (conj(U1(0)), conj(U1(0)))
    end

    @testset "similar" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)

        a2 = similar(a)
        @test a2 isa AbelianGradedArray{Float64, <:Any, 2}
        @test size(a2) == size(a)
        # similar now allocates all allowed blocks (same as constructor)
        @test length(collect(eachblockstoredindex(a2))) == 2

        a3 = similar(a, ComplexF64)
        @test a3 isa AbelianGradedArray{ComplexF64, <:Any, 2}
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
        # Uses the `AbelianGradedMatrix` alias for 2D arrays.
        @test occursin("AbelianGradedMatrix", s)
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

        # Unstored (symmetry-forbidden) blocks error rather than reading back as zero
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

        # fill! fills stored blocks block-wise with any value
        fill!(a, 1.0)
        @test all(==(1.0), a.blockdata[(1, 1)])

        # zero! zeros stored blocks in place (blocks stay allocated)
        a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), ones(2, 2))
        TensorAlgebra.zero!(a)
        @test !isempty(a.blockdata)
        @test all(iszero, a.blockdata[(1, 1)])
    end
end

@testset "FusedGradedMatrix sectors/blocks constructor" begin
    m = FusedGradedMatrix([U1(0), U1(1)], [[1.0 2.0; 3.0 4.0], ones(3, 3)])
    @test m isa FusedGradedMatrix{Float64}
    @test data(m[Block(1, 1)]) == [1.0 2.0; 3.0 4.0]
    @test data(m[Block(2, 2)]) == ones(3, 3)

    # Non-abelian (SU2): the codomain block lengths pick up the irrep
    # dimension `2j + 1`, so `Block(k, k)` has size
    # `(sectorlength × datalength)^2`.
    m_su2 = FusedGradedMatrix(
        [SU2(0), SU2(1 // 2), SU2(1)],
        [[1.0;;], [1.0 2.0; 3.0 4.0], Matrix{Float64}(LinearAlgebra.I, 3, 3)]
    )
    @test m_su2 isa FusedGradedMatrix{Float64, SU2, Matrix{Float64}}
    @test collect(keys(m_su2.blocks)) == [SU2(0), SU2(1 // 2), SU2(1)]
    @test data(m_su2[Block(1, 1)]) == [1.0;;]
    @test data(m_su2[Block(2, 2)]) == [1.0 2.0; 3.0 4.0]
    # Block(2, 2) lives in SU2(1/2), which has dim 2 → 4×4 in dense view.
    @test size(m_su2[Block(2, 2)]) == (4, 4)
end

@testset "FusedGradedMatrix * FusedGradedMatrix" begin
    @testset "U1 (abelian)" begin
        a = FusedGradedMatrix([U1(0), U1(1)], [[2.0;;], [1.0 2.0; 3.0 4.0]])
        b = FusedGradedMatrix([U1(0), U1(1)], [[3.0;;], [0.0 1.0; 1.0 0.0]])
        c = a * b
        @test collect(keys(c.blocks)) == [U1(0), U1(1)]
        @test data(c[Block(1, 1)]) == [6.0;;]
        @test data(c[Block(2, 2)]) == [1.0 2.0; 3.0 4.0] * [0.0 1.0; 1.0 0.0]
    end
    @testset "SU2 (non-abelian)" begin
        a = FusedGradedMatrix([SU2(0), SU2(1 // 2)], [[2.0;;], [1.0 2.0; 3.0 4.0]])
        b = FusedGradedMatrix([SU2(0), SU2(1 // 2)], [[3.0;;], [0.0 1.0; 1.0 0.0]])
        c = a * b
        @test collect(keys(c.blocks)) == [SU2(0), SU2(1 // 2)]
        @test data(c[Block(1, 1)]) == [6.0;;]
        @test data(c[Block(2, 2)]) == [1.0 2.0; 3.0 4.0] * [0.0 1.0; 1.0 0.0]
    end
    @testset "mismatched sectors throws" begin
        a = FusedGradedMatrix([U1(0), U1(1)], [[2.0;;], [1.0 2.0; 3.0 4.0]])
        b = FusedGradedMatrix([U1(0)], [[3.0;;]])
        @test_throws DimensionMismatch a * b
    end
end

@testset "FusedGradedMatrix undef constructor" begin
    sectors = [U1(0), U1(1)]
    cod = Dictionary{U1, Int}(sectors, [2, 3])
    dom = Dictionary{U1, Int}(sectors, [1, 2])

    @testset "Default D = Matrix{T}" begin
        m = FusedGradedMatrix{Float64}(undef, cod, dom)
        @test m isa FusedGradedMatrix{Float64, U1, Matrix{Float64}}
        @test length(m.blocks) == 2
        @test collect(keys(m.blocks)) == sectors
        @test size(m.blocks[U1(0)]) == (2, 1)
        @test size(m.blocks[U1(1)]) == (3, 2)
    end

    @testset "Fully parameterized" begin
        m = FusedGradedMatrix{Float64, U1, Matrix{Float64}}(undef, cod, dom)
        @test m isa FusedGradedMatrix{Float64, U1, Matrix{Float64}}
        @test size(m.blocks[U1(0)]) == (2, 1)
    end

    @testset "Rejects unsorted sectors" begin
        cod_bad = Dictionary{U1, Int}([U1(1), U1(0)], [2, 3])
        @test_throws ArgumentError FusedGradedMatrix{Float64}(undef, cod_bad, dom)
    end

    @testset "Square shorthands set domain = codomain" begin
        # The `(sectors, lengths)`, single-argument pairs, and single-`GradedOneTo` forms all build
        # square blocks equal to the two-argument form with the codomain repeated as the domain.
        square = FusedGradedMatrix{Float64}(undef, sectors, [2, 3], [2, 3])
        @test axes(FusedGradedMatrix{Float64}(undef, sectors, [2, 3])) == axes(square)
        @test axes(FusedGradedMatrix{Float64}(undef, sectors .=> [2, 3])) == axes(square)
        @test axes(FusedGradedMatrix{Float64}(undef, gradedrange(sectors .=> [2, 3]))) ==
            axes(square)
    end
end

@testset "FusedGradedMatrix asymmetric (cod ≠ dom) sectors" begin
    cod = Dictionary{U1, Int}([U1(0), U1(1), U1(2)], [2, 3, 4])
    dom = Dictionary{U1, Int}([U1(1), U1(2), U1(3)], [3, 4, 5])
    blks = Dictionary{U1, Matrix{Float64}}(
        [U1(1), U1(2)],
        [ones(3, 3), 2 * ones(4, 4)]
    )
    m = FusedGradedMatrix(cod, dom, blks)

    @test m isa FusedGradedMatrix{Float64, U1, Matrix{Float64}}
    @test size(m) == (9, 12)            # 2+3+4 = 9, 3+4+5 = 12
    @test sectors(axes(m, 1)) == [U1(0), U1(1), U1(2)]
    @test sectors(axes(m, 2)) == [U1(1), U1(2), U1(3)]
    @test collect(keys(m.blocks)) == [U1(1), U1(2)]

    # Stored block access by sector key.
    @test m.blocks[U1(1)] == ones(3, 3)
    @test m.blocks[U1(2)] == 2 * ones(4, 4)

    # eachblockstoredindex maps sectors to (cod_pos, dom_pos): U1(1) is
    # cod position 2, dom position 1; U1(2) is cod 3, dom 2.
    stored = collect(eachblockstoredindex(m))
    @test Block(2, 1) in stored
    @test Block(3, 2) in stored

    # Adjoint swaps codomain/domain dicts and adjoints each block.
    mh = m'
    @test mh.codomain == m.domain
    @test mh.domain == m.codomain
    @test collect(keys(mh.blocks)) == collect(keys(m.blocks))
    @test mh.blocks[U1(1)] == ones(3, 3)'
    @test size(mh) == (size(m, 2), size(m, 1))

    # Multiplication: A's domain must match B's codomain (sectors and sizes).
    cod_A = Dictionary{U1, Int}([U1(0), U1(1)], [2, 3])
    dom_A = Dictionary{U1, Int}([U1(1), U1(2)], [3, 4])
    blks_A = Dictionary{U1, Matrix{Float64}}([U1(1)], [ones(3, 3)])
    A = FusedGradedMatrix(cod_A, dom_A, blks_A)

    cod_B = Dictionary{U1, Int}([U1(1), U1(2)], [3, 4])
    dom_B = Dictionary{U1, Int}([U1(0), U1(1)], [2, 3])
    blks_B = Dictionary{U1, Matrix{Float64}}([U1(1)], [2 * ones(3, 3)])
    B = FusedGradedMatrix(cod_B, dom_B, blks_B)

    C = A * B
    @test sectors(axes(C, 1)) == [U1(0), U1(1)]
    @test sectors(axes(C, 2)) == [U1(0), U1(1)]
    # Every allowed block of C is allocated. U1(0) lives in both C.codomain
    # and C.domain so it gets a (zero) block — no contraction path through
    # U1(0) since neither A.domain nor B.codomain carries it. U1(1) carries
    # the full matrix product.
    @test collect(keys(C.blocks)) == [U1(0), U1(1)]
    @test all(iszero, C.blocks[U1(0)])
    @test C.blocks[U1(1)] ≈ ones(3, 3) * (2 * ones(3, 3))
end

@testset "FusedGradedMatrix invariant: allowed blocks must be allocated" begin
    cod = Dictionary{U1, Int}([U1(0), U1(1)], [2, 3])
    dom = Dictionary{U1, Int}([U1(0), U1(1)], [4, 5])

    # Missing an allowed block (U1(0)) should error.
    blks_missing = Dictionary{U1, Matrix{Float64}}([U1(1)], [ones(3, 5)])
    @test_throws ArgumentError FusedGradedMatrix(cod, dom, blks_missing)

    # All allowed blocks present → ok.
    blks_full = Dictionary{U1, Matrix{Float64}}(
        [U1(0), U1(1)],
        [ones(2, 4), ones(3, 5)]
    )
    m = FusedGradedMatrix(cod, dom, blks_full)
    @test collect(keys(m.blocks)) == [U1(0), U1(1)]

    # Sectors with zero size on either side are not "allowed" — no block needed.
    cod_z = Dictionary{U1, Int}([U1(1)], [3])
    dom_z = Dictionary{U1, Int}([U1(0), U1(1)], [4, 5])
    blks_z = Dictionary{U1, Matrix{Float64}}([U1(1)], [ones(3, 5)])
    m_z = FusedGradedMatrix{Float64}(undef, cod_z, dom_z)
    @test collect(keys(m_z.blocks)) == [U1(1)]

    # `undef` constructor allocates all allowed blocks automatically.
    m_undef = FusedGradedMatrix{Float64}(undef, cod, dom)
    @test collect(keys(m_undef.blocks)) == [U1(0), U1(1)]
    @test size(m_undef.blocks[U1(0)]) == (2, 4)
    @test size(m_undef.blocks[U1(1)]) == (3, 5)
end

@testset "Block-aware random fills and iszero on AbelianGradedArray" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    rng = Random.Xoshiro(42)

    # In-place rand!/randn! fill each stored block via the underlying
    # block's method, no scalar indexing. Both the no-rng and rng-explicit
    # entry points must work.
    a = AbelianGradedArray{Float64}(undef, g1, dual(g1))
    fill!(a, 0)
    @test iszero(a)
    Random.randn!(rng, a)
    @test !iszero(a)
    fill!(a, 0)
    Random.randn!(a)
    @test !iszero(a)
    Random.rand!(rng, a)
    @test !iszero(a)
    fill!(a, 0)
    Random.rand!(a)
    @test !iszero(a)

    # Constructor form: `rand(T, axes)` / `randn(T, axes)` for graded axes
    # builds an `AbelianGradedArray` with the right block structure.
    r = randn(rng, Float64, (g1, dual(g1)))
    @test r isa GradedArrayT{Float64, <:Any, 2}
    @test axes(r) == (g1, dual(g1))
    @test !iszero(r)
    u = rand(rng, Float64, (g1, dual(g1)))
    @test u isa GradedArrayT{Float64, <:Any, 2}
    @test !iszero(u)
end

@testset "Block-aware ones/fill and rand/randn shorthands" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])

    # `ones`/`fill` allocate the allowed sectors and fill them with the value, exactly
    # like `fill!(zeros(ax...), v)`. Vararg and tuple forms, default and explicit eltype.
    reference1(v) = fill!(zeros(typeof(v), g1, g2), v)
    @testset "ones" begin
        for o in
            (ones(g1, g2), ones(Float64, g1, g2), ones((g1, g2)), ones(Float64, (g1, g2)))
            @test o isa GradedArrayT{Float64, <:Any, 2}
            @test axes(o) == (g1, g2)
            @test Array(o) == Array(reference1(1.0))
        end
        @test ones(ComplexF64, g1, g2) isa GradedArrayT{ComplexF64, <:Any, 2}
    end
    @testset "fill" begin
        for a in (fill(2.5, g1, g2), fill(2.5, (g1, g2)))
            @test a isa GradedArrayT{Float64, <:Any, 2}
            @test axes(a) == (g1, g2)
            @test Array(a) == Array(reference1(2.5))
        end
        @test fill(1.0im, g1, g2) isa GradedArrayT{ComplexF64, <:Any, 2}
    end

    # rand/randn shorthands forward to the canonical `(rng, T, tuple)` form, so a seeded
    # shorthand matches a seeded canonical call. All non-canonical arg shapes are covered.
    @testset "$f shorthands" for f in (rand, randn)
        @test f(g1, g2) isa GradedArrayT{Float64, <:Any, 2}
        @test f(ComplexF64, g1, g2) isa GradedArrayT{ComplexF64, <:Any, 2}
        @test f((g1, g2)) isa GradedArrayT{Float64, <:Any, 2}
        @test f(ComplexF64, (g1, g2)) isa GradedArrayT{ComplexF64, <:Any, 2}
        @test f(Random.Xoshiro(1), g1, g2) isa GradedArrayT{Float64, <:Any, 2}
        @test f(Random.Xoshiro(1), (g1, g2)) isa GradedArrayT{Float64, <:Any, 2}
        # Seeded shorthand == seeded canonical, for every shape that fills in defaults.
        @test Array(f(Random.Xoshiro(1), Float64, g1, g2)) ==
            Array(f(Random.Xoshiro(1), Float64, (g1, g2)))
        @test Array(f(Random.Xoshiro(1), g1, g2)) ==
            Array(f(Random.Xoshiro(1), Float64, (g1, g2)))
    end

    # Regression: the leading mandatory `GradedOneTo` keeps these from pirating the
    # zero-argument / no-graded Base calls.
    @testset "no piracy of zero-argument Base calls" begin
        @test zeros() isa Array{Float64, 0}
        @test ones() isa Array{Float64, 0}
        @test fill(1.0) isa Array{Float64, 0}
        @test rand() isa Float64
        @test randn() isa Float64
        @test zeros(2, 3) isa Matrix{Float64}
        @test ones(Float64, 2, 3) isa Matrix{Float64}
    end
end

@testset "conj flips axis duality" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn(ComplexF64, (g, dual(g)))

    ca = conj(a)
    @test ca isa GradedArrayT
    # Each axis flips its duality, mirroring `conj` on the axis types.
    @test isdual(axes(ca, 1)) == !isdual(axes(a, 1))
    @test isdual(axes(ca, 2)) == !isdual(axes(a, 2))
    @test axes(ca, 1) == conj(axes(a, 1))
    @test axes(ca, 2) == conj(axes(a, 2))
    # The data is conjugated element-wise.
    @test Array(ca) ≈ conj(Array(a))
end

@testset "isdiag on AbelianGradedMatrix" begin
    g = gradedrange([U1(0) => 2, U1(1) => 2])

    # Block-diagonal with each stored block diagonal.
    a = AbelianGradedArray{Float64}(undef, g, dual(g))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), [1.0 0.0; 0.0 2.0])
    a[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), [3.0 0.0; 0.0 4.0])
    @test LinearAlgebra.isdiag(a)

    # A non-diagonal stored block breaks it.
    b = AbelianGradedArray{Float64}(undef, g, dual(g))
    b[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), [1.0 5.0; 0.0 2.0])
    b[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), [3.0 0.0; 0.0 4.0])
    @test !LinearAlgebra.isdiag(b)
end

@testset "projectto! / project" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    dest = AbelianGradedArray{Float64}(undef, g, dual(g))

    # Project a dense source: the symmetry-allowed (block-diagonal) regions are
    # copied, the forbidden off-diagonal regions are dropped.
    src = randn(5, 5)
    @test TensorAlgebra.projectto!(dest, src) === dest
    @test data(dest[Block(1, 1)]) ≈ src[1:2, 1:2]
    @test data(dest[Block(2, 2)]) ≈ src[3:5, 3:5]
    proj = Array(dest)
    @test iszero(proj[1:2, 3:5])
    @test iszero(proj[3:5, 1:2])

    # The checked `project` accepts a source already in the allowed subspace...
    src_allowed = Array(dest)
    dest_ok = TensorAlgebra.project(src_allowed, (g,), (g,))
    @test dest_ok isa GradedArrayT
    @test Array(dest_ok) ≈ src_allowed

    # ... and rejects one carrying significant forbidden-block weight, which the
    # unchecked projection drops silently.
    src_bad = copy(src_allowed)
    src_bad[1, 5] += 10.0
    @test_throws InexactError TensorAlgebra.project(src_bad, (g,), (g,))
    @test Array(TensorAlgebra.unchecked_project(src_bad, (g,), (g,))) ≈ src_allowed

    # A lower-rank `src` may omit trailing length-1 axes; `projectto!` reshapes it. This is
    # the auxiliary flux-canceling leg idiom: an extra length-1 axis carries the charge that
    # keeps an otherwise-forbidden component (here the `U1(1)` sector) in the allowed subspace.
    site = gradedrange([U1(0) => 1, U1(1) => 1])
    aux = gradedrange([U1(0) => 1])
    state = AbelianGradedArray{Float64}(undef, site, aux)
    @test TensorAlgebra.projectto!(state, [1.0, 0.0]) === state
    @test size(state) == (2, 1)
    @test Array(state) == reshape([1.0, 0.0], 2, 1)
    @test TensorAlgebra.project([1.0, 0.0], (site, aux)) isa GradedArrayT
end

@testset "project with a derived auxiliary leg" begin
    # spin-1/2 site: up = U1(1), down = U1(-1)
    g = gradedrange([U1(1) => 1, U1(-1) => 1])
    Splus = [0.0 1.0; 0.0 0.0]   # |down> -> |up>, flux +2
    Sminus = [0.0 0.0; 1.0 0.0]  # |up> -> |down>, flux -2
    Sz = [0.5 0.0; 0.0 -0.5]     # neutral

    # The internal abelian fast path reads the net flux from the dominant entry.
    @test GradedArrays.projected_charge(Splus, (g,), (g,)) == U1(2)
    @test GradedArrays.projected_charge(Sminus, (g,), (g,)) == U1(-2)
    @test GradedArrays.projected_charge(Sz, (g,), (g,)) == U1(0)

    # With all axes given (`src` rank matches the physical axes), `project` is the plain
    # projection into the allowed subspace.
    @test TensorAlgebra.project(Sz, (g,), (g,)) isa GradedArrayT

    # A trailing surplus axis in `src` (here the length-1 last axis of `(2,2,1)`, with only 2
    # physical axes given) is the aux: `project` derives its space into a flux-canceling last
    # domain axis, giving a valid zero-total-flux tensor whose squeezed data is the original
    # operator. The result's shape matches `src`'s.
    t = TensorAlgebra.project(reshape(Splus, 2, 2, 1), (g,), (g,))
    @test t isa GradedArrayT
    @test size(t) == (2, 2, 1)
    @test blockstoredlength(t) == 1
    @test Array(t)[:, :, 1] == Splus

    # A neutral operator still gets an aux, but a trivial one (dummy bond).
    tz = TensorAlgebra.project(reshape(Sz, 2, 2, 1), (g,), (g,))
    @test size(tz) == (2, 2, 1)
    @test Array(tz)[:, :, 1] == Sz

    # Multi-slice / flat-state aux derivation is not yet implemented on the fusion backend, which
    # routes `project` through TensorAlgebra's TensorMap path: its `infer_aux_space` requires the
    # aux slices pre-sorted by canonical sector order rather than deriving that order itself
    # (tracked in the parity plan). Runs fully on `AbelianGradedArray`.
    if FUSION_BACKEND
        @test_broken false
    else
        # A multi-slice aux derives a direct sum, one charge per slice — the MPO-virtual-leg
        # structure: each slice's flux is canceled by its own aux sector, so the whole tensor is
        # invariant even though the slices carry different charges.
        stack = cat(reshape(Splus, 2, 2, 1), reshape(Sminus, 2, 2, 1); dims = 3)
        ts = TensorAlgebra.project(stack, (g,), (g,))
        @test ts isa GradedArrayT
        @test size(ts) == (2, 2, 2)
        @test Array(ts)[:, :, 1] == Splus
        @test Array(ts)[:, :, 2] == Sminus

        # ... including a neutral slice mixed with a charged one.
        tsz = TensorAlgebra.project(
            cat(reshape(Sz, 2, 2, 1), reshape(Splus, 2, 2, 1); dims = 3), (g,), (g,)
        )
        @test Array(tsz)[:, :, 1] == Sz
        @test Array(tsz)[:, :, 2] == Splus

        # Contiguous equal charges merge into one sector of that multiplicity (matching the
        # `TensorMap` backend's `s => m` behavior), and agree with passing the merged aux
        # explicitly through the all-given form.
        pp = cat(reshape.((Splus, Splus), 2, 2, 1)...; dims = 3)
        tpp = TensorAlgebra.project(pp, (g,), (g,))
        @test blocklengths(axes(tpp, 3)) == [2]
        @test Array(tpp) == pp
        tmerged = TensorAlgebra.project(pp, (g,), (g, gradedrange([U1(2) => 2])))
        @test axes(tmerged) == axes(tpp)
        @test Array(tmerged) == pp

        # Non-contiguous repeats stay separate sectors (a `GradedOneTo` permits unmerged
        # duplicates), so slice order is always preserved.
        repeats = cat(reshape.((Splus, Sz, Splus), 2, 2, 1)...; dims = 3)
        trep = TensorAlgebra.project(repeats, (g,), (g,))
        @test size(trep) == (2, 2, 3)
        @test blocklengths(axes(trep, 3)) == [1, 1, 1]
        @test Array(trep) == repeats

        # Only one trailing surplus axis is supported: more is an error, not a silent flattening.
        @test_throws ArgumentError TensorAlgebra.project(
            reshape(stack, 2, 2, 2, 1), (g,), (g,)
        )

        # A lower-rank `src` that omits explicitly-given trailing length-1 axes is the
        # trailing-axes tolerance, not a surplus axis: it pads, it does not derive an extra leg.
        # The domain aux is given codomain-facing (stored dualized), so `[U1(1)]` cancels the
        # charge-1 component.
        site = gradedrange([U1(0) => 1, U1(1) => 1])
        saux = gradedrange([U1(1) => 1])
        po = TensorAlgebra.project([0.0, 1.0], (site,), (saux,))
        @test size(po) == (2, 1)
        @test Array(po) == reshape([0.0, 1.0], 2, 1)

        # The flat all-codomain (state) form also derives: a stack of basis states with different
        # charges gets a multi-sector aux.
        ps = TensorAlgebra.project([1.0 0.0; 0.0 1.0], (site,))
        @test size(ps) == (2, 2)
        @test Array(ps) == [1.0 0.0; 0.0 1.0]

        # `project` verifies nothing was discarded after the derivation: a slice with weight
        # outside its derived (dominant-entry) charge is rejected, where `unchecked_project`
        # silently drops it.
        junk = copy(reshape(Splus, 2, 2, 1))
        junk[2, 2, 1] = 0.3
        @test_throws InexactError TensorAlgebra.project(junk, (g,), (g,))
        @test Array(TensorAlgebra.unchecked_project(junk, (g,), (g,)))[:, :, 1] == Splus

        # `tryproject` is the nullable sibling of `project`: branch on whether the data is
        # invariant in the given axes, falling back to deriving the flux-carrying aux.
        v_inv, v_chg = [1.0, 0.0], [0.0, 1.0]
        @test TensorAlgebra.tryproject(v_inv, (site,)) isa GradedArrayT
        @test isnothing(TensorAlgebra.tryproject(v_chg, (site,)))
        t_chg = @something TensorAlgebra.tryproject(v_chg, (site,)) TensorAlgebra.project(
            reshape(v_chg, 2, 1), (site,)
        )
        @test ndims(t_chg) == 2
        @test Array(t_chg) == reshape(v_chg, 2, 1)
        t_inv = @something TensorAlgebra.tryproject(v_inv, (site,)) TensorAlgebra.project(
            reshape(v_inv, 2, 1), (site,)
        )
        @test ndims(t_inv) == 1
    end
end

@testset "flux-canceling constructor" begin
    r1 = gradedrange([U1(0) => 1, U1(1) => 2])
    r2 = gradedrange([U1(0) => 2, U1(1) => 1])
    r3 = gradedrange([U1(0) => 1, U1(1) => 1])
    c = U1(1)

    # Flat form `randn(c, codomain)`: the aux is the sole domain leg, so this is exactly the
    # split constructor over `(codomain, (aux,))`. The aux is a multiplicity-1 leg carrying
    # `c`, dualized (in the domain) and dangling last, and forces the physical legs to `+c`.
    flat = randn(Random.Xoshiro(1), Float64, c, (r1, r2))
    @test flat == randn_map(Random.Xoshiro(1), Float64, (r1, r2), (to_gradedrange(c),))
    @test flat isa GradedArrayT{Float64}
    @test ndims(flat) == 3
    @test blockstoredlength(flat) > 0                 # a real, non-empty flux tensor
    aux = axes(flat, 3)
    @test length(aux) == 1
    @test isdual(aux)
    @test sectors(aux) == [c]

    # Map form `randn(c, codomain, domain)`: the aux is appended to the (dualized) domain,
    # dangling last, alongside the given domain legs.
    mp = randn(Random.Xoshiro(2), Float64, c, (r1, r2), (r3,))
    @test mp == randn_map(Random.Xoshiro(2), Float64, (r1, r2), (r3, to_gradedrange(c)))
    @test ndims(mp) == 4
    @test isdual(axes(mp, 3))                         # the given domain leg is dualized
    @test isdual(axes(mp, 4)) && length(axes(mp, 4)) == 1 && sectors(axes(mp, 4)) == [c]

    # The codomain-only form forwards to the empty-domain case, down to the RNG stream.
    @test randn(Random.Xoshiro(3), c, (r1, r2)) == randn(Random.Xoshiro(3), c, (r1, r2), ())

    # A different flux changes the aux sector (the charge really is carried on the leg).
    @test sectors(axes(randn(U1(0), (r1, r2)), 3)) == [U1(0)]

    # A raw `TensorKitSectors.Sector` (here a fermionic sector) works directly as the flux, with
    # no `SectorRange` wrap: these forms carry a physical graded axis, so the signature holds a
    # GradedArrays-owned type and a bare-sector flux is not type piracy. It matches the wrapped
    # flux, and the empty-codomain form does too.
    fn(n) = TKS.FermionNumber(n)
    sf = gradedrange([fn(0) => 2, fn(1) => 2])
    ferm = randn(Random.Xoshiro(4), fn(2), (sf, sf, sf, sf))
    @test ferm == randn_map(
        Random.Xoshiro(4), Float64, (sf, sf, sf, sf),
        (to_gradedrange(fn(2)),)
    )
    @test ferm == randn(Random.Xoshiro(4), SectorRange(fn(2)), (sf, sf, sf, sf))
    @test zeros(fn(0), (), (sf,)) == zeros(SectorRange(fn(0)), (), (sf,))
    @test ndims(ferm) == 5
    @test isdual(axes(ferm, 5))

    # `zeros`, `ones`, `rand`, and `fill` mirror `randn` (`fill` takes the value first).
    @test zeros(c, (r1, r2)) == zeros_map((r1, r2), (to_gradedrange(c),))
    @test ones(c, (r1, r2), (r3,)) == ones_map((r1, r2), (r3, to_gradedrange(c)))
    @test fill(2.5, c, (r1, r2)) == fill_map(2.5, (r1, r2), (to_gradedrange(c),))
    @test rand(Random.Xoshiro(5), Float64, c, (r1, r2), (r3,)) ==
        rand_map(Random.Xoshiro(5), Float64, (r1, r2), (r3, to_gradedrange(c)))
    @test eltype(randn(ComplexF64, c, (r1, r2))) == ComplexF64
    # An explicit empty domain matches omitting the domain.
    @test zeros(c, (r1, r2), ()) == zeros(c, (r1, r2))

    # Empty codomain `f(flux, (), (dom...))`: the physical legs all live in the dualized domain,
    # matching the flux form over `dual.(dom)` with no codomain.
    @test randn(Random.Xoshiro(6), Float64, c, (), (r1,)) ==
        randn(Random.Xoshiro(6), Float64, c, (dual(r1),))
    @test zeros(c, (), (r1, r2)) == zeros(c, (dual(r1), dual(r2)))
    @test ones(Float64, c, (), (r1,)) == ones(Float64, c, (dual(r1),))
    @test fill(2.5, c, (), (r1,)) == fill(2.5, c, (dual(r1),))
    @test zeros(c, (), ([U1(0) => 1, U1(1) => 2],)) == zeros(c, (dual(r1),))

    # Flux-only forms `f(flux)` and `f(flux, ())`: no physical axes, just the dangling flux leg.
    # Both are shorthands for `f(flux, (), ())`, a rank-1 array carrying `c` on a dualized leg.
    fluxonly = randn(Random.Xoshiro(7), Float64, c, (), ())
    @test fluxonly == randn(Random.Xoshiro(7), Float64, c)
    @test fluxonly == randn(Random.Xoshiro(7), Float64, c, ())
    @test ndims(fluxonly) == 1
    @test length(axes(fluxonly, 1)) == 1
    @test isdual(axes(fluxonly, 1))
    @test sectors(axes(fluxonly, 1)) == [c]
    @test zeros(c) == zeros(c, (), ())
    @test zeros(c, ()) == zeros(c, (), ())
    @test ones(Float64, c) == ones(Float64, c, (), ())
    @test ones(c, ()) == ones(c, (), ())
    @test fill(2.5, c) == fill(2.5, c, (), ())
    @test fill(2.5, c, ()) == fill(2.5, c, (), ())
    @test rand(Random.Xoshiro(8), c) == rand(Random.Xoshiro(8), c, (), ())
    @test eltype(randn(ComplexF64, c)) == ComplexF64
end

@testset "pairs-vector axis constructors" begin
    g = gradedrange([U1(0) => 2, U1(1) => 2])
    # An axis given as `sector => multiplicity` pairs is normalized to a `GradedOneTo`. Keys are
    # `SectorRange`s, whether a native GradedArrays sector or a raw TensorKitSectors sector wrapped
    # with `SectorRange`.
    ps = [U1(0) => 2, U1(1) => 2]                                       # native SectorRange keys
    pk = [SectorRange(TKS.U1Irrep(0)) => 2, SectorRange(TKS.U1Irrep(1)) => 2]  # wrapped raw keys
    @testset "$f codomain-only" for (f, ref) in
        ((zeros, zeros(g, g)), (ones, ones(g, g)))
        @test f(ps, ps) == ref
        @test f(pk, pk) == ref
        @test f(Float64, ps) == f((g,))
    end
    @test fill(2.5, ps, ps) == fill(2.5, g, g)
    @test axes(randn(ps)) == (g,)
    @test Array(randn(Random.Xoshiro(1), Float64, ps, ps)) ==
        Array(randn(Random.Xoshiro(1), Float64, g, g))
    @test Array(rand(Random.Xoshiro(1), Float64, pk)) ==
        Array(rand(Random.Xoshiro(1), Float64, (g,)))
end

@testset "split codomain/domain constructor" begin
    g = gradedrange([U1(0) => 1, U1(1) => 2])
    h = gradedrange([U1(0) => 2, U1(1) => 1])
    # `f((cod...), (dom...))` builds a tensor map: the domain axes are stored dual, so it equals
    # the codomain-only call over `(cod..., dual.(dom)...)` (and the `*_map` builder).
    @test randn(Random.Xoshiro(1), Float64, (g,), (h,)) ==
        randn(Random.Xoshiro(1), Float64, (g, dual(h)))
    @test randn(Random.Xoshiro(1), Float64, (g,), (h,)) ==
        randn_map(Random.Xoshiro(1), Float64, (g,), (h,))
    @test zeros((g,), (h,)) == zeros(g, dual(h))
    @test ones(Float64, (g,), (h,)) == ones(g, dual(h))
    @test fill(2.5, (g,), (h,)) == fill(2.5, g, dual(h))
    # Pairs-vector axes work in both slots, matching the `GradedOneTo` result.
    @test zeros(([U1(0) => 1, U1(1) => 2],), ([U1(0) => 2, U1(1) => 1],)) ==
        zeros(g, dual(h))

    # An explicit empty domain matches the codomain-only call.
    @test randn(Random.Xoshiro(1), Float64, (g, h), ()) ==
        randn(Random.Xoshiro(1), Float64, (g, h))
    @test zeros((g, h), ()) == zeros((g, h))
    @test ones(Float64, (g, h), ()) == ones(Float64, (g, h))
    @test fill(2.5, (g, h), ()) == fill(2.5, (g, h))

    # Empty codomain: every leg lives in the dualized domain, matching the codomain-only call
    # over `dual.(dom)`. Available for the pairs-vector spelling too.
    @test randn(Random.Xoshiro(1), Float64, (), (g,)) ==
        randn(Random.Xoshiro(1), Float64, (dual(g),))
    @test zeros((), (g, h)) == zeros(dual(g), dual(h))
    @test ones(Float64, (), (g,)) == ones(Float64, (dual(g),))
    @test fill(2.5, (), (g,)) == fill(2.5, (dual(g),))
    @test zeros((), ([U1(0) => 1, U1(1) => 2],)) == zeros(dual(g))
end

@testset "getindex (project dense onto graded axes)" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])

    # Indexing a dense array with graded ranges projects it onto the allowed blocks and
    # checks the discarded weight. A source already in the allowed subspace round-trips.
    ref = AbelianGradedArray{Float64}(undef, g, dual(g))
    TensorAlgebra.projectto!(ref, randn(5, 5))
    src = Array(ref)
    a = src[g, dual(g)]
    @test a isa GradedArrayT{Float64, <:Any, 2}
    @test axes(a) == (g, dual(g))
    @test Array(a) ≈ src

    # A source carrying forbidden-block weight is rejected.
    src_bad = copy(src)
    src_bad[1, 5] += 10.0
    @test_throws InexactError src_bad[g, dual(g)]

    # Rank reconciliation: a trailing size-1 graded bond is supplied implicitly, so a dense
    # matrix is reshaped up before projecting.
    aux = gradedrange([U1(0) => 1])
    a3 = src[g, dual(g), aux]
    @test a3 isa GradedArrayT{Float64, <:Any, 3}
    @test size(a3) == (5, 5, 1)

    # A single graded axis is disambiguated against `Base.getindex(::Array,
    # ::AbstractUnitRange)`, so a one-leg graded tensor can be built from a dense vector.
    vsrc = zeros(5)
    vsrc[1:2] .= randn(2)              # weight only in the trivial (U1(0)) block
    av = vsrc[g]
    @test av isa GradedArrayT{Float64, <:Any, 1}
    @test Array(av) ≈ vsrc
end

@testset "dot" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{ComplexF64}(undef, g, dual(g))
    b = AbelianGradedArray{ComplexF64}(undef, g, dual(g))
    Random.randn!(a)
    Random.randn!(b)
    @test LinearAlgebra.dot(a, b) ≈ LinearAlgebra.dot(Array(a), Array(b))

    # Mismatched axes are rejected.
    h = gradedrange([U1(0) => 1, U1(1) => 2])
    c = AbelianGradedArray{ComplexF64}(undef, h, dual(h))
    Random.randn!(c)
    @test_throws DimensionMismatch LinearAlgebra.dot(a, c)
end

@testset "sum" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{ComplexF64}(undef, g, dual(g))
    Random.randn!(a)
    @test sum(a) ≈ sum(Array(a))
end

@testset "maximum / minimum / extrema" begin
    # Forbidden and allowed-but-unstored blocks are zeros that the reductions must see, so
    # they agree with the dense array (which includes those zeros).
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = AbelianGradedArray{Float64}(undef, g, dual(g))
    Random.randn!(a)
    @test maximum(a) == maximum(Array(a))
    @test minimum(a) == minimum(Array(a))
    @test maximum(abs, a) == maximum(abs, Array(a))
    @test minimum(abs, a) == minimum(abs, Array(a))
    @test extrema(a) == extrema(Array(a))
    @test extrema(abs, a) == extrema(abs, Array(a))

    # When every element lives in a stored block there is no implicit zero to fold in, so a
    # sign-definite array keeps its sign.
    h = gradedrange([U1(1) => 2])
    c = fill(-1.0, 2, 2)[h, dual(h)]
    @test maximum(c) == -1.0
    @test maximum(c) == maximum(Array(c))
end

# The `FusedGradedMatrix asymmetric` testset above checks the adjoint on U1 with real blocks, which
# only exercises the transpose half. The adjoint itself is symmetry-agnostic (it conjugate-transposes
# each reduced block and swaps codomain/domain), so its structural properties are checked here across
# abelian (U1), fermionic (`FermionParity`), and non-abelian (`SU2`) sectors with complex, non-square
# blocks via direct construction (which, unlike the matricize path, accepts non-abelian sectors).
@testset "FusedGradedMatrix adjoint (direct construction, complex; $name)" for (
        name,
        sectors,
    ) in (
        ("U1", [U1(0), U1(1)]),
        ("FermionParity", [TKS.FermionParity(false), TKS.FermionParity(true)]),
        ("SU2", [SU2(0), SU2(1 // 2)]),
    )
    m = FusedGradedMatrix(sectors, [randn(ComplexF64, 2, 3), randn(ComplexF64, 1, 2)])
    mh = m'

    # Adjoint swaps codomain/domain and dualizes the axes.
    @test mh.codomain == m.domain
    @test mh.domain == m.codomain
    @test axes(mh) == (dual(axes(m, 2)), dual(axes(m, 1)))
    @test size(mh) == (size(m, 2), size(m, 1))

    # Each block is the conjugate-transpose of the original (no per-sector sign), and for this
    # complex data conjugation genuinely differs from a plain transpose.
    for c in keys(m.blocks)
        @test mh.blocks[c] == m.blocks[c]'
        @test mh.blocks[c] != transpose(m.blocks[c])
    end

    # Double adjoint is the identity.
    mhh = mh'
    @test axes(mhh) == axes(m)
    @test all(c -> mhh.blocks[c] == m.blocks[c], keys(m.blocks))
end

# The adjoint also composes with the matricize/unmatricize round-trip. This path is abelian-only, so
# it covers U1 and `FermionParity` (not the non-abelian `SU2` above).
@testset "FusedGradedMatrix adjoint round-trip (matricized, complex; $name)" for (
        name,
        r1,
        r2,
    ) in (
        ("U1", gradedrange([U1(0) => 2, U1(1) => 2]), gradedrange([U1(0) => 1, U1(1) => 2])),
        (
            "FermionParity",
            gradedrange([TKS.FermionParity(false) => 2, TKS.FermionParity(true) => 2]),
            gradedrange([TKS.FermionParity(false) => 1, TKS.FermionParity(true) => 2]),
        ),
    )
    # Matricize a 3-leg array (codomain legs 1, 2; domain leg 3) into a `FusedGradedMatrix`.
    a = randn(ComplexF64, r1, r2, dual(r1))
    m = matricize(a, Val(2))
    mh = m'

    # Unmatricize the adjoint (norm-preserving), then re-matricize recovers its blocks.
    back = unmatricize(mh, (r1,), (r1, r2))
    @test LinearAlgebra.norm(back) ≈ LinearAlgebra.norm(a)
    remat = matricize(back, Val(1))
    for c in keys(mh.blocks)
        @test remat.blocks[c] ≈ mh.blocks[c]
    end
end

@testset "real / imag" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 1, U1(-1) => 2])
    a = randn(ComplexF64, (g1, dual(g2)))

    ra = real(a)
    ia = imag(a)
    @test ra isa GradedArrayT
    @test eltype(ra) == Float64
    # `real` / `imag` act element-wise on the data and keep the axes (unlike `conj`, which dualizes).
    @test axes(ra) == axes(a)
    @test axes(ia) == axes(a)
    @test Array(ra) == real(Array(a))
    @test Array(ia) == imag(Array(a))
    @test Array(ra) + im * Array(ia) ≈ Array(a)
    # A real-eltype array is returned unchanged (no copy).
    @test real(ra) === ra
end

# `real` / `imag` conjugate-transpose nothing and touch no sector: they act element-wise on the
# reduced block data across abelian (U1), fermionic (`FermionParity`), and non-abelian (`SU2`)
# sectors, checked via direct construction with complex, non-square blocks.
@testset "real / imag on FusedGradedMatrix (direct construction, complex; $name)" for (
        name,
        sectors,
    ) in (
        ("U1", [U1(0), U1(1)]),
        ("FermionParity", [TKS.FermionParity(false), TKS.FermionParity(true)]),
        ("SU2", [SU2(0), SU2(1 // 2)]),
    )
    m = FusedGradedMatrix(sectors, [randn(ComplexF64, 2, 3), randn(ComplexF64, 1, 2)])
    rm = real(m)
    imm = imag(m)
    @test rm isa FusedGradedMatrix
    @test eltype(rm) == Float64
    @test axes(rm) == axes(m)
    for c in keys(m.blocks)
        @test rm.blocks[c] == real.(m.blocks[c])
        @test imm.blocks[c] == imag.(m.blocks[c])
        @test real.(m.blocks[c]) + im * imag.(m.blocks[c]) ≈ m.blocks[c]
    end
end
