using BlockArrays: BlockArrays, Block, blocklength
using Dictionaries: Dictionary
using GradedArrays: GradedArrays, AbelianGradedArray, AbelianSectorArray,
    AbstractGradedArray, FusedGradedMatrix, GradedOneTo, SU2, SectorRange, U1, data,
    datalengths, dual, eachblockstoredindex, gradedrange, isdual, sectoraxes, sectors,
    sectortype
using LinearAlgebra: LinearAlgebra
using Random: Random
using TensorAlgebra: TensorAlgebra
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

    @testset "Scalar indexing errors" begin
        a = AbelianGradedArray{Float64}(undef, g1, g2)
        a[Block(1, 1)] = ones(2, 1)
        @test_throws ErrorException a[1, 1]
        @test_throws ErrorException (a[1, 1] = 42.0)
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

        # fill! fills stored blocks block-wise with any value
        fill!(a, 1.0)
        @test all(==(1.0), a.blockdata[(1, 1)])

        # zero! zeros stored blocks in place (blocks stay allocated)
        a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), ones(2, 2))
        GradedArrays.FI.zero!(a)
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
    @test m_su2 isa FusedGradedMatrix{Float64, Matrix{Float64}, SU2}
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
        @test m isa FusedGradedMatrix{Float64, Matrix{Float64}, U1}
        @test length(m.blocks) == 2
        @test collect(keys(m.blocks)) == sectors
        @test size(m.blocks[U1(0)]) == (2, 1)
        @test size(m.blocks[U1(1)]) == (3, 2)
    end

    @testset "Fully parameterized" begin
        m = FusedGradedMatrix{Float64, Matrix{Float64}, U1}(undef, cod, dom)
        @test m isa FusedGradedMatrix{Float64, Matrix{Float64}, U1}
        @test size(m.blocks[U1(0)]) == (2, 1)
    end

    @testset "Rejects unsorted sectors" begin
        cod_bad = Dictionary{U1, Int}([U1(1), U1(0)], [2, 3])
        @test_throws ArgumentError FusedGradedMatrix{Float64}(undef, cod_bad, dom)
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

    @test m isa FusedGradedMatrix{Float64, Matrix{Float64}, U1}
    @test size(m) == (9, 12)            # 2+3+4 = 9, 3+4+5 = 12
    @test sectors(axes(m, 1)) == [U1(0), U1(1), U1(2)]
    @test sectors(axes(m, 2)) == dual.([U1(1), U1(2), U1(3)])
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
    @test sectors(axes(C, 2)) == dual.([U1(0), U1(1)])
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
    @test r isa AbelianGradedArray{Float64, 2}
    @test axes(r) == (g1, dual(g1))
    @test !iszero(r)
    u = rand(rng, Float64, (g1, dual(g1)))
    @test u isa AbelianGradedArray{Float64, 2}
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
            @test o isa AbelianGradedArray{Float64, 2}
            @test axes(o) == (g1, g2)
            @test Array(o) == Array(reference1(1.0))
        end
        @test ones(ComplexF64, g1, g2) isa AbelianGradedArray{ComplexF64, 2}
    end
    @testset "fill" begin
        for a in (fill(2.5, g1, g2), fill(2.5, (g1, g2)))
            @test a isa AbelianGradedArray{Float64, 2}
            @test axes(a) == (g1, g2)
            @test Array(a) == Array(reference1(2.5))
        end
        @test fill(1.0im, g1, g2) isa AbelianGradedArray{ComplexF64, 2}
    end

    # rand/randn shorthands forward to the canonical `(rng, T, tuple)` form, so a seeded
    # shorthand matches a seeded canonical call. All non-canonical arg shapes are covered.
    @testset "$f shorthands" for f in (rand, randn)
        @test f(g1, g2) isa AbelianGradedArray{Float64, 2}
        @test f(ComplexF64, g1, g2) isa AbelianGradedArray{ComplexF64, 2}
        @test f((g1, g2)) isa AbelianGradedArray{Float64, 2}
        @test f(ComplexF64, (g1, g2)) isa AbelianGradedArray{ComplexF64, 2}
        @test f(Random.Xoshiro(1), g1, g2) isa AbelianGradedArray{Float64, 2}
        @test f(Random.Xoshiro(1), (g1, g2)) isa AbelianGradedArray{Float64, 2}
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
    a = AbelianGradedArray{ComplexF64}(undef, g, dual(g))
    a[Block(1, 1)] = AbelianSectorArray((U1(0), dual(U1(0))), randn(ComplexF64, 2, 2))
    a[Block(2, 2)] = AbelianSectorArray((U1(1), dual(U1(1))), randn(ComplexF64, 3, 3))

    ca = conj(a)
    @test ca isa AbelianGradedArray
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

@testset "projectto! / checked_projectto!" begin
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

    # `checked_projectto!` accepts a source already in the allowed subspace.
    src_allowed = Array(dest)
    dest_ok = AbelianGradedArray{Float64}(undef, g, dual(g))
    @test TensorAlgebra.checked_projectto!(dest_ok, src_allowed) === dest_ok
    @test Array(dest_ok) ≈ src_allowed

    # ... and rejects one carrying significant forbidden-block weight.
    src_bad = copy(src_allowed)
    src_bad[1, 5] += 10.0
    dest_bad = AbelianGradedArray{Float64}(undef, g, dual(g))
    @test_throws InexactError TensorAlgebra.checked_projectto!(dest_bad, src_bad)
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
