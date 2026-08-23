using BlockArrays: Block, blocklengths, blocks
using Dictionaries: dictionary
using GradedArrays: GradedArrays, FusedGradedDiagonal, FusedGradedMatrix, FusedGradedOneTo,
    FusedGradedVector, GradedArray, SU2, SectorRange, U1, UniqueSectorArray, Z2,
    checksquare, data, dual, fusedgradeddiagonal, fusedgradedmatrix, fusedgradedvector,
    gradedrange, isblockdiag, isdual, issquare, ndims_codomain, ndims_domain, sector,
    sectordata, tensor_product, to_tensormap, with_block_indexing, with_scalar_indexing
using LinearAlgebra: Diagonal, diag, lmul!, rmul!
using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using Random: randn!
using TensorAlgebra: TensorAlgebra, bipermutedims, contract, eig_full, eigh_full, matricize,
    project_hermitian, svd_compact, unmatricize
using TensorKit: TensorKit, @tensor
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

# `GradedArray` delegates its heavy fusion-tree work (braiding, fermion signs, recoupling) to
# `TensorKit.TensorMap`, so every check here validates against the corresponding TensorKit
# operation on `TensorMap(fa)`. Contractions that change a factor's codomain/domain split are
# included on purpose: they exercise the leg-bend path in `matricize`, which is not a free
# reshape for the block-diagonal storage.

const fP0 = SectorRange(TKS.FermionParity(false))  # even parity
const fP1 = SectorRange(TKS.FermionParity(true))   # odd parity

# Bring a contraction result to a canonical all-codomain `TensorMap` with legs in `want` order, so
# results with different codomain/domain splits or operand orders compare with `≈`. Uses TensorKit's
# sign-aware `permute`, not `convert(Array, …)`: a dense array does not preserve fermionic signs.
function canonical(t, labels, want)
    perm = Tuple(findfirst(==(x), labels) for x in want)
    return TensorKit.permute(TensorKit.TensorMap(t), (perm, ()))
end

@testset "GradedArray" begin
    @testset "construction and TensorMap round-trip ($G)" for (G, i, j) in (
            (
                "U1",
                gradedrange([U1(0) => 2, U1(1) => 1]),
                gradedrange([U1(0) => 1, U1(1) => 2]),
            ),
            (
                "SU2", gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
                gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
            ),
        )
        a = randn((i,), (j,))
        @test a isa GradedArray
        @test size(a) == (length(i), length(j))
        # Codomain axis is stored as given; the domain axis is stored dualized.
        @test !isdual(axes(a, 1))
        @test isdual(axes(a, 2))
        # Round-tripping through a `TensorMap` and back preserves the data (axes derived from `t`).
        t = TensorKit.TensorMap(a)
        b = GradedArray(t)
        @test TensorKit.TensorMap(b) ≈ t
    end

    # The invariant the buffer redesign rests on: the matricized buffer is laid out exactly as
    # TensorKit's `.data`, so `to_tensormap` is a genuine zero-copy `TensorMap` view over it whose dense
    # form equals the copy-based reference conversion. (We compare against TensorKit's own dense form,
    # not `Array(::FusedGradedMatrix)`, because our `_to_blockarray` and TensorKit order the
    # degeneracy/multiplicity index differently within a non-abelian block, see the plan note.)
    @testset "to_tensormap is a zero-copy TensorMap ($G)" for (G, g) in (
            ("U1", gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])),
            ("SU2", gradedrange([SU2(0) => 3, SU2(1 // 2) => 2, SU2(1) => 1])),
        )
        a = randn((g,), (g,))
        t = to_tensormap(a)
        @test t isa TensorKit.TensorMap
        @test t.data === matricize(a).buffer                                # shares the buffer
        @test convert(Array, t) ≈ convert(Array, TensorKit.TensorMap(a))  # == copy-based reference
        @test convert(Array, to_tensormap(GradedArray(t))) ≈ convert(Array, t)  # round-trip

        # A diagonal factor maps to a zero-copy `DiagonalTensorMap` over its diagonal buffer.
        U, S, Vᴴ = MAK.svd_compact(matricize(a))
        Sa = GradedArray(S, (g,), (g,))
        ts = to_tensormap(Sa)
        @test ts isa TensorKit.DiagonalTensorMap
        @test ts.data === S.diag.buffer
        @test collect(MAK.diagview(ts)) ≈ S.diag.buffer
        @test convert(Array, ts) ≈ convert(Array, TensorKit.TensorMap(Sa))
    end

    @testset "external axes may be unfused or unsorted" begin
        ok = gradedrange([U1(0) => 2, U1(1) => 1])
        unsorted = gradedrange([U1(1) => 1, U1(0) => 2])
        unfused = gradedrange([U1(0) => 2, U1(1) => 1, U1(0) => 1])
        # The array carries unfused / unsorted external axes; the `matricized` backing stays
        # fused-sorted, so the per-leg sort permutation relates the two.
        @test GradedArray{Float64}(undef, (unsorted,), (ok,)) isa GradedArray
        @test GradedArray{Float64}(undef, (ok,), (unfused,)) isa GradedArray
        # The `TensorMap` / `ElementarySpace` conversion stays strict: it expects a fused-sorted range
        # (callers normalize with `sectormergesort` at the boundary).
        @test_throws ArgumentError TensorKit.ElementarySpace(unsorted)
        @test_throws ArgumentError TensorKit.ElementarySpace(unfused)
    end

    # `project` reorders unfused/unsorted external axes into the fused-sorted matricized backing.
    @testset "unfused/unsorted project round-trip ($name)" for (name, T, cod, dom) in (
            (
                "U1 unsorted codomain", Float64,
                (gradedrange([U1(1) => 1, U1(0) => 2]),),
                (gradedrange([U1(0) => 2, U1(1) => 1]),),
            ),
            (
                "U1 unfused codomain", Float64,
                (gradedrange([U1(0) => 2, U1(1) => 1, U1(0) => 1]),),
                (gradedrange([U1(0) => 1, U1(1) => 2]),),
            ),
            (
                "U1 unfused both, complex", ComplexF64,
                (gradedrange([U1(0) => 1, U1(1) => 1, U1(0) => 1]),),
                (gradedrange([U1(1) => 1, U1(0) => 2]),),
            ),
            (
                "SU2 unsorted", Float64,
                (gradedrange([SU2(1 // 2) => 1, SU2(0) => 2]),),
                (gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),),
            ),
            (
                "fermion unfused", Float64,
                (gradedrange([fP0 => 1, fP1 => 1, fP0 => 1]),),
                (gradedrange([fP0 => 2, fP1 => 1]),),
            ),
            (
                "U1 multi-leg unfused", Float64,
                (
                    gradedrange([U1(0) => 1, U1(1) => 1]),
                    gradedrange([U1(1) => 1, U1(0) => 1, U1(1) => 1]),
                ),
                (gradedrange([U1(0) => 1, U1(1) => 1]),),
            ),
        )
        all_axes = (cod..., dom...)
        # A dense source exactly in the allowed subspace over the (unfused/unsorted) axes: `project`
        # reorders it into fused-sorted order (block permutation) and `Array` scatters back.
        raw = Array(
            TensorAlgebra.unchecked_project(
                randn(T, map(length, all_axes)...),
                cod,
                dom
            )
        )
        @test !iszero(raw)
        a = TensorAlgebra.project(raw, cod, dom)
        @test a isa GradedArray
        # The requested (unfused/unsorted) axes are carried, not the fused-sorted backing order.
        @test axes(a) == (cod..., map(dual, dom)...)
        # Dense round-trip through the reorder in and out. Non-abelian recoupling adds float round-off,
        # so compare with `≈`.
        @test Array(a) ≈ raw
    end

    @testset "viewblock on unfused/unsorted axes" begin
        # A repeated sector on a leg means positional blocks are no longer 1-1 with the merged backing;
        # `viewblock` must return each positional block's own slice.
        g1 = gradedrange([U1(0) => 1, U1(1) => 1, U1(0) => 2])   # U1(0) repeated
        g2 = gradedrange([U1(1) => 2, U1(0) => 1, U1(1) => 1])   # U1(1) repeated, out of order
        a = randn((g1,), (g2,))
        dense = Array(a)
        elranges(g) = (
            c = cumsum(collect(blocklengths(g)));
            [(c[k] - blocklengths(g)[k] + 1):c[k] for k in eachindex(c)]
        )
        r1, r2 = elranges(g1), elranges(g2)
        with_block_indexing() do
            for B in GradedArrays.eachblockstoredindex(a)
                i, j = Int.(Tuple(B))
                # Each stored block is the dense sub-block at its positional (i, j) location.
                @test Array(GradedArrays.viewblock(a, B)) ≈ dense[r1[i], r2[j]]
            end
        end
        # The view shares storage, so writes land in the backing.
        B = first(GradedArrays.eachblockstoredindex(a))
        i, j = Int.(Tuple(B))
        with_block_indexing() do
            with_scalar_indexing() do
                return GradedArrays.viewblock(a, B)[1, 1] = 42.0
            end
        end
        @test Array(a)[r1[i][1], r2[j][1]] == 42.0
    end

    @testset "real / imag ($G)" for (G, i, j) in (
            (
                "U1",
                gradedrange([U1(0) => 2, U1(1) => 1]),
                gradedrange([U1(0) => 1, U1(1) => 2]),
            ),
            (
                "SU2", gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
                gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
            ),
        )
        a = randn(ComplexF64, (i,), (j,))
        ra = real(a)
        ia = imag(a)
        @test ra isa GradedArray
        @test ia isa GradedArray
        @test eltype(ra) == Float64
        @test axes(ra) == axes(a)
        # Forwarded to the matricized fused matrix, so real/imag act block-wise on the reduced data.
        ma = matricize(a)
        for c in keys(sectordata(ma))
            @test sectordata(matricize(ra))[c] == real.(sectordata(ma)[c])
            @test sectordata(matricize(ia))[c] == imag.(sectordata(ma)[c])
        end
    end

    @testset "conj (split-preserving) ($G)" for (G, cod, dom) in (
            (
                "U1 (1,1)",
                (gradedrange([U1(0) => 2, U1(1) => 1]),),
                (gradedrange([U1(0) => 1, U1(1) => 2]),),
            ),
            (
                "SU2 (1,2)",
                (gradedrange([SU2(0) => 1, SU2(1 // 2) => 2]),),
                (
                    gradedrange([SU2(0) => 2, SU2(1 // 2) => 1]),
                    gradedrange([SU2(0) => 1, SU2(1 // 2) => 2]),
                ),
            ),
            (
                "fermion (2,1)",
                (gradedrange([fP0 => 2, fP1 => 1]), gradedrange([fP0 => 1, fP1 => 2])),
                (gradedrange([fP0 => 2, fP1 => 1]),),
            ),
        )
        a = randn(ComplexF64, cod, dom)
        c = conj(a)
        @test c isa GradedArray
        # Split preserved (unlike the `conj.(a)` broadcast, which materializes all-codomain), per-leg
        # axes dualized.
        @test (ndims_codomain(c), ndims_domain(c)) == (ndims_codomain(a), ndims_domain(a))
        @test axes(c) == map(dual, axes(a))
        # Same tensor as the broadcast conj and an involution, compared at a common split via `≈` (a
        # non-abelian double conj picks up recoupling round-off, so `==` is too strict).
        @test c ≈ conj.(a)
        @test conj(c) ≈ a

        # The abelian block type carries the split too, so extracting a block commutes with conj
        # (matching TensorKit): `conj(a[I])` equals `conj(a)[I]`, split and axes preserved. The SU2
        # blocks take a separate non-abelian conj path, so skip them here.
        if !startswith(G, "SU2")
            with_block_indexing() do
                for I in GradedArrays.eachblockstoredindex(a)
                    @test Array(conj(a[I])) ≈ Array(c[I])
                    @test axes(conj(a[I])) == axes(c[I])
                end
            end
        end
    end

    @testset "block biperm matches parent bipermutedims (fermion split)" begin
        # A block-level `bipermutedimsopadd!` on a split block must carry the same fermion sign
        # (permutation braiding plus the codomain/domain bends) that TensorKit applies when the whole
        # `GradedArray` is bipermuted. Compare each stored block against the corresponding block of
        # the parent `bipermutedims`, across biperms that move legs across the codomain/domain split.
        fax =
            () -> gradedrange([TKS.FermionParity(false) => 1, TKS.FermionParity(true) => 2])
        for (cod, dom, pc, pd) in (
                ((fax(), fax()), (fax(),), (3, 1), (2,)),
                ((fax(), fax()), (fax(),), (2,), (3, 1)),
                ((fax(), fax()), (fax(),), (), (1, 2, 3)),
                ((fax(), fax()), (fax(), fax()), (1, 4, 2), (3,)),
            )
            fa = randn(ComplexF64, cod, dom)
            perm = (pc..., pd...)
            fp = bipermutedims(fa, pc, pd)
            with_block_indexing() do
                for I in GradedArrays.eachblockstoredindex(fa)
                    Ip = Block(ntuple(d -> Int(Tuple(I)[perm[d]]), ndims(fa))...)
                    gt = fp[Ip]
                    dest = UniqueSectorArray(similar(data(gt)), sector(gt))
                    TensorAlgebra.bipermutedimsopadd!(
                        dest,
                        identity,
                        fa[I],
                        pc,
                        pd,
                        true,
                        false
                    )
                    @test Array(dest) ≈ Array(gt)
                    @test axes(dest) == axes(gt)
                end
            end
        end
    end

    @testset "conj of a rank-0 GradedArray" begin
        a = randn!(GradedArray{ComplexF64, U1}(undef, (), ()))
        c = conj(a)
        @test c isa GradedArray
        @test ndims(c) == 0
        @test c[] ≈ conj(a[])
    end

    @testset "contraction ($G)" for (G, i, j, k, l) in (
            (
                "U1", gradedrange([U1(0) => 2, U1(1) => 1]),
                gradedrange([U1(0) => 1, U1(1) => 2]),
                gradedrange([U1(0) => 1, U1(1) => 1]), gradedrange([U1(0) => 2, U1(1) => 1]),
            ),
            (
                "SU2", gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
                gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
                gradedrange([SU2(1 // 2) => 1, SU2(1) => 1]),
                gradedrange([SU2(0) => 1, SU2(1 // 2) => 1]),
            ),
        )
        # 2-leg: the stored split already matches, matmul composition compares directly.
        m1 = randn((i,), (k,))
        m2 = randn((k,), (j,))
        c2, = contract(m1, (:i, :k), m2, (:k, :j))
        @test c2 isa GradedArray
        @test TensorKit.TensorMap(c2) ≈ TensorKit.TensorMap(m1) * TensorKit.TensorMap(m2)

        # 3-leg over two shared indices: the free/contracted split differs from the stored
        # split, so this exercises the leg bend in `matricize`.
        a = randn((i, j), (k,))          # (i,j; k)
        b = randn((k,), (j, l))          # (k; j,l)
        ta = TensorKit.TensorMap(a)
        tb = TensorKit.TensorMap(b)
        c, lc = contract(a, (:i, :j, :k), b, (:k, :j, :l))
        @tensor ref[i, l] := ta[i, j, k] * tb[k, j, l]
        @test canonical(c, lc, [:i, :l]) ≈ ref
    end

    @testset "permutedims (braiding)" begin
        i = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        k = gradedrange([SU2(1 // 2) => 1, SU2(1) => 1])
        a = randn((i, j), (k,))
        # Move a domain leg into the codomain: a braid + bend that TensorKit handles.
        p = bipermutedims(a, (1, 3), (2,))
        @test p isa GradedArray
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(a), ((1, 3), (2,)))
    end

    @testset "bend of a Diagonal-blocked GradedArray" begin
        # `Diagonal` blocks arise from factorization factors (e.g. the singular values of a gauge).
        # A non-trivial bend reads them through the same path TensorKit uses for `DiagonalTensorMap`,
        # so it must match the dense-blocked equivalent.
        g = gradedrange([Z2(0) => 2, Z2(1) => 3])
        d0, d1 = randn(2), randn(3)
        diag = GradedArray(
            fusedgradedmatrix([Z2(0), Z2(1)] .=> [Diagonal(d0), Diagonal(d1)]), (g,),
            (g,)
        )
        dense = GradedArray(
            fusedgradedmatrix(
                [Z2(0), Z2(1)] .=> [Matrix(Diagonal(d0)), Matrix(Diagonal(d1))]
            ),
            (g,), (g,)
        )
        p = bipermutedims(diag, (1, 2), ())
        @test p isa GradedArray
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(dense), ((1, 2), ()))
    end

    @testset "fused matrix permute stays fused and rejects unrepresentable permutations" begin
        i = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        M = matricize(randn((i,), (j,)))
        @test M isa FusedGradedMatrix

        # The identity-copy permute stays a fused matrix and copies (no bend into a `GradedArray`).
        for p in (TensorAlgebra.permutedims(M, (1,), (2,)), bipermutedims(M, (1,), (2,)))
            @test p isa FusedGradedMatrix
            @test p == M
            @test p !== M
        end

        # An all-codomain or transposing permute is unrepresentable for a fused matrix and errors
        # instead of silently densifying into a `GradedArray`.
        @test_throws ArgumentError TensorAlgebra.permutedims(M, (1, 2))
        @test_throws ArgumentError TensorAlgebra.permutedims(M, (2, 1))
        @test_throws ArgumentError permutedims(M, (1, 2))
        @test_throws ArgumentError TensorAlgebra.permutedims(M, (2,), (1,))

        # `similar_map` with explicit axes off fused storage is undefined.
        @test_throws ArgumentError TensorAlgebra.similar_map(
            M,
            eltype(M),
            (axes(M, 1),),
            ()
        )
    end

    @testset "factorization (svd_compact)" begin
        i = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 1, SU2(1 // 2) => 2])
        m = randn((i,), (j,))
        u, s, v = svd_compact(m, (1,), (2,))
        @test u isa GradedArray
        @test s isa FusedGradedDiagonal
        @test v isa GradedArray
        us, = contract(u, (:i, :b), s, (:b, :c))
        rec, = contract(us, (:i, :c), v, (:c, :j))
        @test rec ≈ m
    end

    @testset "factorization (eig_full)" begin
        g = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        m = randn((g,), (g,))
        d, v = eig_full(m, (1,), (2,))
        @test d isa FusedGradedDiagonal
        @test v isa GradedArray
        # The diagonal factor contracts back as a matrix-level operand: A V ≈ V D.
        av, = contract(m, (:i, :k), v, (:k, :b))
        vd, = contract(v, (:i, :b), d, (:b, :c))
        @test av ≈ vd
    end

    @testset "factorization (eigh_full)" begin
        g = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        m = project_hermitian(randn((g,), (g,)), (1,), (2,))
        @test m isa GradedArray
        d, v = eigh_full(m, (1,), (2,))
        @test d isa FusedGradedDiagonal
        @test v isa GradedArray
        av, = contract(m, (:i, :k), v, (:k, :b))
        vd, = contract(v, (:i, :b), d, (:b, :c))
        @test av ≈ vd
    end

    @testset "construct from a factorization bond axis (FusedGradedOneTo)" begin
        g = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        m = randn((g,), (g,))
        d, v = eig_full(m, (1,), (2,))
        # A factorization exposes `FusedGradedOneTo` bond axes; the public construction surface takes
        # any `AbstractGradedOneTo`, so feeding one back in builds a `GradedArray`.
        bond = axes(d, 1)
        @test bond isa FusedGradedOneTo
        @test zeros(Float64, (bond,), (bond,)) isa GradedArray
        @test randn(Float64, (bond,), (bond,)) isa GradedArray
    end

    @testset "broadcasting (linear combinations)" begin
        i = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        a = randn((i,), (j,))
        b = randn((i,), (j,))
        # Linear combinations move all axes to the codomain, so normalize back to a `(i; j)`
        # `TensorMap` before comparing.
        back(x) = TensorKit.permute(TensorKit.TensorMap(x), ((1,), (2,)))
        @test a + b isa GradedArray
        @test back(a + b) ≈ back(a) + back(b)
        @test back(a - b) ≈ back(a) - back(b)
        @test back(2 * a - 3 * b) ≈ 2 * back(a) - 3 * back(b)
        # Operands with different codomain/domain splits but equal axes still add (each is bent).
        c = randn((i, dual(j)), ())
        @test (ndims_codomain(c), ndims_domain(c)) != (ndims_codomain(a), ndims_domain(a))
        @test axes(c) == axes(a)
        @test back(a + c) ≈ back(a) + back(c)
    end

    @testset "broadcasting through a PermutedDims operand" begin
        # ITensorBase wraps by-name-aligned operands in `TensorAlgebra.PermutedDims`, so `similar`
        # looks through it; a single wrapped operand would otherwise leave the operand filter empty.
        i = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        a = randn((i,), (j,))
        b = randn((i,), (j,))
        @test 2 .* TensorAlgebra.PermutedDims(a, (1, 2)) isa GradedArray
        @test a .+ TensorAlgebra.PermutedDims(b, (1, 2)) isa GradedArray

        S = fusedgradeddiagonal([SU2(0) => randn(2), SU2(1 // 2) => randn(1)])
        out = 2 .* TensorAlgebra.PermutedDims(S, (1, 2))
        @test out isa FusedGradedDiagonal
        @test diag(out) ≈ 2 .* diag(S)
    end

    @testset "mixed dense/diagonal broadcasting and conj" begin
        S = fusedgradeddiagonal([SU2(0) => randn(2), SU2(1 // 2) => randn(1)])
        M = matricize(randn((axes(S, 1),), (axes(S, 2),)))
        @test axes(M) == axes(S)

        # All-diagonal linear broadcasts stay diagonal (diagonal destination through the
        # generic block loop).
        @test 2 .* S isa FusedGradedDiagonal
        @test Array(2 .* S) ≈ 2 .* Array(S)
        @test S .+ S isa FusedGradedDiagonal
        @test Array(S .+ S) ≈ Array(S) .+ Array(S)

        # Mixing a diagonal with a dense fused matrix promotes to dense.
        @test M .+ S isa FusedGradedMatrix
        @test Array(M .+ S) ≈ Array(M) .+ Array(S)
        @test S .+ M isa FusedGradedMatrix
        @test Array(S .+ M) ≈ Array(S) .+ Array(M)
        @test 2 .* S .+ M isa FusedGradedMatrix
        @test Array(2 .* S .+ M) ≈ 2 .* Array(S) .+ Array(M)

        # `conj` dualizes the first axis, which fused storage forbids, so it errors for both
        # storage types in the function and broadcast forms; `adjoint` is the valid conjugation.
        @test_throws Exception conj(S)
        @test_throws Exception conj.(S)
        @test_throws Exception conj(M)
        @test_throws Exception conj.(M)
        @test_throws Exception M .+ conj.(S)
        @test Array(S') ≈ Array(S)'
    end

    @testset "fermionic" begin
        i = gradedrange([fP0 => 2, fP1 => 1])
        j = gradedrange([fP0 => 1, fP1 => 2])
        k = gradedrange([fP0 => 1, fP1 => 1])
        l = gradedrange([fP0 => 2, fP1 => 1])

        # Fermion signs on a permute ride `tensoradd!` for free.
        a3 = randn((i, j), (k,))
        p = bipermutedims(a3, (2, 1), (3,))
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(a3), ((2, 1), (3,)))

        # The contraction twist: 2-leg matches TensorKit composition directly.
        m1 = randn((i,), (k,))
        m2 = randn((k,), (j,))
        c2, = contract(m1, (:i, :k), m2, (:k, :j))
        @test TensorKit.TensorMap(c2) ≈ TensorKit.TensorMap(m1) * TensorKit.TensorMap(m2)

        # Multi-leg fermionic contraction matches `@tensor`, and is independent of operand
        # order — the property the twist exists to guarantee.
        a = randn((i, j), (k,))          # (i,j; k)
        b = randn((k,), (j, l))          # (k; j,l)
        ta = TensorKit.TensorMap(a)
        tb = TensorKit.TensorMap(b)
        c1, lc1 = contract(a, (:i, :j, :k), b, (:k, :j, :l))
        c2, lc2 = contract(b, (:k, :j, :l), a, (:i, :j, :k))
        @tensor ref[i, l] := ta[i, j, k] * tb[k, j, l]
        @test canonical(c1, lc1, [:i, :l]) ≈ ref
        @test canonical(c1, lc1, [:i, :l]) ≈ canonical(c2, lc2, [:i, :l])
    end
end

# The diagonal factor `S`/`D` (SVD singular values, eigenvalues) is a `FusedGradedDiagonal`. These
# check that `matricize`/`unmatricize`, `contract`, broadcast addition, and band-wise matrix
# functions preserve the diagonal type exactly where the result stays diagonal and densify where it
# genuinely is not (the operand-driven policy).
@testset "diagonal algebra output types ($G)" for (G, g) in (
        ("U1", gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])),
        ("SU2", gradedrange([SU2(0) => 3, SU2(1 // 2) => 2, SU2(1) => 1])),
    )
    S = MAK.svd_compact(matricize(randn((g,), (g,))))[2]
    @test S isa FusedGradedDiagonal

    @testset "matricize" begin
        # `{1,1}` is the identity matricization of a diagonal. Any other codomain rank bends a leg,
        # which matrix-level fused storage cannot represent, so it errors.
        @test matricize(S, Val(1)) === S
        @test_throws ArgumentError matricize(S, Val(2))
    end

    @testset "unmatricize" begin
        # `{1,1}` bond preserves the diagonal, wrapped to the tensor-level `GradedArray`.
        r = unmatricize(S, (g,), (g,))
        @test r isa GradedArray
        @test matricize(r) isa FusedGradedDiagonal
        @test matricize(r) === S
    end

    @testset "contract" begin
        Sa = GradedArray(S, (g,), (g,))
        # Wrapped spectra contract through the generic dense allocation. Diagonal contract algebra
        # lives at the matrix level, where factorizations return spectra.
        C, _ = contract(Sa, (1, -1), Sa, (-1, 2))
        @test matricize(C) isa FusedGradedMatrix
        @test Array(C) ≈ Array(Sa) * Array(Sa)

        # Diagonal × dense densifies.
        adense = randn((g,), (g,))
        Cmix, _ = contract(Sa, (1, -1), adense, (-1, 2))
        @test matricize(Cmix) isa FusedGradedMatrix
        @test Array(Cmix) ≈ Array(Sa) * Array(adense)

        # Full contraction is a scalar; outer product is rank 4 — both dense.
        Cfull, _ = contract(Sa, (1, 2), Sa, (2, 1))
        @test ndims(Cfull) == 0
        @test matricize(Cfull) isa FusedGradedMatrix

        Couter, _ = contract(Sa, (1, 2), Sa, (3, 4))
        @test ndims(Couter) == 4
        @test matricize(Couter) isa FusedGradedMatrix
    end

    @testset "matrix product" begin
        # The matrix-level product of two spectra stays diagonal.
        S2 = MAK.svd_compact(matricize(randn((g,), (g,))))[2]
        P = S * S2
        @test P isa FusedGradedDiagonal
        for c in keys(sectordata(S))
            @test sectordata(P)[c] ≈ sectordata(S)[c] * sectordata(S2)[c]
        end
    end

    @testset "broadcast addition" begin
        S2 = MAK.svd_compact(matricize(randn((g,), (g,))))[2]
        # All-diagonal broadcast stays diagonal; a dense operand promotes to a fused matrix.
        r_dd = S + S2
        @test r_dd isa FusedGradedDiagonal
        @test MAK.diagview(r_dd).buffer ≈ MAK.diagview(S).buffer .+ MAK.diagview(S2).buffer
        r_lc = 2 .* S .- S2
        @test r_lc isa FusedGradedDiagonal

        M = matricize(randn((g,), (g,)))
        @test S + M isa FusedGradedMatrix
        @test Array(S + M) ≈ Array(S) + Array(M)
        @test M + S isa FusedGradedMatrix
    end

    @testset "band-wise matrix functions" begin
        # A band-wise function on the diagonal band stays diagonal.
        Sq = MAK.diagonal(map(sqrt, MAK.diagview(S)))
        @test Sq isa FusedGradedDiagonal
        @test MAK.diagview(Sq).buffer ≈ sqrt.(MAK.diagview(S).buffer)
    end
end

@testset "GradedArray wrap of a matrix-level fused operand" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    m = matricize(randn((g,), (g,)))
    wm = GradedArray(m)
    @test wm isa GradedArray
    @test matricize(wm) === m
    @test (ndims_codomain(wm), ndims_domain(wm)) == (1, 1)
    @test axes(wm) == axes(m)
    d = MAK.svd_compact(m)[2]
    wd = GradedArray(d)
    @test matricize(wd) === d
    @test (ndims_codomain(wd), ndims_domain(wd)) == (1, 1)
    @test axes(wd) == axes(d)
    # A lazy adjoint wraps lazily, sharing the parent's storage.
    ma = m'
    wa = GradedArray(ma)
    @test matricize(wa) === ma
    @test parent(matricize(wa)) === m
    @test axes(wa) == axes(ma)
    @test Array(wa) ≈ Array(m)'
end

# Factorization spectra are matrix-level fused operands (a `FusedGradedDiagonal`, or a lazy adjoint
# of a matricized factor). `contract` lifts them to their tensor-level `{1,1}` `GradedArray` wrap at
# the entry point, so every operand order works. The fermionic case is the load-bearing one: the
# lift happens before algorithm selection, so a lifted right factor still gets the contraction
# twist.
@testset "contract with matrix-level fused operands ($G)" for (G, g) in (
        ("U1", gradedrange([U1(0) => 2, U1(1) => 2])),
        ("fermion", gradedrange([fP0 => 2, fP1 => 2])),
    )
    h = project_hermitian(randn((g,), (g,)), (1,), (2,))
    d, v = eigh_full(h, (1,), (2,))
    @test d isa FusedGradedDiagonal
    vdag = matricize(v)'
    href = canonical(h, (:i, :j), [:i, :j])
    # `V D V†` with the bare spectra in every contract slot and orientation: the fused operand in
    # `a2` over its codomain leg (`vd`) and over its domain leg (`vdagd`), in `a1` (`dv`), and in
    # both slots (`dvdag`).
    vd, lvd = contract(v, (:i, :b), d, (:b, :c))
    hr1, lr1 = contract(vd, (:i, :c), vdag, (:c, :j))
    dv, ldv = contract(d, (:b, :c), v, (:i, :b))
    hr2, lr2 = contract(dv, (:c, :i), vdag, (:c, :j))
    dvdag, _ = contract(d, (:b, :c), vdag, (:c, :j))
    hr3, lr3 = contract(v, (:i, :b), dvdag, (:b, :j))
    vdagd, _ = contract(vdag, (:c, :j), d, (:b, :c))
    hr4, lr4 = contract(v, (:i, :b), vdagd, (:j, :b))
    for (hr, lr) in ((hr1, lr1), (hr2, lr2), (hr3, lr3), (hr4, lr4))
        @test hr isa GradedArray
        @test axes(hr) == axes(h)
        @test canonical(hr, lr, [:i, :j]) ≈ href
        # A dense comparison is sign-safe only without fermionic braiding.
        G == "U1" && @test Array(hr) ≈ Array(h)
    end
end

# The dense reshape equivalence is abelian-only (a non-abelian unmatricize recouples), so this
# runs on `U1` rather than inside the symmetry loop above.
@testset "unmatricize bond-split densifies a diagonal" begin
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    bond = tensor_product(g, g)
    D = MAK.svd_compact(matricize(randn((bond,), (bond,))))[2]
    rsplit = unmatricize(D, (g, g), (bond,))
    @test rsplit isa GradedArray
    @test matricize(rsplit) isa FusedGradedMatrix
    @test ndims_codomain(rsplit) == 2
    @test Array(rsplit) ≈ reshape(Array(D), length(g), length(g), length(bond))
end

@testset "dense converters route around scalar indexing" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn((g,), (g,))
    @test Matrix(a) == Array(a)
    @test Matrix(matricize(a)) == Array(matricize(a))
    @test Array{Float64}(a) == Array(a)
    @test Matrix{Float32}(a) == Float32.(Array(a))
    @test Array{Float32}(matricize(a)) == Float32.(Array(matricize(a)))
    @test Vector{Float32}(diag(matricize(a))) == Float32.(Array(diag(matricize(a))))
end

@testset "scalar rmul! and lmul!" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn((g,), (g,))
    aref = Array(a)
    rmul!(a, 2.5)
    @test Array(a) ≈ 2.5 * aref
    lmul!(-0.5, a)
    @test Array(a) ≈ -1.25 * aref
end

@testset "issquare, checksquare, isblockdiag, and diag" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    m = matricize(randn((g,), (g,)))
    @test issquare(m)
    @test isnothing(checksquare(m))
    @test isblockdiag(m)
    d = diag(m)
    @test d isa FusedGradedVector
    @test Array(d) ≈ diag(Array(m))
    @test Vector(d) == Array(d)
    @test Matrix(m) == Array(m)
    @test with_scalar_indexing(() -> d[1]) == Vector(d)[1]
    @test all(i -> GradedArrays.isstored(blocks(d), i), 1:length(blocks(d)))
    @test_throws BoundsError GradedArrays.isstored(blocks(d), length(blocks(d)) + 1)
    grect = gradedrange([U1(0) => 3, U1(1) => 3])
    mrect = matricize(randn((g,), (grect,)))
    @test !issquare(mrect)
    @test_throws DimensionMismatch checksquare(mrect)
    @test_throws DimensionMismatch diag(mrect)
end

# The single-axis and empty `fuseaxes` fast paths (a cached-field read and the trivial range)
# must agree with the general reduce-over-`tensor_product` spelling, and the fused root must
# depend only on the multiset of leaves — the order-independence that lets a contraction carry
# an operand's stored coupled axis to a permuted output. Conjugating every leaf flips the root.
@testset "fuseaxes fast paths and leaf-order independence ($G)" for (G, g, h) in (
        ("U1", gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 1, U1(1) => 2])),
        ("fermion", gradedrange([fP0 => 2, fP1 => 3]), gradedrange([fP1 => 2])),
        ("SU2", gradedrange([SU2(0) => 2, SU2(1 // 2) => 1]), gradedrange([SU2(1 // 2) => 2])),
    )
    S = GradedArrays.sectortype(g)
    init = GradedArrays.trivial_gradedrange(S)
    @test GradedArrays.fuseaxes(S, ()) == init
    for gs in ((g,), (dual(g),), (g, h), (dual(g), h), (g, dual(h), g))
        @test GradedArrays.fuseaxes(S, gs) == reduce(tensor_product, gs; init)
    end
    for (gs, gs_perm) in (((g, h), (h, g)), ((dual(g), h, g), (g, dual(g), h)))
        @test GradedArrays.fuseaxes(S, gs) == GradedArrays.fuseaxes(S, gs_perm)
    end
    # Conjugating every leaf conjugates the root's sectors; `flip` also flips the arrow, which
    # `dual` resets (a fused root is always non-dual).
    @test GradedArrays.fuseaxes(S, (conj(g), conj(h))) ==
        dual(GradedArrays.flip(GradedArrays.fuseaxes(S, (g, h))))
end

# `output_axes` carries a side's root from the operand's stored coupled axis exactly when that
# side is the operand's stored codomain/domain group (in any order); any other split falls back
# to fusing the leaves, which must give the same root for the same multiset.
@testset "contract output_axes carries stored roots" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    h = gradedrange([U1(0) => 1, U1(1) => 2])
    a = randn((g, h), (g,))
    b = randn((g,), (h, g))
    S = GradedArrays.sectortype(a)
    root_a = GradedArrays.axis_codomain(matricize(a))
    root_b = GradedArrays.axis_domain(matricize(b))
    # Identity groupings carry both sides, with the generic leaves.
    cod, dom = TensorAlgebra.output_axes(
        TensorAlgebra.contract, (1, 2), (3, 4), a, (1, 2), (3,), b, (1,), (2, 3)
    )
    @test GradedArrays.leaves(cod) == GradedArrays.axes_codomain(a)
    @test GradedArrays.leaves(dom) == GradedArrays.axes_domain(b)
    @test GradedArrays.root(cod) === root_a
    @test GradedArrays.root(dom) === root_b
    # Permutations within each group still carry (the root is order-independent).
    cod, dom = TensorAlgebra.output_axes(
        TensorAlgebra.contract, (2, 1), (4, 3), a, (2, 1), (3,), b, (1,), (3, 2)
    )
    @test GradedArrays.root(cod) === root_a
    @test GradedArrays.root(dom) === root_b
    # A group-crossing destination split falls back to fusing on both sides; the fused roots
    # must equal the carried ones on the sides whose multiset is unchanged.
    cod, dom = TensorAlgebra.output_axes(
        TensorAlgebra.contract, (1, 2, 3), (4,), a, (1, 2), (3,), b, (1,), (2, 3)
    )
    @test GradedArrays.root(cod) == GradedArrays.fuseaxes(S, GradedArrays.leaves(cod))
    @test GradedArrays.root(dom) == GradedArrays.fuseaxes(S, GradedArrays.leaves(dom))
    # Contracting part of a stored group falls back on that side.
    cod, dom = TensorAlgebra.output_axes(
        TensorAlgebra.contract, (1, 2), (3, 4), a, (1, 3), (2,), b, (1,), (2, 3)
    )
    @test GradedArrays.root(cod) == GradedArrays.fuseaxes(S, GradedArrays.leaves(cod))
    @test GradedArrays.root(dom) === root_b
end

# Whatever the grouping, the contract output's backing coupled axes must equal the fusion of its
# external leaves (a carried root agrees with the fused one), and the values must match the
# TensorKit reference. Groupings cover: both sides carried, per-operand groups permuted, a
# contracted leg inside a stored group, and a group-crossing destination.
@testset "contract carries coupled axes across groupings ($G)" for (G, g, h) in (
        ("U1", gradedrange([U1(0) => 2, U1(1) => 3]), gradedrange([U1(0) => 1, U1(1) => 2])),
        ("fermion", gradedrange([fP0 => 2, fP1 => 3]), gradedrange([fP1 => 2])),
        ("SU2", gradedrange([SU2(0) => 2, SU2(1 // 2) => 1]), gradedrange([SU2(1 // 2) => 2])),
    )
    S = GradedArrays.sectortype(g)
    a = randn((g, h), (g,))
    b = randn((g,), (h, g))
    ta = TensorKit.TensorMap(a)
    tb = TensorKit.TensorMap(b)
    function check_coupled(c)
        mc = matricize(c)
        @test GradedArrays.axis_codomain(mc) ==
            GradedArrays.fuseaxes(S, GradedArrays.axes_codomain(c))
        @test GradedArrays.axis_domain(mc) ==
            GradedArrays.fuseaxes(S, GradedArrays.axes_domain(c))
        return nothing
    end

    # Both sides carried (each side is exactly the operand's stored group).
    c, lc = contract(a, (:i, :j, :m), b, (:m, :k, :l))
    @tensor ref[i, j, k, l] := ta[i, j, m] * tb[m, k, l]
    refc = TensorKit.permute(ref, ((1, 2, 3, 4), ()))
    check_coupled(c)
    @test canonical(c, lc, [:i, :j, :k, :l]) ≈ refc

    # Permuted within each stored group (still carried).
    c, lc = contract(a, (:j, :i, :m), b, (:m, :l, :k))
    @tensor ref2[i, j, k, l] := ta[j, i, m] * tb[m, l, k]
    check_coupled(c)
    @test canonical(c, lc, [:i, :j, :k, :l]) ≈ TensorKit.permute(ref2, ((1, 2, 3, 4), ()))

    # Contracted leg inside a's stored codomain (fused fallback on that side).
    c, lc = contract(a, (:m, :j, :i), b, (:l, :k, :m))
    @tensor ref3[i, j, k, l] := ta[m, j, i] * tb[l, k, m]
    check_coupled(c)
    @test canonical(c, lc, [:i, :j, :k, :l]) ≈ TensorKit.permute(ref3, ((1, 2, 3, 4), ()))

    # Group-crossing destination split (a `b` leg lands in the destination codomain).
    c = contract((:k, :i, :j, :l), a, (:i, :j, :m), b, (:m, :k, :l))
    check_coupled(c)
    @test canonical(c, (:k, :i, :j, :l), [:i, :j, :k, :l]) ≈ refc
end

# `Base.dataids` forwards to the shared buffer, so `Base.mightalias` sees storage sharing
# through the `GradedArray`/matricized-wrapper boundary (this is what lets a contraction that
# multiplied straight into the destination's stored matrix skip the scatter-back).
@testset "dataids sees through the matricized wrapper" begin
    g = gradedrange([U1(0) => 2, U1(1) => 3])
    a = randn((g,), (g,))
    b = randn((g,), (g,))
    @test Base.mightalias(matricize(a), a)
    @test Base.mightalias(a, matricize(a))
    @test Base.mightalias(matricize(a)', a)
    @test !Base.mightalias(matricize(a), b)
    @test !Base.mightalias(a, b)
    @test !Base.mightalias(copy(matricize(a)), a)
    d = fusedgradeddiagonal([SectorRange(U1(0)) => randn(2)])
    @test Base.mightalias(d, MAK.diagview(d))
end

# Contract destinations are allocated without a `zero!` pass, so every consumer must overwrite
# them in full. The critical case is a destination whose coupled-sector set strictly contains
# the product's stored sectors (here the bond misses U1(1), which both external legs carry):
# the blocks the product never reaches must come out exactly zero, not undef garbage.
@testset "contract zero-fills the dest blocks the product misses" begin
    gext = gradedrange([U1(0) => 2, U1(1) => 3])
    gbond = gradedrange([U1(0) => 2])
    a = randn((gext,), (gbond,))
    b = randn((gbond,), (gext,))
    for _ in 1:3
        c, = contract(a, (1, -1), b, (-1, 2))
        mc = matricize(c)
        @test issetequal(collect(keys(sectordata(mc))), SectorRange.([U1(0), U1(1)]))
        @test iszero(sectordata(mc)[SectorRange(U1(1))])
        @test Array(c) ≈ Array(a) * Array(b)
    end
end

# `contractadd!` with nonzero β: `mul!` straight into the destination's stored matrix for an
# identity destination bipermutation, a seeded gather / `mul!` / scatter for a permuted one.
# Pin both against the dense reference, over a bond that misses a destination sector so β must
# also scale the blocks the product never reaches.
@testset "contractadd! with nonzero beta (identity and permuted dest)" begin
    gext = gradedrange([U1(0) => 2, U1(1) => 3])
    gbond = gradedrange([U1(0) => 2])
    a = randn((gext,), (gbond,))
    b = randn((gbond,), (gext,))
    α, β = 2.0, -3.0

    d = randn((gext,), (gext,))
    dref = Array(d)
    TensorAlgebra.contractadd!(d, (1, 2), a, (1, -1), b, (-1, 2), α, β)
    @test Array(d) ≈ α * Array(a) * Array(b) + β * dref

    c, = contract(a, (1, -1), b, (-1, 2))
    dp = TensorAlgebra.permutedims(c, (2, 1))
    randn!(matricize(dp).buffer)
    dpref = Array(dp)
    TensorAlgebra.contractadd!(dp, (2, 1), a, (1, -1), b, (-1, 2), α, β)
    @test Array(dp) ≈ α * permutedims(Array(a) * Array(b), (2, 1)) + β * dpref

    # β = 0 with a permuted destination takes the detached-product branch instead; the blocks
    # the product misses must still come out zero.
    dz = TensorAlgebra.permutedims(c, (2, 1))
    randn!(matricize(dz).buffer)
    TensorAlgebra.contractadd!(dz, (2, 1), a, (1, -1), b, (-1, 2), 1.0, 0.0)
    @test Array(dz) ≈ permutedims(Array(a) * Array(b), (2, 1))
end

# `sectordata` is backed by the array's carried sorted-vector index structure (computed once at
# construction) rather than a per-call `Dictionary` build; pin its dictionary interface (keys,
# `getindex`, `pairs`, value iteration, lookup misses) against a reference `Dictionary` built the
# old way — an intersect over the axes' sector sets plus a running-offset walk — for each storage
# variant, so a backing change cannot silently reorder, drop, or misplace blocks.
@testset "sectordata dictionary interface matches the reference build" begin
    g_cod = gradedrange([U1(0) => 2, U1(1) => 3, U1(2) => 2])
    g_dom = gradedrange([U1(0) => 2, U1(1) => 1, U1(3) => 2])   # mismatched sector sets
    m = matricize(randn((g_cod,), (g_dom,)))
    codl = GradedArrays.sectordatalengths(GradedArrays.axis_codomain(m))
    doml = GradedArrays.sectordatalengths(GradedArrays.axis_domain(m))
    coupled = sort!(intersect(collect(keys(codl)), collect(keys(doml))))
    offset = 0
    ref = dictionary(
        map(coupled) do c
            sz = (codl[c], doml[c])
            block = reshape(m.buffer[(offset + 1):(offset + prod(sz))], sz)
            offset += prod(sz)
            return c => block
        end
    )
    sd = sectordata(m)
    @test collect(keys(sd)) == collect(keys(ref))
    @test all(sd[c] == ref[c] for c in keys(ref))
    @test collect(pairs(sd)) == collect(pairs(ref))
    @test collect(sd) == collect(ref)
    @test !haskey(sd, SectorRange(U1(2)))   # codomain-only sector is not coupled
    @test !haskey(sd, SectorRange(U1(3)))   # domain-only sector is not coupled
    @test isnothing(get(sd, SectorRange(U1(9)), nothing))

    # Adjoint: same coupled sectors, each block the parent's adjoint.
    sda = sectordata(m')
    @test collect(keys(sda)) == collect(keys(ref))
    @test all(sda[c] == ref[c]' for c in keys(ref))
    @test collect(sda) == [ref[c]' for c in keys(ref)]

    # Vector: one block per axis sector, offsets the prefix sums of the data lengths.
    v = fusedgradedvector([U1(0) => randn(2), U1(1) => randn(3)])
    vref = dictionary(
        [SectorRange(U1(0)) => v.buffer[1:2], SectorRange(U1(1)) => v.buffer[3:5]]
    )
    sdv = sectordata(v)
    @test collect(keys(sdv)) == collect(keys(vref))
    @test all(sdv[c] == vref[c] for c in keys(vref))
    @test collect(pairs(sdv)) == collect(pairs(vref))

    # Diagonal: the vector blocks wrapped as `Diagonal`s.
    d = fusedgradeddiagonal([U1(0) => randn(2), U1(1) => randn(3)])
    dv = sectordata(MAK.diagview(d))
    sdd = sectordata(d)
    @test collect(keys(sdd)) == collect(keys(dv))
    @test all(sdd[c] == Diagonal(dv[c]) for c in keys(dv))
    @test collect(pairs(sdd)) == [c => Diagonal(dv[c]) for c in keys(dv)]
end

# The carried index structure is immutable metadata determined by the axes, so constructions that
# keep the axes (`similar`, `copy`) share it rather than recomputing it.
@testset "carried index structure is shared under similar/copy" begin
    g_cod = gradedrange([U1(0) => 2, U1(1) => 3])
    g_dom = gradedrange([U1(0) => 2, U1(2) => 1])
    m = matricize(randn((g_cod,), (g_dom,)))
    for m′ in (similar(m), similar(m, ComplexF64), copy(m))
        @test GradedArrays.sectordatalayout(m′) === GradedArrays.sectordatalayout(m)
    end
    v = fusedgradedvector([U1(0) => randn(2), U1(1) => randn(3)])
    for v′ in (similar(v), similar(v, ComplexF64), copy(v))
        @test GradedArrays.sectordatalayout(v′) === GradedArrays.sectordatalayout(v)
    end

    # A non-canonical (hash, unsorted) layout dictionary canonicalizes in the inner constructor;
    # passing a matrix's own carried layout back in must stay the identity (no copy).
    gc = FusedGradedOneTo(gradedrange([U1(0) => 2, U1(2) => 3]))
    gd = FusedGradedOneTo(gradedrange([U1(0) => 2, U1(2) => 1]))
    lay = dictionary(
        [
            SectorRange(U1(2)) => (offset = 4, size = (3, 1)),
            SectorRange(U1(0)) => (offset = 0, size = (2, 2)),
        ]
    )
    mc = FusedGradedMatrix(collect(1.0:7.0), gc, gd, lay)
    @test collect(keys(sectordata(mc))) == SectorRange.([U1(0), U1(2)])
    @test sectordata(mc)[SectorRange(U1(0))] == [1.0 3.0; 2.0 4.0]
    @test typeof(GradedArrays.sectordatalayout(mc)) ===
        typeof(GradedArrays.sectordatalayout(m))
    m4 = FusedGradedMatrix(mc.buffer, gc, gd, GradedArrays.sectordatalayout(mc))
    @test GradedArrays.sectordatalayout(m4) === GradedArrays.sectordatalayout(mc)
end
