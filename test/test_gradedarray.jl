using BlockArrays: Block, blocklengths
using GradedArrays: GradedArrays, FusedGradedDiagonal, FusedGradedMatrix, FusedGradedOneTo,
    FusedGradedVector, GradedArray, SU2, SectorRange, U1, UniqueSectorArray, Z2, data, dual,
    gradedrange, isdual, ndims_codomain, ndims_domain, sector, sectordata, to_tensormap,
    with_block_indexing, with_scalar_indexing
using LinearAlgebra: Diagonal, diag
using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using Random: randn!
using TensorAlgebra: TensorAlgebra, bipermutedims, contract, eig_full, eigh_full, matricize,
    project_hermitian, svd_compact
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
            FusedGradedMatrix([Diagonal(d0), Diagonal(d1)], [Z2(0), Z2(1)]), (g,), (g,)
        )
        dense = GradedArray(
            FusedGradedMatrix([Matrix(Diagonal(d0)), Matrix(Diagonal(d1))], [Z2(0), Z2(1)]),
            (g,), (g,)
        )
        p = bipermutedims(diag, (1, 2), ())
        @test p isa GradedArray
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(dense), ((1, 2), ()))
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

        S = FusedGradedDiagonal(
            FusedGradedVector([randn(2), randn(1)], [SU2(0), SU2(1 // 2)])
        )
        out = 2 .* TensorAlgebra.PermutedDims(S, (1, 2))
        @test out isa FusedGradedDiagonal
        @test diag(out) ≈ 2 .* diag(S)
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
