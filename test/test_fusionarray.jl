using GradedArrays: FusedGradedMatrix, FusionArray, SU2, SectorRange, U1, Z2, dual,
    gradedrange, isdual, ndims_codomain, ndims_domain
using LinearAlgebra: Diagonal
using Random: randn!
using TensorAlgebra: bipermutedims, contract, matricize, svd_compact
using TensorKit: TensorKit, @tensor
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

# Build a random `FusionArray`: allocate an undef map over the given per-leg codomain/domain
# axes (codomain-facing) and fill the reduced blocks.
function randn_fusionarray(::Type{T}, cod::Tuple, dom::Tuple) where {T}
    return randn!(FusionArray{T}(undef, cod, dom))
end
randn_fusionarray(cod::Tuple, dom::Tuple) = randn_fusionarray(Float64, cod, dom)

# `FusionArray` delegates its heavy fusion-tree work (braiding, fermion signs, recoupling) to
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

@testset "FusionArray" begin
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
        a = randn_fusionarray((i,), (j,))
        @test a isa FusionArray
        @test size(a) == (length(i), length(j))
        # Codomain axis is stored as given; the domain axis is stored dualized.
        @test !isdual(axes(a, 1))
        @test isdual(axes(a, 2))
        # Round-tripping through a `TensorMap` and back preserves the data (axes derived from `t`).
        t = TensorKit.TensorMap(a)
        b = FusionArray(t)
        @test TensorKit.TensorMap(b) ≈ t
    end

    @testset "external axes may be unfused or unsorted" begin
        ok = gradedrange([U1(0) => 2, U1(1) => 1])
        unsorted = gradedrange([U1(1) => 1, U1(0) => 2])
        unfused = gradedrange([U1(0) => 2, U1(1) => 1, U1(0) => 1])
        # The array carries unfused / unsorted external axes; the `matricized` backing stays
        # fused-sorted, so the per-leg sort permutation relates the two.
        @test FusionArray{Float64}(undef, (unsorted,), (ok,)) isa FusionArray
        @test FusionArray{Float64}(undef, (ok,), (unfused,)) isa FusionArray
        # The `TensorMap` / `ElementarySpace` conversion stays strict: it expects a fused-sorted range
        # (callers normalize with `sectormergesort` at the boundary).
        @test_throws ArgumentError TensorKit.ElementarySpace(unsorted)
        @test_throws ArgumentError TensorKit.ElementarySpace(unfused)
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
        a = randn_fusionarray(ComplexF64, (i,), (j,))
        ra = real(a)
        ia = imag(a)
        @test ra isa FusionArray
        @test ia isa FusionArray
        @test eltype(ra) == Float64
        @test axes(ra) == axes(a)
        # Forwarded to the matricized fused matrix, so real/imag act block-wise on the reduced data.
        ma = matricize(a)
        for c in keys(ma.blocks)
            @test matricize(ra).blocks[c] == real.(ma.blocks[c])
            @test matricize(ia).blocks[c] == imag.(ma.blocks[c])
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
        a = randn_fusionarray(ComplexF64, cod, dom)
        c = conj(a)
        @test c isa FusionArray
        # Split preserved (unlike the `conj.(a)` broadcast, which materializes all-codomain), per-leg
        # axes dualized.
        @test (ndims_codomain(c), ndims_domain(c)) == (ndims_codomain(a), ndims_domain(a))
        @test axes(c) == map(dual, axes(a))
        # Same tensor as the broadcast conj and an involution, compared at a common split via `≈` (a
        # non-abelian double conj picks up recoupling round-off, so `==` is too strict).
        @test c ≈ conj.(a)
        @test conj(c) ≈ a
    end

    @testset "conj of a rank-0 FusionArray" begin
        a = randn!(FusionArray{ComplexF64, U1}(undef, (), ()))
        c = conj(a)
        @test c isa FusionArray
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
        m1 = randn_fusionarray((i,), (k,))
        m2 = randn_fusionarray((k,), (j,))
        c2, = contract(m1, (:i, :k), m2, (:k, :j))
        @test c2 isa FusionArray
        @test TensorKit.TensorMap(c2) ≈ TensorKit.TensorMap(m1) * TensorKit.TensorMap(m2)

        # 3-leg over two shared indices: the free/contracted split differs from the stored
        # split, so this exercises the leg bend in `matricize`.
        a = randn_fusionarray((i, j), (k,))          # (i,j; k)
        b = randn_fusionarray((k,), (j, l))          # (k; j,l)
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
        a = randn_fusionarray((i, j), (k,))
        # Move a domain leg into the codomain: a braid + bend that TensorKit handles.
        p = bipermutedims(a, (1, 3), (2,))
        @test p isa FusionArray
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(a), ((1, 3), (2,)))
    end

    @testset "bend of a Diagonal-blocked FusionArray" begin
        # `Diagonal` blocks arise from factorization factors (e.g. the singular values of a gauge).
        # A non-trivial bend reads them through the same path TensorKit uses for `DiagonalTensorMap`,
        # so it must match the dense-blocked equivalent.
        g = gradedrange([Z2(0) => 2, Z2(1) => 3])
        d0, d1 = randn(2), randn(3)
        diag = FusionArray(
            FusedGradedMatrix([Z2(0), Z2(1)], [Diagonal(d0), Diagonal(d1)]), (g,), (g,)
        )
        dense = FusionArray(
            FusedGradedMatrix([Z2(0), Z2(1)], [Matrix(Diagonal(d0)), Matrix(Diagonal(d1))]),
            (g,), (g,)
        )
        p = bipermutedims(diag, (1, 2), ())
        @test p isa FusionArray
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(dense), ((1, 2), ()))
    end

    @testset "factorization (svd_compact)" begin
        i = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 1, SU2(1 // 2) => 2])
        m = randn_fusionarray((i,), (j,))
        u, s, v = svd_compact(m, (1,), (2,))
        @test all(x -> x isa FusionArray, (u, s, v))
        us, = contract(u, (:i, :b), s, (:b, :c))
        rec, = contract(us, (:i, :c), v, (:c, :j))
        @test TensorKit.TensorMap(rec) ≈ TensorKit.TensorMap(m)
    end

    @testset "broadcasting (linear combinations)" begin
        i = gradedrange([SU2(0) => 1, SU2(1 // 2) => 1])
        j = gradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        a = randn_fusionarray((i,), (j,))
        b = randn_fusionarray((i,), (j,))
        # Linear combinations move all axes to the codomain, so normalize back to a `(i; j)`
        # `TensorMap` before comparing.
        back(x) = TensorKit.permute(TensorKit.TensorMap(x), ((1,), (2,)))
        @test a + b isa FusionArray
        @test back(a + b) ≈ back(a) + back(b)
        @test back(a - b) ≈ back(a) - back(b)
        @test back(2 * a - 3 * b) ≈ 2 * back(a) - 3 * back(b)
        # Operands with different codomain/domain splits but equal axes still add (each is bent).
        c = randn_fusionarray((i, dual(j)), ())
        @test axes(c) == axes(a)
        @test back(a + c) ≈ back(a) + back(c)
    end

    @testset "fermionic" begin
        i = gradedrange([fP0 => 2, fP1 => 1])
        j = gradedrange([fP0 => 1, fP1 => 2])
        k = gradedrange([fP0 => 1, fP1 => 1])
        l = gradedrange([fP0 => 2, fP1 => 1])

        # Fermion signs on a permute ride `tensoradd!` for free.
        a3 = randn_fusionarray((i, j), (k,))
        p = bipermutedims(a3, (2, 1), (3,))
        @test TensorKit.TensorMap(p) ≈
            TensorKit.permute(TensorKit.TensorMap(a3), ((2, 1), (3,)))

        # The contraction twist: 2-leg matches TensorKit composition directly.
        m1 = randn_fusionarray((i,), (k,))
        m2 = randn_fusionarray((k,), (j,))
        c2, = contract(m1, (:i, :k), m2, (:k, :j))
        @test TensorKit.TensorMap(c2) ≈ TensorKit.TensorMap(m1) * TensorKit.TensorMap(m2)

        # Multi-leg fermionic contraction matches `@tensor`, and is independent of operand
        # order — the property the twist exists to guarantee.
        a = randn_fusionarray((i, j), (k,))          # (i,j; k)
        b = randn_fusionarray((k,), (j, l))          # (k; j,l)
        ta = TensorKit.TensorMap(a)
        tb = TensorKit.TensorMap(b)
        c1, lc1 = contract(a, (:i, :j, :k), b, (:k, :j, :l))
        c2, lc2 = contract(b, (:k, :j, :l), a, (:i, :j, :k))
        @tensor ref[i, l] := ta[i, j, k] * tb[k, j, l]
        @test canonical(c1, lc1, [:i, :l]) ≈ ref
        @test canonical(c1, lc1, [:i, :l]) ≈ canonical(c2, lc2, [:i, :l])
    end
end
