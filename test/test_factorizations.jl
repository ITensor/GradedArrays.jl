using BlockArrays: Block, blocksizes
using GradedArrays: U1, dual, flux, gradedrange, trivial
using LinearAlgebra: I, diag, svdvals
using MatrixAlgebraKit:
    left_orth,
    left_polar,
    lq_compact,
    lq_full,
    qr_compact,
    qr_full,
    right_orth,
    right_polar,
    svd_compact,
    svd_full,
    svd_trunc
using Test: @test, @test_broken, @testset

const elts = (Float32, Float64, ComplexF32, ComplexF64)
@testset "svd_compact (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        u, s, vᴴ = svd_compact(a)
        @test sort(diag(Matrix(s)); rev = true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
        @test u * s * vᴴ ≈ a
        @test Array(u'u) ≈ I
        @test Array(vᴴ * vᴴ') ≈ I
        @test flux(u) == trivial(flux(a))
        @test flux(s) == flux(a)
        @test flux(vᴴ) == trivial(flux(a))

        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        u, s, vᴴ = svd_compact(a)
        @test sort(diag(Matrix(s)); rev = true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
        @test u * s * vᴴ ≈ a
        @test Array(u'u) ≈ I
        @test Array(vᴴ * vᴴ') ≈ I
        @test flux(u) == trivial(flux(a))
        @test flux(s) == flux(a)
        @test flux(vᴴ) == trivial(flux(a))
    end
end

@testset "svd_full (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        u, s, vᴴ = svd_full(a)
        @test u * s * vᴴ ≈ a
        @test Array(u'u) ≈ I
        @test Array(u * u') ≈ I
        @test Array(vᴴ * vᴴ') ≈ I
        @test Array(vᴴ'vᴴ) ≈ I
        @test flux(u) == trivial(flux(a))
        @test flux(s) == flux(a)
        @test flux(vᴴ) == trivial(flux(a))

        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        u, s, vᴴ = svd_full(a)
        @test u * s * vᴴ ≈ a
        @test Array(u'u) ≈ I
        @test Array(u * u') ≈ I
        @test Array(vᴴ * vᴴ') ≈ I
        @test Array(vᴴ'vᴴ) ≈ I
        @test flux(u) == trivial(flux(a))
        @test flux(s) == flux(a)
        @test flux(vᴴ) == trivial(flux(a))
    end
end

@testset "svd_trunc (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        u, s, vᴴ = svd_trunc(a; trunc = (; maxrank = 1))
        @test sort(diag(Matrix(s)); rev = true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
        @test size(u) == (size(a, 1), 1)
        @test size(s) == (1, 1)
        @test size(vᴴ) == (1, size(a, 2))
        @test Array(u'u) ≈ I
        @test Array(vᴴ * vᴴ') ≈ I
        @test flux(u) == trivial(flux(a))
        @test flux(s) == flux(a)
        @test flux(vᴴ) == trivial(flux(a))

        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        u, s, vᴴ = svd_trunc(a; trunc = (; maxrank = 1))
        @test sort(diag(Matrix(s)); rev = true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
        @test size(u) == (size(a, 1), 1)
        @test size(s) == (1, 1)
        @test size(vᴴ) == (1, size(a, 2))
        @test Array(u'u) ≈ I
        @test Array(vᴴ * vᴴ') ≈ I
        @test flux(u) == trivial(flux(a))
        @test flux(s) == flux(a)
        @test flux(vᴴ) == trivial(flux(a))
    end
end

@testset "qr_compact, left_orth (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        for f in (qr_compact, left_orth)
            q, r = f(a)
            @test q * r ≈ a
            @test Array(q'q) ≈ I
            @test flux(q) == trivial(flux(a))
            @test flux(r) == flux(a)
        end

        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        for f in (qr_compact, left_orth)
            q, r = f(a)
            @test q * r ≈ a
            @test Array(q'q) ≈ I
            @test flux(q) == trivial(flux(a))
            @test flux(r) == flux(a)
        end
    end
end

@testset "qr_full (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        q, r = qr_full(a)
        @test q * r ≈ a
        @test Array(q'q) ≈ I
        @test Array(q * q') ≈ I
        @test flux(q) == trivial(flux(a))
        @test flux(r) == flux(a)

        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        q, r = qr_full(a)
        @test q * r ≈ a
        @test Array(q'q) ≈ I
        @test Array(q * q') ≈ I
        @test flux(q) == trivial(flux(a))
        @test flux(r) == flux(a)
    end
end

@testset "left_polar (eltype=$elt)" for elt in elts
    r1 = gradedrange([U1(0) => 3, U1(1) => 4])
    r2 = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(elt, r1, dual(r2))
    a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
    @test flux(a) == U1(0)
    q, r = left_polar(a)
    @test q * r ≈ a
    @test Array(q'q) ≈ I
    @test flux(q) == trivial(flux(a))
    @test flux(r) == flux(a)

    r1 = gradedrange([U1(0) => 3, U1(1) => 4])
    r2 = gradedrange([U1(0) => 2, U1(1) => 3])
    a = zeros(elt, r1, dual(r2))
    a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
    @test flux(a) == U1(-1)
    q, r = left_polar(a)
    @test q * r ≈ a
    @test_broken Array(q'q) ≈ I
    @test flux(q) == trivial(flux(a))
    @test flux(r) == flux(a)
end

@testset "lq_compact, right_orth (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        l, q = lq_compact(a)
        @test l * q ≈ a
        @test Array(q * q') ≈ I
        @test flux(l) == flux(a)
        @test flux(q) == trivial(flux(a))
    end
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        l, q = lq_compact(a)
        @test l * q ≈ a
        @test Array(q * q') ≈ I
        @test flux(l) == flux(a)
        @test flux(q) == trivial(flux(a))
    end
end

@testset "lq_full (eltype=$elt)" for elt in elts
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
        @test flux(a) == U1(0)
        l, q = lq_full(a)
        @test l * q ≈ a
        @test Array(q * q') ≈ I
        @test Array(q'q) ≈ I
        @test flux(l) == flux(a)
        @test flux(q) == trivial(flux(a))
    end
    for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
        r1 = gradedrange([U1(0) => i, U1(1) => j])
        r2 = gradedrange([U1(0) => k, U1(1) => l])
        a = zeros(elt, r1, dual(r2))
        a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
        @test flux(a) == U1(-1)
        l, q = lq_full(a)
        @test l * q ≈ a
        @test Array(q * q') ≈ I
        @test Array(q'q) ≈ I
        @test flux(l) == flux(a)
        @test flux(q) == trivial(flux(a))
    end
end

@testset "right_polar (eltype=$elt)" for elt in elts
    r1 = gradedrange([U1(0) => 2, U1(1) => 3])
    r2 = gradedrange([U1(0) => 3, U1(1) => 4])
    a = zeros(elt, r1, dual(r2))
    a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
    @test flux(a) == U1(0)
    l, q = right_polar(a)
    @test l * q ≈ a
    @test Array(q * q') ≈ I
    @test flux(l) == flux(a)
    @test flux(q) == trivial(flux(a))

    r1 = gradedrange([U1(0) => 2, U1(1) => 3])
    r2 = gradedrange([U1(0) => 3, U1(1) => 4])
    a = zeros(elt, r1, dual(r2))
    a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
    @test flux(a) == U1(-1)
    l, q = right_polar(a)
    @test l * q ≈ a
    @test_broken Array(q * q') ≈ I
    @test flux(l) == flux(a)
    @test flux(q) == trivial(flux(a))
end
