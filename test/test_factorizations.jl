using BlockArrays: Block, blocksizes
using GradedArrays: U1, dual, flux, gradedrange
using LinearAlgebra: I, diag, svdvals
using MatrixAlgebraKit: svd_compact, svd_full, svd_trunc
using Test: @test, @testset

const elts = (Float32, Float64, ComplexF32, ComplexF64)
@testset "svd_compact (eltype=$elt)" for elt in elts
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    r1 = gradedrange([U1(0) => i, U1(1) => j])
    r2 = gradedrange([U1(0) => k, U1(1) => l])
    a = zeros(elt, r1, dual(r2))
    a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
    @test flux(a) == U1(0)
    u, s, vᴴ = svd_compact(a)
    @test sort(diag(Matrix(s)); rev=true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
    @test u * s * vᴴ ≈ a
    @test Array(u'u) ≈ I
    @test Array(vᴴ * vᴴ') ≈ I
    @test flux(u) == U1(0)
    @test flux(s) == flux(a)
    @test flux(vᴴ) == U1(0)

    r1 = gradedrange([U1(0) => i, U1(1) => j])
    r2 = gradedrange([U1(0) => k, U1(1) => l])
    a = zeros(elt, r1, dual(r2))
    a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
    @test flux(a) == U1(-1)
    u, s, vᴴ = svd_compact(a)
    @test sort(diag(Matrix(s)); rev=true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
    @test u * s * vᴴ ≈ a
    @test Array(u'u) ≈ I
    @test Array(vᴴ * vᴴ') ≈ I
    @test flux(u) == U1(0)
    @test flux(s) == flux(a)
    @test flux(vᴴ) == U1(0)
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
    @test flux(u) == U1(0)
    @test flux(s) == flux(a)
    @test flux(vᴴ) == U1(0)

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
    @test flux(u) == U1(0)
    @test flux(s) == flux(a)
    @test flux(vᴴ) == U1(0)
  end
end

@testset "svd_trunc (eltype=$elt)" for elt in elts
  for i in [2, 3], j in [2, 3], k in [2, 3], l in [2, 3]
    r1 = gradedrange([U1(0) => i, U1(1) => j])
    r2 = gradedrange([U1(0) => k, U1(1) => l])
    a = zeros(elt, r1, dual(r2))
    a[Block(2, 2)] = randn(elt, blocksizes(a)[2, 2])
    @test flux(a) == U1(0)
    u, s, vᴴ = svd_trunc(a; trunc=(; maxrank=1))
    @test sort(diag(Matrix(s)); rev=true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
    @test size(u) == (size(a, 1), 1)
    @test size(s) == (1, 1)
    @test size(vᴴ) == (1, size(a, 2))
    @test Array(u'u) ≈ I
    @test Array(vᴴ * vᴴ') ≈ I
    @test flux(u) == U1(0)
    @test flux(s) == flux(a)
    @test flux(vᴴ) == U1(0)

    r1 = gradedrange([U1(0) => i, U1(1) => j])
    r2 = gradedrange([U1(0) => k, U1(1) => l])
    a = zeros(elt, r1, dual(r2))
    a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])
    @test flux(a) == U1(-1)
    u, s, vᴴ = svd_trunc(a; trunc=(; maxrank=1))
    @test sort(diag(Matrix(s)); rev=true) ≈ svdvals(Matrix(a))[1:size(s, 1)]
    @test size(u) == (size(a, 1), 1)
    @test size(s) == (1, 1)
    @test size(vᴴ) == (1, size(a, 2))
    @test Array(u'u) ≈ I
    @test Array(vᴴ * vᴴ') ≈ I
    @test flux(u) == U1(0)
    @test flux(s) == flux(a)
    @test flux(vᴴ) == U1(0)
  end
end
