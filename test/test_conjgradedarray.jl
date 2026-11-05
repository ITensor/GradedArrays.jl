using GradedArrays: GradedArray, fZ2, gradedrange
using LinearAlgebra: norm
using TensorAlgebra: TensorAlgebra, ConjArray, conjed, qr_compact
using Test: @test, @testset

# A lazy conjugate of a graded array works through its parent, so it must agree with the eager
# conjugate (fermion signs included) without ever indexing elements.
@testset "lazy conjugate of a GradedArray (eltype = $elt)" for elt in (Float64, ComplexF64)
    g = gradedrange([fZ2(false) => 1, fZ2(true) => 2])
    a = randn(elt, (g, g), (g,))
    b = randn(elt, (g, g), (g,))
    c = conjed(a)
    @test c isa ConjArray
    @test axes(c) == axes(conj(a))

    @test c == conj(a)
    @test conj(a) == c
    @test c ≈ conj(a)
    @test norm(c) ≈ norm(a)
    @test Array(c) == Array(conj(a))
    @test startswith(sprint(show, MIME"text/plain"(), c), "conjed of ")

    @test 2 .* c isa GradedArray
    @test 2 .* c ≈ 2 .* conj(a)
    @test c .+ conj(b) ≈ conj(a) .+ conj(b)
    @test Base.permutedims(c, (3, 1, 2)) ≈ Base.permutedims(conj(a), (3, 1, 2))
    @test similar(c) isa GradedArray
    @test axes(similar(c)) == axes(c)

    Q, R = qr_compact(c, (1, 2), (3,))
    Qe, Re = qr_compact(conj(a), (1, 2), (3,))
    @test Q ≈ Qe
    @test R ≈ Re

    # Writes go to the parent as the conjugate of what is written.
    d = conjed(zero(a))
    @test copyto!(d, conj(b)) === d
    @test parent(d) ≈ b
    @test TensorAlgebra.zero!(d) === d
    @test iszero(norm(parent(d)))
    v = elt(2)
    @test fill!(d, v) === d
    @test copy(d) == fill!(similar(conj(a)), v)
end
