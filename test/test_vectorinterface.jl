using BlockArrays: Block
using GradedArrays: AbelianSectorArray, FusedGradedMatrix, FusedGradedVector, SU2,
    SectorMatrix, SectorVector, U1, data, gradedrange
using LinearAlgebra: dot, norm
using Random: randn!
using TensorAlgebra: TensorAlgebra
using Test: @test, @test_throws, @testset
using VectorInterface: VectorInterface

@testset "VectorInterface scale" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 2, U1(-1) => 3])
    a = randn(g1, g2)
    a0 = copy(a)

    @test VectorInterface.scalartype(a) === Float64

    # Out-of-place `scale` leaves the input untouched.
    c = VectorInterface.scale(a, 2)
    @test c ≈ 2 .* a0
    @test a ≈ a0

    # In-place `scale!` on the same array is the block-wise path, so self-aliasing is fine.
    VectorInterface.scale!(a, 2)
    @test a ≈ 2 .* a0

    a = copy(a0)
    VectorInterface.scale!!(a, 3)
    @test a ≈ 3 .* a0

    # Two-argument `scale!` writes `α * a` into a distinct destination.
    b = zeros(Float64, axes(a0)...)
    VectorInterface.scale!(b, a0, 4)
    @test b ≈ 4 .* a0
end

@testset "VectorInterface add" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 2, U1(-1) => 3])
    a = randn(g1, g2)
    b = randn(g1, g2)
    a0 = copy(a)
    b0 = copy(b)

    z = VectorInterface.add(a, b, 2, 3)
    @test z ≈ 3 .* a0 .+ 2 .* b0
    @test a ≈ a0

    VectorInterface.add!(a, b, 2, 3)
    @test a ≈ 3 .* a0 .+ 2 .* b0

    a = copy(a0)
    VectorInterface.add!!(a, b, 2, 3)
    @test a ≈ 3 .* a0 .+ 2 .* b0
end

@testset "VectorInterface widening" begin
    # A real array scaled/combined by a complex coefficient widens to a complex result. The
    # out-of-place `scale`/`add` allocate the promoted element type, and the `!!` variants fall
    # back to them (returning a new object, leaving the destination untouched) when the promoted
    # type does not fit.
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 2, U1(-1) => 3])
    a = randn(g1, g2)
    b = randn(g1, g2)
    a0 = copy(a)

    c = VectorInterface.scale(a, 2im)
    @test eltype(c) === ComplexF64
    @test c ≈ 2im .* a0

    d = VectorInterface.add(a, b, 1im, 2)
    @test eltype(d) === ComplexF64
    @test d ≈ 2 .* a0 .+ 1im .* b

    e = VectorInterface.scale!!(a, 2im)
    @test eltype(e) === ComplexF64
    @test e ≈ 2im .* a0
    @test a ≈ a0

    f = VectorInterface.add!!(a, b, 1im, 2)
    @test eltype(f) === ComplexF64
    @test f ≈ 2 .* a0 .+ 1im .* b
    @test a ≈ a0
end

@testset "VectorInterface inner and zerovector" begin
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 2, U1(-1) => 3])
    a = randn(g1, g2)
    b = randn(g1, g2)

    @test VectorInterface.inner(a, b) == dot(a, b)

    z = VectorInterface.zerovector(a, Float64)
    @test iszero(z)
    @test axes(z) == axes(a)

    a2 = copy(a)
    VectorInterface.zerovector!(a2)
    @test iszero(a2)

    # `zerovector!!(x, S)` recycles when `S` matches the scalar type and widens otherwise.
    w = VectorInterface.zerovector!!(copy(a), ComplexF64)
    @test iszero(w)
    @test VectorInterface.scalartype(w) === ComplexF64
    @test VectorInterface.scalartype(VectorInterface.zerovector!!(copy(a), Float64)) ===
        Float64
end

@testset "VectorInterface on sector arrays ($(typeof(a0).name.name))" for (a0, b0) in (
        let s = SU2(1)
            (SectorMatrix{Float64}(undef, s, 2, 3), SectorMatrix{Float64}(undef, s, 2, 3))
        end,
        let s = SU2(1)
            (SectorVector{Float64}(undef, s, 4), SectorVector{Float64}(undef, s, 4))
        end,
    )
    randn!(a0)
    randn!(b0)

    @test VectorInterface.scalartype(a0) === Float64
    @test iszero(data(VectorInterface.zerovector(a0, Float64)))
    @test data(VectorInterface.scale(a0, 2)) ≈ 2 .* data(a0)
    @test data(VectorInterface.add(a0, b0, 2, 3)) ≈ 3 .* data(a0) .+ 2 .* data(b0)
    @test VectorInterface.inner(a0, b0) ≈ dot(a0, b0)

    # Scaling a real block by a complex coefficient widens to a complex result.
    c = VectorInterface.scale(a0, 2im)
    @test VectorInterface.scalartype(c) === ComplexF64
    @test data(c) ≈ 2im .* data(a0)
end

@testset "graded dot/norm factorize block-wise ($(typeof(a).name.name))" for (a, b) in (
        (
            FusedGradedMatrix{Float64}(undef, [SU2(0) => 2, SU2(1) => 3]),
            FusedGradedMatrix{Float64}(undef, [SU2(0) => 2, SU2(1) => 3]),
        ),
        (
            FusedGradedVector{Float64}(undef, [SU2(0) => 2, SU2(1) => 3]),
            FusedGradedVector{Float64}(undef, [SU2(0) => 2, SU2(1) => 3]),
        ),
    )
    randn!(a)
    randn!(b)
    # The block-wise sum (each block weighted by its sector's quantum dimension) equals the dense
    # inner product, and the block `p`-norm matches the dense norm for every `p` (`Inf` included).
    @test dot(a, b) ≈ dot(Array(a), Array(b))
    @test VectorInterface.inner(a, b) == dot(a, b)
    for p in (1, 2, 3, Inf)
        @test norm(a, p) ≈ norm(Array(a), p)
    end
end

@testset "aliased trivial permute-add is a scale" begin
    # A self-aliased permute-add with the identity permutation and no conjugation is really a
    # scale (`y = α*y + β*y = (α+β)*y`), so it routes to the block-wise `scale!` rather than the
    # fused permute-add path. This keeps `a .*= 2` and self-aliased `add!` working.
    g1 = gradedrange([U1(0) => 2, U1(1) => 3])
    g2 = gradedrange([U1(0) => 2, U1(-1) => 3])
    a = randn(g1, g2)
    a0 = copy(a)
    a .*= 2
    @test a ≈ 2 .* a0

    a = copy(a0)
    TensorAlgebra.add!(a, a, 2, 3)
    @test a ≈ 5 .* a0
end

@testset "aliased nontrivial permute-add errors" begin
    # A nontrivial aliased permute-add can't run in place: the fused path overwrites the output
    # before reading the input, so it refuses the aliased destination rather than silently
    # corrupting the result, matching `TensorOperations`.
    g = gradedrange([U1(0) => 1, U1(1) => 1])
    b = zeros(Float64, g, g)
    b[Block(1, 1)] = AbelianSectorArray((U1(0), U1(0)), randn(Float64, 1, 1))
    @test_throws ArgumentError TensorAlgebra.permutedims!(b, b, (2, 1))
end
