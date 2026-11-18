using GradedArrays: GradedArray, U1, gradedrange
using TensorAlgebra: TensorAlgebra as TA, MatricizeContract, matricize
using TensorKit: TensorKit
using TensorOperations: BufferAllocator
using Test: @test, @testset

module AlgorithmsTestUtils
    using GradedArrays: GradedArray
    using TensorAlgebra: TensorAlgebra as TA
    # Counts its calls and permutes like the default algorithm.
    struct CountingPermute <: TA.PermuteAlgorithm
        count::Base.RefValue{Int}
    end
    CountingPermute() = CountingPermute(Ref(0))
    function TA.bipermutedimsconjadd!(
            alg::CountingPermute, dest::GradedArray, isconj::Bool, src::GradedArray,
            perm_codomain, perm_domain, α::Number, β::Number
        )
        alg.count[] += 1
        return TA.bipermutedimsconjadd!(
            TA.DefaultPermuteAlgorithm(), dest, isconj, src, perm_codomain, perm_domain, α, β
        )
    end

    # Counts the dense buffers it serves and the checkpoints still open.
    mutable struct CountingAllocator
        count::Int
        depth::Int
    end
    CountingAllocator() = CountingAllocator(0, 0)
    function TA.similar_map(
            allocator::CountingAllocator, prototype::Vector, ::Type{T},
            axes_codomain::Tuple{Base.OneTo}, axes_domain::Tuple{}
        ) where {T}
        allocator.count += 1
        return TA.similar_map(prototype, T, axes_codomain, axes_domain)
    end
    TA.allocator_checkpoint!(allocator::CountingAllocator) = (allocator.depth += 1)
    function TA.allocator_reset!(allocator::CountingAllocator, checkpoint)
        allocator.depth -= 1
        return allocator
    end
end
using .AlgorithmsTestUtils: CountingAllocator, CountingPermute

const i = gradedrange([U1(0) => 2, U1(1) => 3])
const j = gradedrange([U1(0) => 1, U1(-1) => 2])
const k = gradedrange([U1(0) => 2, U1(1) => 2])

@testset "permute algorithm (U1)" begin
    a = randn((i, j), (k,))
    alg = CountingPermute()
    @test TensorKit.TensorMap(TA.permutedims(a, (3, 1), (2,); alg)) ≈
        TensorKit.permute(TensorKit.TensorMap(a), ((3, 1), (2,)))
    @test alg.count[] == 1
    @test matricize(a, (2,), (3, 1); alg) == matricize(a, (2,), (3, 1))
    @test alg.count[] == 2

    b = randn((k,), (j, i))
    labels_dest = (:m, :i, :j, :l)
    c = TA.contractalign(labels_dest, a, (:i, :j, :k), b, (:k, :l, :m))
    c_alg = TA.contractalign(
        labels_dest, a, (:i, :j, :k), b, (:k, :l, :m);
        alg = MatricizeContract(; permute_alg = alg)
    )
    @test TensorKit.TensorMap(c_alg) ≈ TensorKit.TensorMap(c)
    @test alg.count[] > 2
end

@testset "allocator (U1)" begin
    # Both operands are contracted over a leg in the middle of their storage, so both
    # matricizations copy, as does the product, which is permuted into the destination.
    a = randn((i, j), (k,))
    b = randn((k,), (j, i))
    labels_dest = (:m, :i, :k, :l)
    c = TA.contractalign(labels_dest, a, (:i, :j, :k), b, (:l, :j, :m))
    allocator = CountingAllocator()
    c_alg = TA.contractalign(
        labels_dest, a, (:i, :j, :k), b, (:l, :j, :m);
        alg = MatricizeContract(; allocator)
    )
    @test TensorKit.TensorMap(c_alg) ≈ TensorKit.TensorMap(c)
    @test allocator.count == 3
    @test allocator.depth == 0

    buffer = BufferAllocator()
    alg = MatricizeContract(; allocator = buffer)
    for _ in 1:2
        c_alg = TA.contractalign(labels_dest, a, (:i, :j, :k), b, (:l, :j, :m); alg)
        @test TensorKit.TensorMap(c_alg) ≈ TensorKit.TensorMap(c)
        @test buffer.offset == 0
    end
    @test buffer.max_offset > 0

    # A graded temporary is one dense buffer served from the allocator.
    checkpoint = TA.allocator_checkpoint!(buffer)
    m = TA.allocate_output(TA.matricizeconj, false, a, (2,), (3, 1); allocator = buffer)
    @test UInt(pointer(parent(m))) == UInt(pointer(buffer.buffer))
    @test buffer.offset == sizeof(parent(m))
    TA.allocator_reset!(buffer, checkpoint)
    @test buffer.offset == 0
end
