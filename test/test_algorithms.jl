using GradedArrays: GradedArray, U1, gradedrange
using MatrixAlgebraKit: truncrank
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

@testset "factorizations with a permute algorithm and an allocator (U1)" begin
    # A Hermitian positive semi-definite map from dimensions `(2, 1)` to `(3, 4)`, so every
    # factorization applies and the matricization permutes.
    m = matricize(randn((i, j), (i, j)))
    a = TA.permutedims(GradedArray(m * m', (i, j), (i, j)), (2, 1), (3, 4))
    perm_codomain, perm_domain = (2, 1), (3, 4)
    trunc = truncrank(3)
    fs_kwargs = (
        (TA.qr_compact, (;)), (TA.qr_full, (;)), (TA.lq_compact, (;)),
        (TA.lq_full, (;)),
        (TA.left_polar, (;)), (TA.right_polar, (;)),
        (TA.left_orth, (;)), (TA.right_orth, (;)),
        (TA.svd_compact, (;)), (TA.svd_full, (;)), (TA.svd_trunc, (; trunc)),
        (TA.svd_vals, (;)),
        (TA.eigh_full, (;)), (TA.eig_full, (;)), (TA.eigh_trunc, (; trunc)),
        (TA.eig_trunc, (; trunc)), (TA.eigh_vals, (;)), (TA.eig_vals, (;)),
        (TA.left_null, (;)), (TA.right_null, (;)), (TA.project_hermitian, (;)),
        (TA.sqrth_safe, (;)), (TA.invsqrth_safe, (;)), (TA.sqrth_invsqrth_safe, (;)),
    )
    tocompare(x) = x isa GradedArray ? TensorKit.TensorMap(x) : x
    issame(F1, F2) = all(map((x, y) -> tocompare(x) ≈ tocompare(y), F1, F2))
    @testset "$f" for (f, kwargs) in fs_kwargs
        F = f(a, perm_codomain, perm_domain; kwargs...)
        F = F isa Tuple ? F : (F,)

        permute_alg = CountingPermute()
        F_alg = f(a, perm_codomain, perm_domain; permute_alg, kwargs...)
        @test issame(F_alg isa Tuple ? F_alg : (F_alg,), F)
        @test permute_alg.count[] == 1

        buffer = BufferAllocator()
        # The first call sizes the buffer, the second is served from it.
        for _ in 1:2
            F_buffer = f(a, perm_codomain, perm_domain; allocator = buffer, kwargs...)
            @test issame(F_buffer isa Tuple ? F_buffer : (F_buffer,), F)
            @test buffer.offset == 0
        end
        # `project_hermitian` returns its input, so it does not take it from the allocator.
        @test (buffer.max_offset > 0) == (f !== TA.project_hermitian)
        F_buffer = f(a, perm_codomain, perm_domain; allocator = buffer, kwargs...)
        F_buffer = F_buffer isa Tuple ? F_buffer : (F_buffer,)
        # Overwrite the released matricized input: no factor may share memory with it.
        checkpoint = TA.allocator_checkpoint!(buffer)
        a_mat = TA.allocate_output(
            TA.matricizeconj, false, a, perm_codomain, perm_domain; allocator = buffer
        )
        fill!(parent(a_mat), NaN)
        TA.allocator_reset!(buffer, checkpoint)
        @test issame(F_buffer, F)
    end
end
