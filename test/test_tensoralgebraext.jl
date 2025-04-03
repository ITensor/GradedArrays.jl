using BlockArrays: Block, blocksize
using BlockSparseArrays: BlockSparseArray
using GradedArrays: GradedOneTo, blocklabels, dual, gradedrange
using GradedArrays.SymmetrySectors: U1
using Random: randn!
using TensorAlgebra: contract, fusedims, splitdims
using Test: @test, @test_broken, @testset

function randn_blockdiagonal(elt::Type, axes::Tuple)
  a = BlockSparseArray{elt}(undef, axes)
  blockdiaglength = minimum(blocksize(a))
  for i in 1:blockdiaglength
    b = Block(ntuple(Returns(i), ndims(a)))
    a[b] = randn!(a[b])
  end
  return a
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "`contract` `GradedArray` (eltype=$elt)" for elt in elts
  @testset "GradedOneTo with U(1)" begin
    d = gradedrange([U1(0) => 2, U1(1) => 3])
    a1 = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
    a2 = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
    a3 = randn_blockdiagonal(elt, (d, dual(d)))
    a1_dense = convert(Array, a1)
    a2_dense = convert(Array, a2)
    a3_dense = convert(Array, a3)

    # matrix matrix
    a_dest, dimnames_dest = contract(a1, (1, -1, 2, -2), a2, (2, -3, 1, -4))
    a_dest_dense, dimnames_dest_dense = contract(
      a1_dense, (1, -1, 2, -2), a2_dense, (2, -3, 1, -4)
    )
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense

    # matrix vector
    @test_broken a_dest, dimnames_dest = contract(a1, (2, -1, -2, 1), a3, (1, 2))
    #=
    a_dest_dense, dimnames_dest_dense = contract(a1_dense, (2, -1, -2, 1), a3_dense, (1, 2))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense
    =#

    #  vector matrix
    a_dest, dimnames_dest = contract(a3, (1, 2), a1, (2, -1, -2, 1))
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a1_dense, (2, -1, -2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense

    # vector vector
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (2, 1))
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (2, 1))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray{elt,0}
    @test a_dest ≈ a_dest_dense

    # outer product
    a_dest, dimnames_dest = contract(a3, (1, 2), a3, (3, 4))
    a_dest_dense, dimnames_dest_dense = contract(a3_dense, (1, 2), a3_dense, (3, 4))
    @test dimnames_dest == dimnames_dest_dense
    @test size(a_dest) == size(a_dest_dense)
    @test a_dest isa BlockSparseArray
    @test a_dest ≈ a_dest_dense
  end
  @testset "fusedims" begin
    d1 = gradedrange([U1(0) => 1, U1(1) => 1])
    d2 = gradedrange([U1(0) => 1, U1(1) => 1])
    a = randn_blockdiagonal(elt, (d1, d2, d1, d2))
    m = fusedims(a, (1, 2), (3, 4))
    for ax in axes(m)
      @test ax isa GradedOneTo
      @test blocklabels(ax) == [U1(0), U1(1), U1(2)]
    end
    for I in CartesianIndices(m)
      if I ∈ CartesianIndex.([(1, 1), (4, 4)])
        @test !iszero(m[I])
      else
        @test iszero(m[I])
      end
    end
    @test a[1, 1, 1, 1] == m[1, 1]
    @test a[2, 2, 2, 2] == m[4, 4]
    @test blocksize(m) == (3, 3)
    @test a == splitdims(m, (d1, d2), (d1, d2))

    # check block fusing and splitting
    d = gradedrange([U1(0) => 2, U1(1) => 1])
    a = randn_blockdiagonal(elt, (d, d, dual(d), dual(d)))
    @test splitdims(fusedims(a, (1, 2), (3, 4)), axes(a)...) == a
  end
end
