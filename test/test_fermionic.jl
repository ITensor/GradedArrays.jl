import GradedArrays
using BlockArrays: Block, blocksize
using BlockSparseArrays: BlockSparseArray
using GradedArrays: GradedArray, SectorArray, SectorDelta, SectorRange, dual, flip,
    gradedrange, isdual, sector_type, sectorrange, sectors
using Random: randn!
using TensorAlgebra: contract, matricize, unmatricize
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

# Fermionic sector aliases: FermionParity has two values, even (false) and odd (true),
# both with quantum dimension 1.
const fP0 = SectorRange(TKS.FermionParity(false))  # even parity
const fP1 = SectorRange(TKS.FermionParity(true))   # odd parity

function randn_blockdiagonal(elt::Type, axes::Tuple)
    a = BlockSparseArray{elt}(undef, axes)
    blockdiaglength = minimum(blocksize(a))
    for i in 1:blockdiaglength
        b = Block(ntuple(Returns(i), ndims(a)))
        a[b] = randn!(a[b])
    end
    return a
end

@testset "masked_inversion_parity" begin
    mip = GradedArrays.masked_inversion_parity

    # No odd sectors: no phase regardless of permutation
    @test mip((false, false), (2, 1)) == 1
    @test mip((false, false, false), (3, 2, 1)) == 1

    # Identity permutation: always +1
    @test mip((true, true), (1, 2)) == 1
    @test mip((true, true, true), (1, 2, 3)) == 1

    # One odd, one even: swap gives +1 (no odd-odd pair)
    @test mip((true, false), (2, 1)) == 1
    @test mip((false, true), (2, 1)) == 1

    # Two odd sectors: swap gives -1 (one odd-odd inversion)
    @test mip((true, true), (2, 1)) == -1

    # Three odd sectors
    # (2,1,3): inversion at (1,2) only → -1
    @test mip((true, true, true), (2, 1, 3)) == -1
    # (1,3,2): inversion at (2,3) only → -1
    @test mip((true, true, true), (1, 3, 2)) == -1
    # (2,3,1): inversions at (1,3) and (2,3) → even count → +1
    @test mip((true, true, true), (2, 3, 1)) == 1
    # (3,2,1): inversions at (1,2), (1,3), (2,3) → odd count → -1
    @test mip((true, true, true), (3, 2, 1)) == -1

    # Two odd, one even: only odd-odd pairs count
    # (true, false, true), perm (3,2,1):
    #   i=1,j=2: mask[2]=false → skip
    #   i=1,j=3: mask[1]&mask[3]=true, perm[1]=3 > perm[3]=1 → flip
    # → one inversion → -1
    @test mip((true, false, true), (3, 2, 1)) == -1
    # (true, false, true), perm (1,2,3): identity → +1
    @test mip((true, false, true), (1, 2, 3)) == 1
    # (true, false, true), perm (3,1,2):
    #   i=1,j=3: mask[1]&mask[3]=true, perm[1]=3 > perm[3]=2 → flip
    # → -1
    @test mip((true, false, true), (3, 1, 2)) == -1
end

@testset "permutation_phase" begin
    pp = GradedArrays.fermion_permutation_phase

    # Bosonic sector (U1): always +1 regardless of permutation
    u0 = SectorRange(TKS.U1Irrep(0))
    u1 = SectorRange(TKS.U1Irrep(1))
    d_bos = SectorDelta{Float64}((u0, u1))
    @test pp(d_bos, (2, 1)) == 1
    @test pp(d_bos, (1, 2)) == 1

    # Even parity only: always +1
    d_even = SectorDelta{Float64}((fP0, fP0))
    @test pp(d_even, (2, 1)) == 1
    @test pp(d_even, (1, 2)) == 1

    # Two odd sectors: identity = +1, swap = -1
    d_odd = SectorDelta{Float64}((fP1, fP1))
    @test pp(d_odd, (1, 2)) == 1
    @test pp(d_odd, (2, 1)) == -1

    # Mixed even/odd: swap gives +1 (only odd-odd pairs contribute)
    d_mix = SectorDelta{Float64}((fP0, fP1))
    @test pp(d_mix, (2, 1)) == 1

    d_mix2 = SectorDelta{Float64}((fP1, fP0))
    @test pp(d_mix2, (2, 1)) == 1

    # Three odd sectors
    d3 = SectorDelta{Float64}((fP1, fP1, fP1))
    @test pp(d3, (1, 2, 3)) == 1
    @test pp(d3, (2, 3, 1)) == 1   # even number of odd-odd crossings
    @test pp(d3, (2, 1, 3)) == -1  # odd number of odd-odd crossings
    @test pp(d3, (3, 2, 1)) == -1
end

@testset "permutedims on fermionic SectorArray" begin
    # Two odd sectors: swap picks up -1 phase
    sa = SectorArray((fP1, fP1), fill(3.0, 1, 1))
    sp = permutedims(sa, (2, 1))
    @test sp[1, 1] ≈ -3.0
    # sectors are also permuted (both are fP1, so same)
    @test sectors(sp) == (fP1, fP1)

    # Two even sectors: swap gives no phase
    sa_even = SectorArray((fP0, fP0), fill(3.0, 1, 1))
    sp_even = permutedims(sa_even, (2, 1))
    @test sp_even[1, 1] ≈ 3.0

    # Mixed (even, odd): swap gives no phase
    sa_mix = SectorArray((fP0, fP1), fill(3.0, 1, 1))
    sp_mix = permutedims(sa_mix, (2, 1))
    @test sp_mix[1, 1] ≈ 3.0
    @test sectors(sp_mix) == (fP1, fP0)

    # Double permutation recovers original (phase squares to 1)
    @test permutedims(permutedims(sa, (2, 1)), (2, 1))[1, 1] ≈ sa[1, 1]

    # Three-index: cyclic permutation of 3 odd sectors → even crossings → +1
    sa3 = SectorArray((fP1, fP1, fP1), fill(2.0, 1, 1, 1))
    @test permutedims(sa3, (2, 3, 1))[1, 1, 1] ≈ 2.0   # even number of crossings
    @test permutedims(sa3, (3, 2, 1))[1, 1, 1] ≈ -2.0  # odd number of crossings
    @test permutedims(sa3, (2, 1, 3))[1, 1, 1] ≈ -2.0  # one crossing

    # Two odd sectors with non-unit data: verify value propagates
    sa_val = SectorArray((fP1, fP1), fill(7.5, 1, 1))
    @test permutedims(sa_val, (2, 1))[1, 1] ≈ -7.5

    # No phase for bosonic (U1) sectors even though same permutation
    u1 = SectorRange(TKS.U1Irrep(1))
    sa_u1 = SectorArray((u1, u1), fill(3.0, 1, 1))
    sp_u1 = permutedims(sa_u1, (2, 1))
    @test sp_u1[1, 1] ≈ 3.0
end

@testset "matricize/unmatricize round-trip for fermionic SectorArray" begin
    # Even dual codomain leg: twist = +1, no phase modification
    sa_even_dual = SectorArray((dual(fP0), fP0), fill(5.0, 1, 1))
    @test isdual(axes(sa_even_dual, 1))
    m_even = matricize(sa_even_dual, (1,), (2,))
    @test m_even[1, 1] ≈ 5.0  # no phase

    rt_even = unmatricize(m_even, (axes(sa_even_dual, 1),), (axes(sa_even_dual, 2),))
    @test rt_even[1, 1] ≈ 5.0  # round-trip preserves value

    # Non-dual codomain leg: twist not applied regardless of parity
    sa_odd_nondual = SectorArray((fP1, fP1), fill(5.0, 1, 1))
    @test !isdual(axes(sa_odd_nondual, 1))
    m_nondual = matricize(sa_odd_nondual, (1,), (2,))
    @test m_nondual[1, 1] ≈ 5.0  # no twist since not dual

    # Odd dual codomain leg: twist = -1 → phase -1 applied during matricize
    sa_odd_dual = SectorArray((dual(fP1), fP1), fill(5.0, 1, 1))
    @test isdual(axes(sa_odd_dual, 1))
    m_odd = matricize(sa_odd_dual, (1,), (2,))
    @test m_odd[1, 1] ≈ -5.0  # twist -1 applied to odd dual codomain leg

    # Round-trip: unmatricize applies same twist → -1 * (-5.0) = 5.0
    rt_odd = unmatricize(m_odd, (axes(sa_odd_dual, 1),), (axes(sa_odd_dual, 2),))
    @test rt_odd[1, 1] ≈ 5.0

    # Two odd dual codomain legs: phase = (-1) * (-1) = +1 (no net modification)
    sa_two_odd_dual = SectorArray((dual(fP1), dual(fP1), fP1), fill(3.0, 1, 1, 1))
    m_two = matricize(sa_two_odd_dual, (1, 2), (3,))
    @test m_two[1, 1] ≈ 3.0  # phase = -1 * -1 = +1

    rt_two = unmatricize(
        m_two,
        (axes(sa_two_odd_dual, 1), axes(sa_two_odd_dual, 2)),
        (axes(sa_two_odd_dual, 3),)
    )
    @test rt_two[1, 1, 1] ≈ 3.0
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})

# Tests with r_odd (all-odd sectors) and analytically compute the signs
# For all-odd blocks, the total phase from permutation + matricize twist is -1,
# so the GradedArray result equals -1 * the dense result.
@testset "contract GradedArray with FermionParity (eltype=$elt)" for elt in
    elts

    r_odd = gradedrange([fP1 => 2])

    @testset "permutedims of all-odd GradedArray picks up -1 phase" begin
        # (r_odd, dual(r_odd)): swap perm (2,1) on two odd axes → phase -1
        a = randn_blockdiagonal(elt, (r_odd, dual(r_odd)))
        a_perm = permutedims(a, (2, 1))
        a_dense_perm = permutedims(convert(Array, a), (2, 1))
        @test convert(Array, a_perm) ≈ -1 * a_dense_perm
    end

    @testset "matrix-matrix contraction" begin
        for r1 in (r_odd, dual(r_odd)),
                r2 in (r_odd, dual(r_odd)),
                r3 in (r_odd, dual(r_odd))

            a1 = randn_blockdiagonal(elt, (r1, dual(r2)))
            a2 = randn_blockdiagonal(elt, (r2, r3))
            a1_dense = convert(Array, a1)
            a2_dense = convert(Array, a2)
            a_dest, labels_dest = contract(a1, (-1, 1), a2, (1, -2))
            a_dest_dense, labels_dest_dense = contract(a1_dense, (-1, 1), a2_dense, (1, -2))
            @test labels_dest == labels_dest_dense
            a_dest_dense = isdual(r2) ? -a_dest_dense : a_dest_dense
            @test convert(Array, a_dest) ≈ a_dest_dense

            # does not depend on input order
            a_dest = permutedims(contract(a2, (1, -2), a1, (-1, 1))[1], (2, 1))
            @test convert(Array, a_dest) ≈ a_dest_dense

            a_dest = contract((-1, -2), a2, (1, -2), a1, (-1, 1))
            @test convert(Array, a_dest) ≈ a_dest_dense

            # does not depend on permutations
            a3 = permutedims(a1, (2, 1))
            a_dest, _ = contract(a3, (1, -1), a2, (1, -2))
            @test convert(Array, a_dest) ≈ a_dest_dense

            a4 = permutedims(a2, (2, 1))
            a_dest, _ = contract(a1, (-1, 1), a4, (-2, 1))
            @test convert(Array, a_dest) ≈ a_dest_dense

            a_dest, _ = contract(a3, (1, -1), a4, (-2, 1))
            @test convert(Array, a_dest) ≈ a_dest_dense
        end
    end

    # @testset "matrix-matrix contraction B: total phase -1" begin
    #     # labels (1,-1,-2,2) × (2,1,-3,-4): P2=-1, T1=-1, T2=+1, U=-1 → total=-1
    #     a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    #     a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    #     a1_dense = convert(Array, a1)
    #     a2_dense = convert(Array, a2)
    #     a_dest, _ = contract(a1, (1, -1, -2, 2), a2, (2, 1, -3, -4))
    #     a_dest_dense, _ = contract(a1_dense, (1, -1, -2, 2), a2_dense, (2, 1, -3, -4))
    #     @test convert(Array, a_dest) ≈ -1 * a_dest_dense
    # end
    #
    # @testset "matrix-matrix contraction D: total phase -1" begin
    #     # labels (-1,1,2,-2) × (2,-3,1,-4): P1=+1, T1=-1, P2=+1, T2=-1, U=-1 → total=-1
    #     a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    #     a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    #     a1_dense = convert(Array, a1)
    #     a2_dense = convert(Array, a2)
    #     a_dest, _ = contract(a1, (-1, 1, 2, -2), a2, (2, -3, 1, -4))
    #     a_dest_dense, _ = contract(a1_dense, (-1, 1, 2, -2), a2_dense, (2, -3, 1, -4))
    #     @test convert(Array, a_dest) ≈ -1 * a_dest_dense
    # end
    #
    # @testset "matrix-matrix contraction G: total phase -1" begin
    #     # labels (1,2,-1,-2) × (2,-3,1,-4): P1=+1, T1=+1, P2=+1, T2=-1, U=+1 → total=-1
    #     a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    #     a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    #     a1_dense = convert(Array, a1)
    #     a2_dense = convert(Array, a2)
    #     a_dest, _ = contract(a1, (1, 2, -1, -2), a2, (2, -3, 1, -4))
    #     a_dest_dense, _ = contract(a1_dense, (1, 2, -1, -2), a2_dense, (2, -3, 1, -4))
    #     @test convert(Array, a_dest) ≈ -1 * a_dest_dense
    # end
end
