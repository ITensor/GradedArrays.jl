import GradedArrays
using BlockArrays: Block, blocklengths, blocksize
using BlockSparseArrays: eachblockstoredindex
using GradedArrays: AbelianGradedArray, AbelianSectorArray, AbelianSectorDelta, SectorRange,
    U1, data, dual, flip, gradedrange, isdual, sectoraxes, sectors
using Random: randn!
using TensorAlgebra: contract, matricize, unmatricize
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

const fP0 = SectorRange(TKS.FermionParity(false))  # even parity
const fP1 = SectorRange(TKS.FermionParity(true))   # odd parity

function randn_blockdiagonal(elt::Type, axs::Tuple)
    a = AbelianGradedArray{elt}(undef, axs...)
    blockdiaglength = minimum(blocksize(a))
    N = ndims(a)
    for i in 1:blockdiaglength
        block_sectors = ntuple(d -> sectors(axs[d])[i], N)
        block_dims = ntuple(d -> blocklengths(axs[d])[i], N)
        block_data = randn!(Array{elt}(undef, block_dims...))
        a[Block(ntuple(Returns(i), N)...)] = AbelianSectorArray(block_sectors, block_data)
    end
    return a
end

# Dense materialization for AbelianGradedArray (scalar indexing is disallowed,
# so `convert(Array, ...)` doesn't work). Iterate stored blocks and copy each
# block's dense data into the matching slice of a zero-initialized output.
function to_dense(a::AbelianGradedArray{T, N}) where {T, N}
    out = zeros(T, size(a))
    bls = ntuple(d -> blocklengths(axes(a, d)), N)
    for bI in eachblockstoredindex(a)
        bk = ntuple(d -> Int(Tuple(bI)[d]), N)
        ranges = ntuple(N) do d
            start = sum(view(bls[d], 1:(bk[d] - 1)); init = 0) + 1
            return start:(start + bls[d][bk[d]] - 1)
        end
        out[ranges...] = data(a[bI])
    end
    return out
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
    @test mip((true, true, true), (2, 1, 3)) == -1
    @test mip((true, true, true), (1, 3, 2)) == -1
    @test mip((true, true, true), (2, 3, 1)) == 1
    @test mip((true, true, true), (3, 2, 1)) == -1

    # Two odd, one even: only odd-odd pairs count
    @test mip((true, false, true), (3, 2, 1)) == -1
    @test mip((true, false, true), (1, 2, 3)) == 1
    @test mip((true, false, true), (3, 1, 2)) == -1
end

@testset "fermion_permutation_phase" begin
    fpp = GradedArrays.fermion_permutation_phase

    # Bosonic (U1): always +1 regardless of permutation
    u0 = SectorRange(TKS.U1Irrep(0))
    u1 = SectorRange(TKS.U1Irrep(1))
    d_bos = AbelianSectorDelta{Float64}((u0, u1))
    @test fpp(d_bos, (2, 1)) == 1
    @test fpp(d_bos, (1, 2)) == 1

    # Even parity only: always +1
    d_even = AbelianSectorDelta{Float64}((fP0, fP0))
    @test fpp(d_even, (2, 1)) == 1
    @test fpp(d_even, (1, 2)) == 1

    # Two odd sectors: identity = +1, swap = -1
    d_odd = AbelianSectorDelta{Float64}((fP1, fP1))
    @test fpp(d_odd, (1, 2)) == 1
    @test fpp(d_odd, (2, 1)) == -1

    # Mixed even/odd: swap gives +1 (only odd-odd pairs contribute)
    d_mix = AbelianSectorDelta{Float64}((fP0, fP1))
    @test fpp(d_mix, (2, 1)) == 1

    d_mix2 = AbelianSectorDelta{Float64}((fP1, fP0))
    @test fpp(d_mix2, (2, 1)) == 1

    # Three odd sectors
    d3 = AbelianSectorDelta{Float64}((fP1, fP1, fP1))
    @test fpp(d3, (1, 2, 3)) == 1
    @test fpp(d3, (2, 3, 1)) == 1
    @test fpp(d3, (2, 1, 3)) == -1
    @test fpp(d3, (3, 2, 1)) == -1
end

@testset "permutedims on fermionic AbelianSectorArray" begin
    # Two odd sectors: swap picks up -1 phase
    sa = AbelianSectorArray((fP1, fP1), fill(3.0, 1, 1))
    sp = permutedims(sa, (2, 1))
    @test sp[1, 1] ≈ -3.0
    @test sectoraxes(sp) == (fP1, fP1)

    # Two even sectors: swap gives no phase
    sa_even = AbelianSectorArray((fP0, fP0), fill(3.0, 1, 1))
    sp_even = permutedims(sa_even, (2, 1))
    @test sp_even[1, 1] ≈ 3.0

    # Mixed (even, odd): swap gives no phase
    sa_mix = AbelianSectorArray((fP0, fP1), fill(3.0, 1, 1))
    sp_mix = permutedims(sa_mix, (2, 1))
    @test sp_mix[1, 1] ≈ 3.0
    @test sectoraxes(sp_mix) == (fP1, fP0)

    # Double permutation recovers original (phase squares to 1)
    @test permutedims(permutedims(sa, (2, 1)), (2, 1))[1, 1] ≈ sa[1, 1]

    # Three-index: cyclic permutation of 3 odd sectors → even crossings → +1
    sa3 = AbelianSectorArray((fP1, fP1, fP1), fill(2.0, 1, 1, 1))
    @test permutedims(sa3, (2, 3, 1))[1, 1, 1] ≈ 2.0
    @test permutedims(sa3, (3, 2, 1))[1, 1, 1] ≈ -2.0
    @test permutedims(sa3, (2, 1, 3))[1, 1, 1] ≈ -2.0

    # Two odd sectors with non-unit data: verify value propagates
    sa_val = AbelianSectorArray((fP1, fP1), fill(7.5, 1, 1))
    @test permutedims(sa_val, (2, 1))[1, 1] ≈ -7.5

    # No phase for bosonic (U1) sectors even though same permutation
    u1 = SectorRange(TKS.U1Irrep(1))
    sa_u1 = AbelianSectorArray((u1, u1), fill(3.0, 1, 1))
    sp_u1 = permutedims(sa_u1, (2, 1))
    @test sp_u1[1, 1] ≈ 3.0
end

@testset "matricize/unmatricize round-trip for fermionic AbelianSectorArray" begin
    # Even dual codomain leg: twist = +1, no phase modification
    sa_even_dual = AbelianSectorArray((dual(fP0), fP0), fill(5.0, 1, 1))
    @test isdual(axes(sa_even_dual, 1))
    m_even = matricize(sa_even_dual, (1,), (2,))
    @test m_even[1, 1] ≈ 5.0

    rt_even = unmatricize(m_even, (axes(sa_even_dual, 1),), (axes(sa_even_dual, 2),))
    @test rt_even[1, 1] ≈ 5.0

    # Non-dual codomain leg: twist not applied regardless of parity
    sa_odd_nondual = AbelianSectorArray((fP1, fP1), fill(5.0, 1, 1))
    @test !isdual(axes(sa_odd_nondual, 1))
    m_nondual = matricize(sa_odd_nondual, (1,), (2,))
    @test m_nondual[1, 1] ≈ 5.0

    # Odd dual codomain leg: twist = -1 → phase -1 applied during matricize
    sa_odd_dual = AbelianSectorArray((dual(fP1), fP1), fill(5.0, 1, 1))
    @test isdual(axes(sa_odd_dual, 1))
    m_odd = matricize(sa_odd_dual, (1,), (2,))
    @test m_odd[1, 1] ≈ -5.0

    # Round-trip: unmatricize applies same twist → -1 * (-5.0) = 5.0
    rt_odd = unmatricize(m_odd, (axes(sa_odd_dual, 1),), (axes(sa_odd_dual, 2),))
    @test rt_odd[1, 1] ≈ 5.0

    # Two odd dual codomain legs: phase = (-1) * (-1) = +1
    sa_two_odd_dual = AbelianSectorArray((dual(fP1), dual(fP1), fP1), fill(3.0, 1, 1, 1))
    m_two = matricize(sa_two_odd_dual, (1, 2), (3,))
    @test m_two[1, 1] ≈ 3.0

    rt_two = unmatricize(
        m_two,
        (axes(sa_two_odd_dual, 1), axes(sa_two_odd_dual, 2)),
        (axes(sa_two_odd_dual, 3),)
    )
    @test rt_two[1, 1, 1] ≈ 3.0
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})

# For all-odd blocks, the total phase from permutation + matricize twist is -1,
# so the GradedArray result equals -1 * the dense result.
@testset "contract GradedArray with FermionParity (eltype=$elt)" for elt in
    elts

    r_odd = gradedrange([fP1 => 2])

    @testset "permutedims of all-odd GradedArray picks up -1 phase" begin
        a = randn_blockdiagonal(elt, (r_odd, dual(r_odd)))
        a_perm = permutedims(a, (2, 1))
        a_dense_perm = permutedims(to_dense(a), (2, 1))
        @test to_dense(a_perm) ≈ -1 * a_dense_perm
    end

    @testset "matrix-matrix contraction" begin
        for r1 in (r_odd, dual(r_odd)),
                r2 in (r_odd, dual(r_odd)),
                r3 in (r_odd, dual(r_odd))

            a1 = randn_blockdiagonal(elt, (r1, dual(r2)))
            a2 = randn_blockdiagonal(elt, (r2, r3))
            a1_dense = to_dense(a1)
            a2_dense = to_dense(a2)
            a_dest, labels_dest = contract(a1, (-1, 1), a2, (1, -2))
            a_dest_dense, labels_dest_dense = contract(a1_dense, (-1, 1), a2_dense, (1, -2))
            @test labels_dest == labels_dest_dense
            a_dest_dense = isdual(r2) ? -a_dest_dense : a_dest_dense
            @test to_dense(a_dest) ≈ a_dest_dense

            # does not depend on input order
            a_dest = permutedims(contract(a2, (1, -2), a1, (-1, 1))[1], (2, 1))
            @test to_dense(a_dest) ≈ a_dest_dense

            a_dest = contract((-1, -2), a2, (1, -2), a1, (-1, 1))
            @test to_dense(a_dest) ≈ a_dest_dense

            # does not depend on permutations
            a3 = permutedims(a1, (2, 1))
            a_dest, _ = contract(a3, (1, -1), a2, (1, -2))
            @test to_dense(a_dest) ≈ a_dest_dense

            a4 = permutedims(a2, (2, 1))
            a_dest, _ = contract(a1, (-1, 1), a4, (-2, 1))
            @test to_dense(a_dest) ≈ a_dest_dense

            a_dest, _ = contract(a3, (1, -1), a4, (-2, 1))
            @test to_dense(a_dest) ≈ a_dest_dense
        end
    end

    @testset "matrix-matrix contraction B: total phase -1" begin
        # labels (1,-1,-2,2) × (2,1,-3,-4): P2=-1, T1=-1, T2=+1, U=-1 → total=-1
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a1_dense = to_dense(a1)
        a2_dense = to_dense(a2)
        a_dest, _ = contract(a1, (1, -1, -2, 2), a2, (2, 1, -3, -4))
        a_dest_dense, _ = contract(a1_dense, (1, -1, -2, 2), a2_dense, (2, 1, -3, -4))
        @test to_dense(a_dest) ≈ -1 * a_dest_dense
    end

    @testset "matrix-matrix contraction D: total phase -1" begin
        # labels (-1,1,2,-2) × (2,-3,1,-4): P1=+1, T1=-1, P2=+1, T2=-1, U=-1 → total=-1
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a1_dense = to_dense(a1)
        a2_dense = to_dense(a2)
        a_dest, _ = contract(a1, (-1, 1, 2, -2), a2, (2, -3, 1, -4))
        a_dest_dense, _ = contract(a1_dense, (-1, 1, 2, -2), a2_dense, (2, -3, 1, -4))
        @test to_dense(a_dest) ≈ -1 * a_dest_dense
    end

    @testset "matrix-matrix contraction G: total phase -1" begin
        # labels (1,2,-1,-2) × (2,-3,1,-4): P1=+1, T1=+1, P2=+1, T2=-1, U=+1 → total=-1
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a1_dense = to_dense(a1)
        a2_dense = to_dense(a2)
        a_dest, _ = contract(a1, (1, 2, -1, -2), a2, (2, -3, 1, -4))
        a_dest_dense, _ = contract(a1_dense, (1, 2, -1, -2), a2_dense, (2, -3, 1, -4))
        @test to_dense(a_dest) ≈ -1 * a_dest_dense
    end
end
