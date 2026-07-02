import GradedArrays
using BlockArrays: Block, blocklengths, blocksize
using GradedArrays: AbelianGradedArray, AbelianSectorArray, AbelianSectorDelta,
    SectorProduct, SectorRange, U1, dual, eachblockstoredindex, eachsectoraxis, flip,
    gradedrange, isdual, sectoraxes, sectors
using Random: randn!
using TensorAlgebra:
    contract, matricize, matricizeopperm, permutedimsop, unmatricize, unmatricizeperm!
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

const fP0 = SectorRange(TKS.FermionParity(false))  # even parity
const fP1 = SectorRange(TKS.FermionParity(true))   # odd parity

@testset "fermionparity / twist" begin
    # `FermionNumber = U1Irrep ⊠ FermionParity` is a product sector with a bosonic
    # component, on which `TKS.fermionparity` errors; decomposing over components with a
    # bosonic-irrep fallback handles it.
    for n in -2:2
        c = SectorRange(TKS.FermionNumber(n))
        @test GradedArrays.twist(c) == (isodd(n) ? -1 : 1)
        @test GradedArrays.fermionparity(c) == isodd(n)
    end

    # The same holds for GradedArrays' own `SectorProduct`.
    for n in -2:2
        c = SectorRange(SectorProduct(TKS.U1Irrep(n), TKS.FermionParity(isodd(n))))
        @test GradedArrays.fermionparity(c) == isodd(n)
    end

    # Plain bosonic group irreps have even fermion parity.
    @test GradedArrays.fermionparity(SectorRange(TKS.U1Irrep(2))) == false
    @test GradedArrays.fermionparity(U1(0)) == false

    # `FermionParity` delegates to TensorKitSectors unchanged.
    @test GradedArrays.fermionparity(fP0) == false
    @test GradedArrays.fermionparity(fP1) == true

    # A sector with no fermion parity (an anyon) has no method.
    @test_throws MethodError GradedArrays.fermionparity(SectorRange(TKS.FibonacciAnyon(:τ)))
end

function randn_blockdiagonal(elt::Type, axs::Tuple)
    a = AbelianGradedArray{elt}(undef, axs...)
    blockdiaglength = minimum(blocksize(a))
    N = ndims(a)
    for i in 1:blockdiaglength
        block_sectors = ntuple(d -> eachsectoraxis(axs[d])[i], N)
        block_dims = ntuple(d -> blocklengths(axs[d])[i], N)
        block_data = randn!(Array{elt}(undef, block_dims...))
        a[Block(ntuple(Returns(i), N)...)] = AbelianSectorArray(block_sectors, block_data)
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

@testset "conj on fermionic AbelianSectorArray" begin
    # Two odd sectors: reversing 2 odd legs is one odd-odd inversion → -1 phase
    sa = AbelianSectorArray((fP1, fP1), fill(3.0, 1, 1))
    sc = conj(sa)
    @test sc[1, 1] ≈ -3.0
    @test sectoraxes(sc) == (dual(fP1), dual(fP1))

    # Two even sectors: no phase, data just conjugated
    @test conj(AbelianSectorArray((fP0, fP0), fill(3.0, 1, 1)))[1, 1] ≈ 3.0

    # Mixed (even, odd): single odd leg → no odd-odd inversion → no phase
    sa_mix = AbelianSectorArray((fP0, fP1), fill(3.0, 1, 1))
    @test conj(sa_mix)[1, 1] ≈ 3.0
    @test sectoraxes(conj(sa_mix)) == (dual(fP0), dual(fP1))

    # Three odd sectors: reverse(1,2,3) has 3 odd-odd inversions → odd → -1 phase
    @test conj(AbelianSectorArray((fP1, fP1, fP1), fill(2.0, 1, 1, 1)))[1, 1, 1] ≈ -2.0

    # Complex data: conjugates the data *and* applies the fermionic phase
    sa_c = AbelianSectorArray((fP1, fP1), fill(1.0 + 2.0im, 1, 1))
    @test conj(sa_c)[1, 1] ≈ -(1.0 - 2.0im)

    # Involution: conj ∘ conj recovers data and sectors (phase squares to 1)
    @test conj(conj(sa))[1, 1] ≈ sa[1, 1]
    @test sectoraxes(conj(conj(sa))) == (fP1, fP1)
    @test conj(conj(sa_c))[1, 1] ≈ sa_c[1, 1]

    # Mutation safety: conj must not scale the parent block in place
    sa_mut = AbelianSectorArray((fP1, fP1), fill(5.0, 1, 1))
    conj(sa_mut)
    @test sa_mut[1, 1] ≈ 5.0

    # Bosonic (U1) sectors: no fermionic phase, just data conj
    u1 = SectorRange(TKS.U1Irrep(1))
    @test conj(AbelianSectorArray((u1, u1), fill(1.0 + 2.0im, 1, 1)))[1, 1] ≈ 1.0 - 2.0im
end

# `conj` is routed through `conj.`, so a direct `conj.(a) == conj(a)` check is vacuous. The
# single-operand fermionic sign is pinned by the hand-computed values in the testset above;
# these tests cover what broadcasting adds: composing conj in larger expressions, the graded
# block loop, axis dualization, and the involution.
@testset "conj broadcast composes linearly (sector)" begin
    sa = AbelianSectorArray((fP1, fP1), fill(1.0 + 2.0im, 1, 1))
    sb = AbelianSectorArray((fP1, fP1), fill(3.0 - 1.0im, 1, 1))
    cs = conj.(sa) .- conj.(sb) ./ 2
    @test cs[1, 1] ≈ conj.(sa)[1, 1] - conj.(sb)[1, 1] / 2
    @test sectoraxes(cs) == sectoraxes(conj.(sa))
end

@testset "conj broadcast on fermionic graded arrays (eltype=$elt)" for elt in
    (Float64, ComplexF64)

    # Mixed even/odd sectors, so the two diagonal blocks pick up different reversal signs.
    g = gradedrange([fP0 => 2, fP1 => 3])
    a = randn_blockdiagonal(elt, (g, dual(g)))
    b = randn_blockdiagonal(elt, (g, dual(g)))

    ca = conj.(a)
    @test ca isa AbelianGradedArray
    @test isdual(axes(ca, 1)) == !isdual(axes(a, 1))
    @test isdual(axes(ca, 2)) == !isdual(axes(a, 2))

    # The graded block loop agrees block-by-block with the sector-level conj broadcast.
    for I in eachblockstoredindex(a)
        @test ca[I] ≈ conj.(a[I])
    end

    # Involution: the reversal sign squares to 1, recovering the array and its axes.
    @test Array(conj.(ca)) ≈ Array(a)
    @test axes(conj.(ca)) == axes(a)

    # Compound conj broadcasts combine linearly, block by block.
    for I in eachblockstoredindex(a)
        @test (conj.(a) .+ conj.(b))[I] ≈ conj.(a[I]) .+ conj.(b[I])
        @test (conj.(a) .- conj.(b) ./ 2)[I] ≈ conj.(a[I]) .- conj.(b[I]) ./ 2
    end

    # Conjugating only one operand leaves dualized axes against non-dual ones: rejected.
    @test_throws DimensionMismatch conj.(a) .- b
end

# A 4-leg all-odd block exercises the multi-leg reversal sign through a broadcast.
@testset "conj broadcast on 4D all-odd graded array (eltype=$elt)" for elt in
    (Float64, ComplexF64)
    r_odd = gradedrange([fP1 => 2])
    a = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
    ca = conj.(a)
    @test all(d -> isdual(axes(ca, d)) == !isdual(axes(a, d)), 1:4)
    for I in eachblockstoredindex(a)
        @test ca[I] ≈ conj.(a[I])
    end
    @test Array(conj.(ca)) ≈ Array(a)
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
        a_dense_perm = permutedims(Array(a), (2, 1))
        @test Array(a_perm) ≈ -1 * a_dense_perm
    end

    @testset "conj of all-odd GradedArray picks up -1 phase" begin
        a = randn_blockdiagonal(elt, (r_odd, dual(r_odd)))
        a_dense_before = Array(a)
        ac = conj(a)
        # Every stored block is all-odd → reversal phase -1, on top of data conj.
        @test Array(ac) ≈ -1 * conj(a_dense_before)
        # Axis dualities are flipped.
        @test isdual(axes(ac, 1)) == !isdual(axes(a, 1))
        @test isdual(axes(ac, 2)) == !isdual(axes(a, 2))
        # Involution: conj ∘ conj recovers the original.
        @test Array(conj(ac)) ≈ a_dense_before
        # Mutation safety: conj must not scale the parent's blocks in place.
        @test Array(a) ≈ a_dense_before
    end

    @testset "matrix-matrix contraction" begin
        for r1 in (r_odd, dual(r_odd)),
                r2 in (r_odd, dual(r_odd)),
                r3 in (r_odd, dual(r_odd))

            a1 = randn_blockdiagonal(elt, (r1, dual(r2)))
            a2 = randn_blockdiagonal(elt, (r2, r3))
            a1_dense = Array(a1)
            a2_dense = Array(a2)
            a_dest, labels_dest = contract(a1, (-1, 1), a2, (1, -2))
            a_dest_dense, labels_dest_dense = contract(a1_dense, (-1, 1), a2_dense, (1, -2))
            @test labels_dest == labels_dest_dense
            a_dest_dense = isdual(r2) ? -a_dest_dense : a_dest_dense
            @test Array(a_dest) ≈ a_dest_dense

            # does not depend on input order
            a_dest = permutedims(contract(a2, (1, -2), a1, (-1, 1))[1], (2, 1))
            @test Array(a_dest) ≈ a_dest_dense

            a_dest = contract((-1, -2), a2, (1, -2), a1, (-1, 1))
            @test Array(a_dest) ≈ a_dest_dense

            # does not depend on permutations
            a3 = permutedims(a1, (2, 1))
            a_dest, _ = contract(a3, (1, -1), a2, (1, -2))
            @test Array(a_dest) ≈ a_dest_dense

            a4 = permutedims(a2, (2, 1))
            a_dest, _ = contract(a1, (-1, 1), a4, (-2, 1))
            @test Array(a_dest) ≈ a_dest_dense

            a_dest, _ = contract(a3, (1, -1), a4, (-2, 1))
            @test Array(a_dest) ≈ a_dest_dense
        end
    end

    @testset "matrix-matrix contraction B: total phase -1" begin
        # Canonical contraction (a1 codomain ↔ a2 domain). The two odd contracted
        # indices appear in swapped order on a2, one transposition, so the
        # contraction twists by -1 relative to the dense result.
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a1_dense = Array(a1)
        a2_dense = Array(a2)
        a_dest, _ = contract(a1, (1, 2, -1, -2), a2, (-3, -4, 2, 1))
        a_dest_dense, _ = contract(a1_dense, (1, 2, -1, -2), a2_dense, (-3, -4, 2, 1))
        @test Array(a_dest) ≈ -1 * a_dest_dense
    end

    @testset "matrix-matrix contraction D: total phase -1" begin
        # labels (-1,1,2,-2) × (2,-3,1,-4): P1=+1, T1=-1, P2=+1, T2=-1, U=-1 → total=-1
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a1_dense = Array(a1)
        a2_dense = Array(a2)
        a_dest, _ = contract(a1, (-1, 1, 2, -2), a2, (2, -3, 1, -4))
        a_dest_dense, _ = contract(a1_dense, (-1, 1, 2, -2), a2_dense, (2, -3, 1, -4))
        @test Array(a_dest) ≈ -1 * a_dest_dense
    end

    @testset "matrix-matrix contraction G: total phase -1" begin
        # Canonical contraction with the contracted indices on a1's domain (dual)
        # side and a2's codomain side. Same single-transposition twist as case B.
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a1_dense = Array(a1)
        a2_dense = Array(a2)
        a_dest, _ = contract(a1, (-1, -2, 1, 2), a2, (2, 1, -3, -4))
        a_dest_dense, _ = contract(a1_dense, (-1, -2, 1, 2), a2_dense, (2, 1, -3, -4))
        @test Array(a_dest) ≈ -1 * a_dest_dense
    end

    @testset "full contraction to a scalar (rank-0)" begin
        # A full contraction collapses to a rank-0 graded array. The rank-0 unmatricize is
        # sign-free (a 1×1 block, no permute), so all fermion signs come from the matricize
        # twist and must still reach the scalar. Legs (r,r,dr,dr) contracted with
        # (dr,dr,r,r): parallel pairing has no twist, swapping the first two contracted (odd)
        # legs is one transposition and twists by -1.
        a1 = randn_blockdiagonal(elt, (r_odd, r_odd, dual(r_odd), dual(r_odd)))
        a2 = randn_blockdiagonal(elt, (dual(r_odd), dual(r_odd), r_odd, r_odd))
        a1_dense = Array(a1)
        a2_dense = Array(a2)

        parallel = contract((), a1, (-1, -2, -3, -4), a2, (-1, -2, -3, -4))
        @test parallel isa AbelianGradedArray{elt, <:Any, 0}
        @test Array(parallel) ≈
            contract((), a1_dense, (-1, -2, -3, -4), a2_dense, (-1, -2, -3, -4))

        crossed = contract((), a1, (-1, -2, -3, -4), a2, (-2, -1, -3, -4))
        @test crossed isa AbelianGradedArray{elt, <:Any, 0}
        @test Array(crossed) ≈
            -1 * contract((), a1_dense, (-1, -2, -3, -4), a2_dense, (-2, -1, -3, -4))
    end
end

# The fused permuting `matricizeopperm` (folding the permute into the gather) must agree
# block-by-block with the reference two-pass `permutedimsop` then `matricize`, including
# the per-block fermion permutation sign. `U1` exercises the non-self-dual `op = conj`
# axis dualization that `FermionParity` (self-dual) hides.
@testset "fused matricizeopperm matches permute-then-matricize (eltype=$elt)" for elt in
    (
        Float64,
        ComplexF64,
    )
    # The two-pass permute-then-matricize that the fused `matricizeopperm` replaces.
    matricizeopperm_ref(op, a, perm_codomain, perm_domain) =
        matricize(
        permutedimsop(op, a, perm_codomain, perm_domain),
        Val(length(perm_codomain))
    )

    gb = gradedrange([U1(0) => 2, U1(1) => 3])
    gf = gradedrange([fP0 => 2, fP1 => 3])
    ro = gradedrange([fP1 => 2])
    for op in (identity, conj)
        cases = [
            (randn(elt, (gb, dual(gb))), (1,), (2,)),
            (randn(elt, (gb, dual(gb))), (2,), (1,)),
            (randn(elt, (gf, dual(gf))), (1,), (2,)),
            (randn(elt, (gf, dual(gf))), (2,), (1,)),
            (randn(elt, (gf, dual(gf), gf)), (2, 1), (3,)),
            (randn(elt, (gf, dual(gf), gf)), (3,), (1, 2)),
            (randn(elt, (ro, ro, dual(ro), dual(ro))), (3, 1), (4, 2)),
            (randn(elt, (gb, dual(gb), gb, dual(gb))), (3, 1), (2, 4)),
        ]
        for (a, perm_codomain, perm_domain) in cases
            @test matricizeopperm(op, a, perm_codomain, perm_domain) ≈
                matricizeopperm_ref(op, a, perm_codomain, perm_domain)
        end
    end
end

# The fused `unmatricizeperm!` folds the permutation into the scatter. Check it against the
# two-pass it replaces: the non-permuting `unmatricize` into codomain/domain order, then an
# array-level permute back to destination order, carrying the per-block fermion sign.
@testset "fused unmatricizeperm! matches unmatricize-then-permute (eltype=$elt)" for elt in
    (
        Float64,
        ComplexF64,
    )
    function unmatricizeperm_ref(a_dest, m, invperm_codomain, invperm_domain)
        K = length(invperm_codomain)
        codomain_axes = ntuple(i -> axes(a_dest)[invperm_codomain[i]], K)
        domain_axes = ntuple(i -> axes(a_dest)[invperm_domain[i]], ndims(a_dest) - K)
        return permutedims(
            # `axes(a_dest)` are stored (dualized) domain axes, but `unmatricize` takes them
            # codomain-facing, so un-dualize with `conj` before the call.
            unmatricize(m, codomain_axes, conj.(domain_axes)),
            invperm((invperm_codomain..., invperm_domain...))
        )
    end

    gb = gradedrange([U1(0) => 2, U1(1) => 3])
    gf = gradedrange([fP0 => 2, fP1 => 3])
    ro = gradedrange([fP1 => 2])
    cases = [
        (randn(elt, (gb, dual(gb))), (1,), (2,)),
        (randn(elt, (gb, dual(gb))), (2,), (1,)),
        (randn(elt, (gf, dual(gf))), (1,), (2,)),
        (randn(elt, (gf, dual(gf))), (2,), (1,)),
        (randn(elt, (gf, dual(gf), gf)), (2, 1), (3,)),
        (randn(elt, (gf, dual(gf), gf)), (3,), (1, 2)),
        (randn(elt, (ro, ro, dual(ro), dual(ro))), (3, 1), (4, 2)),
        (randn(elt, (gb, dual(gb), gb, dual(gb))), (3, 1), (2, 4)),
    ]
    for (a, invperm_codomain, invperm_domain) in cases
        m = matricizeopperm(identity, a, invperm_codomain, invperm_domain)
        @test unmatricizeperm!(similar(a), m, invperm_codomain, invperm_domain) ≈
            unmatricizeperm_ref(similar(a), m, invperm_codomain, invperm_domain)
    end
end
