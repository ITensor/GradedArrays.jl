import GradedArrays
using GradedArrays: AbelianSectorArray, AbelianSectorDelta, SectorRange, dual, sectors
using Random: randn!
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @testset

# Fermionic sector aliases: FermionParity has two values, even (false) and odd (true),
# both with quantum dimension 1.
const fP0 = SectorRange(TKS.FermionParity(false))  # even parity
const fP1 = SectorRange(TKS.FermionParity(true))   # odd parity

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
end

@testset "fermion_permutation_phase" begin
    fpp = GradedArrays.fermion_permutation_phase

    # Bosonic: always +1
    delta_bos = AbelianSectorDelta{Float64}(
        (TKS.U1Irrep(0), TKS.U1Irrep(1)),
        (false, false)
    )
    @test fpp(delta_bos, (2, 1)) == true  # Bosonic returns `true` (== 1)

    # Fermionic: even parity swap → +1
    delta_even = AbelianSectorDelta{Float64}(
        (TKS.FermionParity(false), TKS.FermionParity(false)),
        (false, false)
    )
    @test fpp(delta_even, (2, 1)) == 1

    # Fermionic: two odd parity swap → -1
    delta_odd = AbelianSectorDelta{Float64}(
        (TKS.FermionParity(true), TKS.FermionParity(true)),
        (false, false)
    )
    @test fpp(delta_odd, (2, 1)) == -1

    # Fermionic: one odd one even swap → +1
    delta_mixed = AbelianSectorDelta{Float64}(
        (TKS.FermionParity(true), TKS.FermionParity(false)),
        (false, false)
    )
    @test fpp(delta_mixed, (2, 1)) == 1
end

@testset "AbelianSectorArray permutedims applies fermionic phase" begin
    # Two odd-parity legs: permuting should negate
    data = randn!(Matrix{Float64}(undef, 1, 1))
    s = AbelianSectorArray(
        (TKS.FermionParity(true), TKS.FermionParity(true)),
        (false, false),
        copy(data)
    )
    sp = permutedims(s, (2, 1))
    @test sp[1, 1] ≈ -data[1, 1]

    # Two even-parity legs: no phase
    data_even = randn!(Matrix{Float64}(undef, 1, 1))
    s_even = AbelianSectorArray(
        (TKS.FermionParity(false), TKS.FermionParity(false)),
        (false, false),
        copy(data_even)
    )
    sp_even = permutedims(s_even, (2, 1))
    @test sp_even[1, 1] ≈ data_even[1, 1]
end
