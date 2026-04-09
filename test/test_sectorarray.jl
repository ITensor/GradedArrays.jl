using GradedArrays: GradedArrays, SU2, SectorArray, SectorDelta, SectorMatrix, SectorRange,
    U1, dual, isdual, label, labels, sector, sector_multiplicities, sector_type, sectoraxes
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "SectorArray" begin
    @testset "Construction from labels, isdual, data" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = SectorArray((TKS.U1Irrep(1), TKS.U1Irrep(-1)), (false, true), data)
        @test sa isa SectorArray{Float64, 2, TKS.U1Irrep, Matrix{Float64}}
        @test sa isa AbstractArray{Float64, 2}
        @test !(sa isa GradedArrays.KroneckerArrays.AbstractKroneckerArray)
    end

    @testset "Construction from SectorRange tuples (backward compat)" begin
        sr1 = SectorRange(TKS.U1Irrep(1), false)
        sr2 = SectorRange(TKS.U1Irrep(-1), true)
        data = [1.0 2.0; 3.0 4.0]
        sa = SectorArray((sr1, sr2), data)
        @test label(sa, 1) == TKS.U1Irrep(1)
        @test label(sa, 2) == TKS.U1Irrep(-1)
        @test isdual(sa, 1) == false
        @test isdual(sa, 2) == true
    end

    @testset "Undef constructor" begin
        sa = SectorArray{Float64}(
            undef,
            (TKS.U1Irrep(0), TKS.U1Irrep(1)),
            (false, false),
            (3, 4)
        )
        @test size(sa) == (3, 4)
        @test eltype(sa) == Float64
        @test label(sa, 1) == TKS.U1Irrep(0)
        @test label(sa, 2) == TKS.U1Irrep(1)
    end

    @testset "Primitive accessors" begin
        data = ones(2, 3, 4)
        ls = (TKS.U1Irrep(1), TKS.U1Irrep(0), TKS.U1Irrep(-1))
        ds = (false, true, false)
        sa = SectorArray(ls, ds, data)

        @test labels(sa) === ls
        @test label(sa, 1) == TKS.U1Irrep(1)
        @test label(sa, 2) == TKS.U1Irrep(0)
        @test label(sa, 3) == TKS.U1Irrep(-1)
        @test isdual(sa, 1) == false
        @test isdual(sa, 2) == true
        @test isdual(sa, 3) == false
    end

    @testset "Derived accessors — sectoraxes" begin
        data = ones(2, 3)
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(-1)), (false, true), data
        )
        # sectoraxes(sa, d) returns SectorRange
        @test sectoraxes(sa, 1) == SectorRange(TKS.U1Irrep(1), false)
        @test sectoraxes(sa, 2) == SectorRange(TKS.U1Irrep(-1), true)

        secs = sectoraxes(sa)
        @test secs ==
            (SectorRange(TKS.U1Irrep(1), false), SectorRange(TKS.U1Irrep(-1), true))
    end

    @testset "sector(::SectorArray) returns SectorDelta" begin
        data = ones(2, 3)
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(-1)), (false, true), data
        )
        sd = sector(sa)
        @test sd isa SectorDelta{Float64, 2, TKS.U1Irrep}
        @test axes(sd) == sectoraxes(sa)
    end

    @testset "Derived accessors — sector_multiplicities (U1, dim=1)" begin
        # U1 has quantum dimension 1, so multiplicity = data size
        data = ones(3, 5)
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (false, false), data
        )
        @test sector_multiplicities(sa) == (3, 5)
    end

    @testset "Derived accessors — sector_multiplicities (SU2)" begin
        # SU2 j=1/2 has dim=2, so multiplicity = data_size / 2
        data = ones(4, 6)
        sa = SectorArray(
            (TKS.SU2Irrep(1 // 2), TKS.SU2Irrep(1 // 2)), (false, false), data
        )
        @test sector_multiplicities(sa) == (2, 3)
    end

    @testset "sector_type" begin
        data = ones(2, 2)
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (false, false), data
        )
        @test sector_type(typeof(sa)) == SectorRange{TKS.U1Irrep}
    end

    @testset "AbstractArray interface — size, getindex, setindex!" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (false, false), data
        )
        @test size(sa) == (2, 2)
        @test sa[1, 1] == 1.0
        @test sa[2, 1] == 3.0
        @test sa[1, 2] == 2.0
        @test sa[2, 2] == 4.0

        sa[1, 2] = 99.0
        @test sa[1, 2] == 99.0
    end

    @testset "copy" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (false, false), data
        )
        sa2 = copy(sa)
        @test sa2[1, 1] == sa[1, 1]
        @test labels(sa2) === labels(sa)
        @test sa2.isdual === sa.isdual

        # Verify it's a deep copy of data
        sa2[1, 1] = 999.0
        @test sa[1, 1] == 1.0
    end

    @testset "convert" begin
        data = [1 2; 3 4]
        sa = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(1)), (false, false), data
        )
        T = SectorArray{Float64, 2, TKS.U1Irrep, Matrix{Float64}}
        sa2 = convert(T, sa)
        @test eltype(sa2) == Float64
        @test sa2[1, 1] === 1.0
    end

    @testset "SectorMatrix alias" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (false, false), data
        )
        @test sa isa SectorMatrix
    end

    @testset "1D SectorArray" begin
        data = [1.0, 2.0, 3.0]
        sa = SectorArray((TKS.U1Irrep(1),), (false,), data)
        @test size(sa) == (3,)
        @test sa[2] == 2.0
        @test ndims(sa) == 1
    end

    @testset "3D SectorArray" begin
        data = ones(2, 3, 4)
        ls = (TKS.U1Irrep(1), TKS.U1Irrep(0), TKS.U1Irrep(-1))
        ds = (false, true, false)
        sa = SectorArray(ls, ds, data)
        @test size(sa) == (2, 3, 4)
        @test ndims(sa) == 3
        @test sa[1, 2, 3] == 1.0
    end

    @testset "permutedims" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        sa = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (false, true), data
        )
        sa_perm = permutedims(sa, (2, 1))
        @test size(sa_perm) == (3, 2)
        @test label(sa_perm, 1) == TKS.U1Irrep(0)
        @test label(sa_perm, 2) == TKS.U1Irrep(1)
        @test isdual(sa_perm, 1) == true
        @test isdual(sa_perm, 2) == false
        @test sa_perm[1, 1] == 1.0
        @test sa_perm[1, 2] == 4.0
    end

    @testset "mul!" begin
        using LinearAlgebra: mul!
        a_data = [1.0 2.0; 3.0 4.0]
        b_data = [5.0 6.0; 7.0 8.0]
        c_data = zeros(2, 2)
        # a has label(2) = U1(1), isdual=false => sectoraxes(a,2) = U1(1)
        # b has label(1) = U1(1), isdual=true  => sectoraxes(b,1) = dual(U1(1)) = U1(-1)
        # For check_mul_axes: sectoraxes(a,2) must == dual(sectoraxes(b,1))
        # sectoraxes(a,2) = U1(1), dual(sectoraxes(b,1)) = dual(U1(-1)) = U1(1)
        a = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(1)), (false, false), a_data
        )
        b = SectorArray(
            (TKS.U1Irrep(1), TKS.U1Irrep(0)), (true, false), b_data
        )
        c = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(0)), (false, false), c_data
        )
        mul!(c, a, b, 1.0, 0.0)
        @test c.data ≈ a_data * b_data
    end

    @testset "TensorAlgebra.add! (SectorArray to SectorArray)" begin
        using TensorAlgebra: TensorAlgebra
        data1 = [1.0 2.0; 3.0 4.0]
        data2 = [10.0 20.0; 30.0 40.0]
        sa1 = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(1)), (false, false), data1
        )
        sa2 = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(1)), (false, false), data2
        )
        TensorAlgebra.add!(sa1, sa2, 2.0, 1.0)
        @test sa1.data ≈ [21.0 42.0; 63.0 84.0]
    end

    @testset "TensorAlgebra.add! (SectorArray to plain Array)" begin
        using TensorAlgebra: TensorAlgebra
        dest = zeros(2, 2)
        data = [1.0 2.0; 3.0 4.0]
        sa = SectorArray(
            (TKS.U1Irrep(0), TKS.U1Irrep(1)), (false, false), data
        )
        TensorAlgebra.add!(dest, sa, 3.0, 0.0)
        @test dest ≈ [3.0 6.0; 9.0 12.0]
    end

    @testset "fill! abelian" begin
        sa = SectorArray((U1(0), dual(U1(0))), [1.0 2.0; 3.0 4.0])
        fill!(sa, 7.0)
        @test all(==(7.0), sa.data)

        fill!(sa, 0.0)
        @test all(iszero, sa.data)
    end

    @testset "fill! non-abelian errors for nonzero" begin
        sa = SectorArray(
            (SU2(TKS.SU2Irrep(1 // 2)), dual(SU2(TKS.SU2Irrep(1 // 2)))),
            ones(2, 2)
        )
        # fill! with zero is fine
        fill!(sa, 0.0)
        @test all(iszero, sa.data)

        # fill! with nonzero errors for non-abelian
        @test_throws ErrorException fill!(sa, 1.0)
    end

    @testset "zero!" begin
        sa = SectorArray((U1(0), dual(U1(0))), [1.0 2.0; 3.0 4.0])
        GradedArrays.FI.zero!(sa)
        @test all(iszero, sa.data)
    end
end
