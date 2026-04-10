using GradedArrays: GradedArrays, AbelianSectorArray, AbelianSectorDelta,
    AbelianSectorMatrix, SU2, SectorRange, U1, data, dual, isdual, sector,
    sector_multiplicities, sector_type, sectoraxes
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "AbelianSectorArray" begin
    @testset "Construction from SectorRange tuples" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = AbelianSectorArray((U1(1), U1(-1)'), data)
        @test sa isa AbelianSectorArray{Float64, 2, Matrix{Float64}, U1}
        @test sa isa AbstractArray{Float64, 2}
        @test !(sa isa GradedArrays.KroneckerArrays.AbstractKroneckerArray)
    end

    @testset "Construction with dual sectors" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = AbelianSectorArray((U1(1), U1(-1)'), data)
        @test sectoraxes(sa) == (U1(1), U1(-1)')
    end

    @testset "Undef constructor" begin
        sa = AbelianSectorArray{Float64}(
            undef,
            (U1(0), U1(1)),
            (3, 4)
        )
        @test size(sa) == (3, 4)
        @test eltype(sa) == Float64
        @test sectoraxes(sa) == (U1(0), U1(1))
    end

    @testset "Primitive accessors" begin
        data = ones(2, 3, 4)
        sa = AbelianSectorArray((U1(1), U1(0)', U1(-1)), data)

        @test sectoraxes(sa) == (U1(1), U1(0)', U1(-1))
        @test sectoraxes(sa, 1) == U1(1)
        @test sectoraxes(sa, 2) == U1(0)'
        @test sectoraxes(sa, 3) == U1(-1)
        @test isdual(sa, 1) == false
        @test isdual(sa, 2) == true
        @test isdual(sa, 3) == false
    end

    @testset "Derived accessors — sectoraxes" begin
        data = ones(2, 3)
        sa = AbelianSectorArray((U1(1), U1(-1)'), data)
        @test sectoraxes(sa, 1) == U1(1)
        @test sectoraxes(sa, 2) == U1(-1)'
        @test sectoraxes(sa) == (U1(1), U1(-1)')
    end

    @testset "sector(::AbelianSectorArray) returns AbelianSectorDelta" begin
        data = ones(2, 3)
        sa = AbelianSectorArray((U1(1), U1(-1)'), data)
        sd = sector(sa)
        @test sd isa AbelianSectorDelta{Float64, 2, U1}
        @test axes(sd) == sectoraxes(sa)
    end

    @testset "Derived accessors — sector_multiplicities (U1, dim=1)" begin
        data = ones(3, 5)
        sa = AbelianSectorArray((U1(1), U1(0)), data)
        @test sector_multiplicities(sa) == (3, 5)
    end

    @testset "Derived accessors — sector_multiplicities (SU2)" begin
        data = ones(4, 6)
        sa = AbelianSectorArray(
            (SU2(1 // 2), SU2(1 // 2)), data
        )
        @test sector_multiplicities(sa) == (2, 3)
    end

    @testset "sector_type" begin
        data = ones(2, 2)
        sa = AbelianSectorArray((U1(1), U1(0)), data)
        @test sector_type(typeof(sa)) == U1
    end

    @testset "AbstractArray interface — size, getindex, setindex!" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = AbelianSectorArray((U1(1), U1(0)), data)
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
        sa = AbelianSectorArray((U1(1), U1(0)), data)
        sa2 = copy(sa)
        @test sa2[1, 1] == sa[1, 1]
        @test sectoraxes(sa2) == sectoraxes(sa)

        sa2[1, 1] = 999.0
        @test sa[1, 1] == 1.0
    end

    @testset "convert" begin
        data = [1 2; 3 4]
        sa = AbelianSectorArray((U1(0), U1(1)), data)
        T = AbelianSectorArray{Float64, 2, Matrix{Float64}, U1}
        sa2 = convert(T, sa)
        @test eltype(sa2) == Float64
        @test sa2[1, 1] === 1.0
    end

    @testset "AbelianSectorMatrix alias" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = AbelianSectorArray((U1(1), U1(0)), data)
        @test sa isa AbelianSectorMatrix
    end

    @testset "1D AbelianSectorArray" begin
        data = [1.0, 2.0, 3.0]
        sa = AbelianSectorArray((U1(1),), data)
        @test size(sa) == (3,)
        @test sa[2] == 2.0
        @test ndims(sa) == 1
    end

    @testset "3D AbelianSectorArray" begin
        data = ones(2, 3, 4)
        sa = AbelianSectorArray((U1(1), U1(0)', U1(-1)), data)
        @test size(sa) == (2, 3, 4)
        @test ndims(sa) == 3
        @test sa[1, 2, 3] == 1.0
    end

    @testset "permutedims" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        sa = AbelianSectorArray((U1(1), U1(0)'), data)
        sa_perm = permutedims(sa, (2, 1))
        @test size(sa_perm) == (3, 2)
        @test sectoraxes(sa_perm) == (U1(0)', U1(1))
        @test sa_perm[1, 1] == 1.0
        @test sa_perm[1, 2] == 4.0
    end

    @testset "mul!" begin
        using LinearAlgebra: mul!
        a_data = [1.0 2.0; 3.0 4.0]
        b_data = [5.0 6.0; 7.0 8.0]
        c_data = zeros(2, 2)
        a = AbelianSectorArray((U1(0), U1(1)), a_data)
        b = AbelianSectorArray((U1(1)', U1(0)), b_data)
        c = AbelianSectorArray((U1(0), U1(0)), c_data)
        mul!(c, a, b, 1.0, 0.0)
        @test data(c) ≈ a_data * b_data
    end

    @testset "TensorAlgebra.add! (AbelianSectorArray to AbelianSectorArray)" begin
        using TensorAlgebra: TensorAlgebra
        data1 = [1.0 2.0; 3.0 4.0]
        data2 = [10.0 20.0; 30.0 40.0]
        sa1 = AbelianSectorArray((U1(0), U1(1)), data1)
        sa2 = AbelianSectorArray((U1(0), U1(1)), data2)
        TensorAlgebra.add!(sa1, sa2, 2.0, 1.0)
        @test data(sa1) ≈ [21.0 42.0; 63.0 84.0]
    end

    @testset "TensorAlgebra.add! (AbelianSectorArray to plain Array)" begin
        using TensorAlgebra: TensorAlgebra
        dest = zeros(2, 2)
        data = [1.0 2.0; 3.0 4.0]
        sa = AbelianSectorArray((U1(0), U1(1)), data)
        TensorAlgebra.add!(dest, sa, 3.0, 0.0)
        @test dest ≈ [3.0 6.0; 9.0 12.0]
    end

    @testset "fill! abelian" begin
        sa = AbelianSectorArray((U1(0), dual(U1(0))), [1.0 2.0; 3.0 4.0])
        fill!(sa, 7.0)
        @test all(==(7.0), data(sa))

        fill!(sa, 0.0)
        @test all(iszero, data(sa))
    end

    @testset "fill! non-abelian errors for nonzero" begin
        sa = AbelianSectorArray(
            (SU2(1 // 2), dual(SU2(1 // 2))),
            ones(2, 2)
        )
        fill!(sa, 0.0)
        @test all(iszero, data(sa))

        @test_throws ErrorException fill!(sa, 1.0)
    end

    @testset "zero!" begin
        sa = AbelianSectorArray((U1(0), dual(U1(0))), [1.0 2.0; 3.0 4.0])
        GradedArrays.FI.zero!(sa)
        @test all(iszero, data(sa))
    end
end
