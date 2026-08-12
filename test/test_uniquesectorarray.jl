using GradedArrays: GradedArrays, SU2, SectorOneTo, SectorRange, U1, UniqueSectorArray,
    UniqueSectorDelta, UniqueSectorMatrix, UniqueSectorVector, data, dual, isdual, sector,
    sector_kron, sectoraxes, sectortype, with_scalar_indexing
using LinearAlgebra: tr
using TensorKitSectors: TensorKitSectors as TKS
using Test: @test, @test_throws, @testset

@testset "UniqueSectorArray" begin
    @testset "Construction from SectorRange tuples" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(data, (U1(1), conj(U1(-1))))
        @test sa isa UniqueSectorArray{Float64, U1, 2, <:Any, <:Any, Matrix{Float64}}
        @test sa isa AbstractArray{Float64, 2}
    end

    @testset "Construction with dual sectors" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(data, (U1(1), conj(U1(-1))))
        @test sectoraxes(sa) == (U1(1), conj(U1(-1)))
    end

    @testset "Undef constructor (SectorOneTo)" begin
        sa = UniqueSectorArray{Float64}(
            undef,
            (SectorOneTo(U1(0), 3), SectorOneTo(U1(1), 4))
        )
        @test size(sa) == (3, 4)
        @test eltype(sa) == Float64
        @test sectoraxes(sa) == (U1(0), U1(1))
    end

    @testset "Primitive accessors" begin
        data = ones(2, 3, 4)
        sa = UniqueSectorArray(data, (U1(1), conj(U1(0)), U1(-1)))

        @test sectoraxes(sa) == (U1(1), conj(U1(0)), U1(-1))
        @test sectoraxes(sa, 1) == U1(1)
        @test sectoraxes(sa, 2) == conj(U1(0))
        @test sectoraxes(sa, 3) == U1(-1)
        @test isdual(axes(sa, 1)) == false
        @test isdual(axes(sa, 2)) == true
        @test isdual(axes(sa, 3)) == false
    end

    @testset "Derived accessors — sectoraxes" begin
        data = ones(2, 3)
        sa = UniqueSectorArray(data, (U1(1), conj(U1(-1))))
        @test sectoraxes(sa, 1) == U1(1)
        @test sectoraxes(sa, 2) == conj(U1(-1))
        @test sectoraxes(sa) == (U1(1), conj(U1(-1)))
    end

    @testset "sector(::UniqueSectorArray) returns UniqueSectorDelta" begin
        data = ones(2, 3)
        sa = UniqueSectorArray(data, (U1(1), conj(U1(-1))))
        sd = sector(sa)
        @test sd isa UniqueSectorDelta{Float64, U1, 2}
        @test axes(sd) == sectoraxes(sa)
    end

    @testset "sectortype" begin
        data = ones(2, 2)
        sa = UniqueSectorArray(data, (U1(1), U1(0)))
        @test sectortype(typeof(sa)) == U1
    end

    @testset "rank-0 (scalar) array" begin
        # A rank-0 array has an empty `sectors` tuple, so `sector` and the delta/data
        # constructor take the sector type from the type rather than inferring it.
        sa = UniqueSectorArray{Float64, U1, 0, 0, 0, Array{Float64, 0}}(fill(2.0), (), ())
        @test ndims(sa) == 0
        @test sectortype(sa) === U1
        with_scalar_indexing() do
            @test sa[] == 2.0
        end

        sd = sector(sa)
        @test sd isa UniqueSectorDelta{Float64, U1, 0}
        @test sectortype(sd) === U1

        rebuilt = UniqueSectorArray(fill(5.0), sd)
        @test rebuilt isa
            UniqueSectorArray{Float64, U1, 0, <:Any, <:Any, Array{Float64, 0}}
        with_scalar_indexing() do
            @test rebuilt[] == 5.0
        end

        # The convenience constructors infer `S` from the axes/sectors, which is
        # impossible for an empty tuple, so they require at least one; a rank-0 value
        # uses the fully-parameterized form above.
        @test_throws MethodError UniqueSectorArray{Float64}(undef, ())
        @test_throws MethodError UniqueSectorDelta{Float64}(())
    end

    @testset "AbstractArray interface — size, getindex, setindex!" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(data, (U1(1), U1(0)))
        @test size(sa) == (2, 2)
        with_scalar_indexing() do
            @test sa[1, 1] == 1.0
            @test sa[2, 1] == 3.0
            @test sa[1, 2] == 2.0
            @test sa[2, 2] == 4.0

            sa[1, 2] = 99.0
            @test sa[1, 2] == 99.0
        end
    end

    @testset "copy" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(data, (U1(1), U1(0)))
        sa2 = copy(sa)
        with_scalar_indexing() do
            @test sa2[1, 1] == sa[1, 1]
            @test sectoraxes(sa2) == sectoraxes(sa)

            sa2[1, 1] = 999.0
            @test sa[1, 1] == 1.0
        end
    end

    @testset "copyto! / broadcast-assign from a plain array" begin
        src = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(zeros(2, 2), (U1(1), conj(U1(1))))
        copyto!(sa, src)
        @test data(sa) == src

        sa2 = UniqueSectorArray(zeros(2, 2), (U1(1), conj(U1(1))))
        sa2 .= src
        @test data(sa2) == src

        sa3 = UniqueSectorArray(zeros(2, 2), (U1(1), conj(U1(1))))
        sa3 .= 2 .* src
        @test data(sa3) == 2 .* src
    end

    @testset "convert" begin
        data = [1 2; 3 4]
        sa = UniqueSectorArray(data, (U1(0), U1(1)))
        T = UniqueSectorArray{Float64, U1, 2, 2, 0, Matrix{Float64}}
        sa2 = convert(T, sa)
        @test eltype(sa2) == Float64
        with_scalar_indexing() do
            @test sa2[1, 1] === 1.0
        end
    end

    @testset "UniqueSectorMatrix alias" begin
        data = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(data, (U1(1), U1(0)))
        @test sa isa UniqueSectorMatrix
    end

    @testset "UniqueSectorVector alias" begin
        data = [1.0, 2.0, 3.0]
        sa = UniqueSectorArray(data, (U1(1),))
        @test sa isa UniqueSectorVector
    end

    @testset "1D UniqueSectorArray" begin
        data = [1.0, 2.0, 3.0]
        sa = UniqueSectorArray(data, (U1(1),))
        @test size(sa) == (3,)
        with_scalar_indexing() do
            @test sa[2] == 2.0
        end
        @test ndims(sa) == 1
    end

    @testset "3D UniqueSectorArray" begin
        data = ones(2, 3, 4)
        sa = UniqueSectorArray(data, (U1(1), conj(U1(0)), U1(-1)))
        @test size(sa) == (2, 3, 4)
        @test ndims(sa) == 3
        with_scalar_indexing() do
            @test sa[1, 2, 3] == 1.0
        end
    end

    @testset "permutedims" begin
        data = [1.0 2.0 3.0; 4.0 5.0 6.0]
        sa = UniqueSectorArray(data, (U1(1), conj(U1(0))))
        sa_perm = permutedims(sa, (2, 1))
        @test size(sa_perm) == (3, 2)
        @test sectoraxes(sa_perm) == (conj(U1(0)), U1(1))
        with_scalar_indexing() do
            @test sa_perm[1, 1] == 1.0
            @test sa_perm[1, 2] == 4.0
        end
    end

    @testset "mul!" begin
        using LinearAlgebra: mul!
        a_data = [1.0 2.0; 3.0 4.0]
        b_data = [5.0 6.0; 7.0 8.0]
        c_data = zeros(2, 2)
        a = UniqueSectorArray(a_data, (U1(0), U1(1)))
        b = UniqueSectorArray(b_data, (conj(U1(1)), U1(0)))
        c = UniqueSectorArray(c_data, (U1(0), U1(0)))
        mul!(c, a, b, 1.0, 0.0)
        @test data(c) ≈ a_data * b_data
    end

    @testset "TensorAlgebra.add! (UniqueSectorArray to UniqueSectorArray)" begin
        using TensorAlgebra: TensorAlgebra
        data1 = [1.0 2.0; 3.0 4.0]
        data2 = [10.0 20.0; 30.0 40.0]
        sa1 = UniqueSectorArray(data1, (U1(0), U1(1)))
        sa2 = UniqueSectorArray(data2, (U1(0), U1(1)))
        TensorAlgebra.add!(sa1, sa2, 2.0, 1.0)
        @test data(sa1) ≈ [21.0 42.0; 63.0 84.0]
    end

    @testset "TensorAlgebra.add! (UniqueSectorArray to plain Array)" begin
        using TensorAlgebra: TensorAlgebra
        dest = zeros(2, 2)
        data = [1.0 2.0; 3.0 4.0]
        sa = UniqueSectorArray(data, (U1(0), U1(1)))
        TensorAlgebra.add!(dest, sa, 3.0, 0.0)
        @test dest ≈ [3.0 6.0; 9.0 12.0]
    end

    @testset "fill! abelian" begin
        sa = UniqueSectorArray([1.0 2.0; 3.0 4.0], (U1(0), dual(U1(0))))
        fill!(sa, 7.0)
        @test all(==(7.0), data(sa))

        fill!(sa, 0.0)
        @test all(iszero, data(sa))
    end

    @testset "fill! non-abelian sets the reduced data" begin
        # `fill!` is a shorthand for setting the symmetry-allowed (reduced) values, like `rand!`, so
        # it fills the reduced data even for a non-abelian sector (it is not a dense-array fill).
        sa = UniqueSectorArray(ones(2, 2), (SU2(1 // 2), dual(SU2(1 // 2))))
        fill!(sa, 3.0)
        @test all(==(3.0), data(sa))

        fill!(sa, 0.0)
        @test all(iszero, data(sa))
    end

    @testset "zero!" begin
        using TensorAlgebra: TensorAlgebra
        sa = UniqueSectorArray([1.0 2.0; 3.0 4.0], (U1(0), dual(U1(0))))
        TensorAlgebra.zero!(sa)
        @test all(iszero, data(sa))
    end

    @testset "real / imag (data-wise, keeps sector)" begin
        d = randn(ComplexF64, 2, 2)
        sa = UniqueSectorArray(d, (U1(1), dual(U1(1))))
        ra = real(sa)
        ia = imag(sa)
        @test ra isa UniqueSectorArray
        @test ia isa UniqueSectorArray
        # The structural sector factor is left intact; only the reduced data takes real/imag parts.
        @test sectoraxes(ra) == sectoraxes(sa)
        @test data(ra) == real.(d)
        @test data(ia) == imag.(d)
        @test data(ra) + im * data(ia) ≈ d
    end

    @testset "split block: round-trip, real/imag, eltype independence" begin
        d = randn(ComplexF64, 2, 3)
        sa = UniqueSectorArray(d, (U1(1),), (U1(1),))   # a genuine (1, 1) split
        sd = sector(sa)
        @test (length(sd.sectors_codomain), length(sd.sectors_domain)) == (1, 1)
        # The domain leg is stored codomain-facing but reads as dual externally.
        @test sectoraxes(sa) == (U1(1), dual(U1(1)))
        # Exact structural round-trip keeps the split (and the data object).
        @test sector_kron(sector(sa), data(sa)) === sa
        # real/imag keep the split and are eltype-independent: the delta's `T` need not track the data's.
        ra = real(sa)
        ia = imag(sa)
        @test (length(sector(ra).sectors_codomain), length(sector(ra).sectors_domain)) ==
            (1, 1)
        @test eltype(ra) == Float64
        @test sectoraxes(ra) == sectoraxes(sa)
        @test data(ra) + im * data(ia) ≈ d
    end
end

@testset "UniqueSectorDelta is not a matrix (no `tr`)" begin
    # Matrix operations are defined only on the matrix storage types (`FusedSectorMatrix` /
    # `FusedGradedMatrix`) and, among the structural deltas, only `SectorIdentity`. `UniqueSectorDelta`
    # is a general structural delta, so `tr` (a matrix operation) errors on it.
    @test_throws ErrorException tr(UniqueSectorDelta{Float64}((U1(1), dual(U1(1)))))
    @test_throws ErrorException tr(
        UniqueSectorDelta{Float64}((SU2(1 // 2), dual(SU2(1 // 2))))
    )
end
