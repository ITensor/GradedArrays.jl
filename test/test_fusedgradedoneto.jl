using BlockArrays: blocklength, blocklengths
using Dictionaries: Dictionary
using GradedArrays: GradedArrays, AbstractGradedOneTo, FusedGradedOneTo, GradedOneTo, SU2,
    SectorRange, U1, datalengths, dual, flip, fusedgradedrange, gradedrange, isdual, label,
    sectorlengths, sectors, sectortype
using TensorAlgebra: TensorAlgebra
using Test: @test, @test_throws, @testset

@testset "FusedGradedOneTo" begin
    @testset "fusedgradedrange from SectorRange (U1)" begin
        g = fusedgradedrange([U1(0) => 2, U1(1) => 3])
        @test g isa FusedGradedOneTo{U1}
        @test g isa AbstractGradedOneTo{U1}
        @test sectors(g) == [U1(0), U1(1)]
        @test sectors(g) isa Vector{U1}
        @test datalengths(g) == [2, 3]
        @test isdual(g) == false
    end

    @testset "rejects non-canonical pairs" begin
        @test_throws ArgumentError fusedgradedrange([U1(1) => 3, U1(0) => 2])   # unsorted
        @test_throws Exception fusedgradedrange([U1(0) => 2, U1(0) => 1])       # repeated
    end

    @testset "empty" begin
        g = fusedgradedrange(U1[] .=> Int[])
        @test g isa FusedGradedOneTo{U1}
        @test isempty(sectors(g))
        @test length(g) == 0
    end

    @testset "dual axis via dual; dual sectors rejected" begin
        # `fusedgradedrange` builds a non-dual axis; the arrow is applied separately with `dual`.
        g = dual(fusedgradedrange([U1(0) => 2, U1(1) => 3]))
        @test isdual(g) == true
        @test sectors(g) == [U1(0), U1(1)]   # stored non-dual
        @test datalengths(g) == [2, 3]
        @test_throws ArgumentError fusedgradedrange([conj(U1(0)) => 2])   # dual sector rejected
    end

    @testset "constructors reject unsorted / accept a Dictionary" begin
        @test_throws ArgumentError FusedGradedOneTo([U1(1), U1(0)], [3, 2])
        d = Dictionary{U1, Int}([U1(0), U1(1)], [2, 3])
        g = FusedGradedOneTo(d)            # arrow defaults to non-dual
        @test isdual(g) == false
        @test sectors(g) == [U1(0), U1(1)]
        @test isdual(FusedGradedOneTo(d, true))
    end

    @testset "range interface (shared via AbstractGradedOneTo)" begin
        g = fusedgradedrange([U1(0) => 2, U1(1) => 3])
        @test first(g) == 1
        @test length(g) == 5
        @test axes(g) == (g,)
        @test blocklength(g) == 2                     # number of sectors
        @test blocklengths(g) == [2, 3]               # length(sector) * mult
        @test sectorlengths(g) == [1, 1]              # abelian: length(sector) == 1
        @test sectortype(g) === U1
        @test sectortype(typeof(g)) === U1
    end

    @testset "non-abelian SU2 blocklengths carry the quantum dimension" begin
        g = fusedgradedrange([SU2(0) => 2, SU2(1 // 2) => 1])
        @test sectors(g) == [SU2(0), SU2(1 // 2)]
        @test datalengths(g) == [2, 1]
        @test sectorlengths(g) == [1, 2]              # dim(SU2(0))=1, dim(SU2(1/2))=2
        @test blocklengths(g) == [2, 2]               # 1*2 and 2*1
        @test length(g) == 4
    end

    @testset "dual (conj) flips the arrow, keeps stored order" begin
        g = fusedgradedrange([U1(0) => 2, U1(1) => 3])
        gd = dual(g)
        @test isdual(gd) == true
        @test conj(g) == gd
        @test sectors(gd) == sectors(g)
        @test datalengths(gd) == datalengths(g)
        @test dual(dual(g)) == g
    end

    @testset "flip conjugates labels and re-sorts" begin
        # Sectors stored in canonical (TensorKit) order: U1(1) sorts before U1(-1).
        g = fusedgradedrange([U1(1) => 3, U1(-1) => 2])
        @test sectors(g) == [U1(1), U1(-1)]
        @test datalengths(g) == [3, 2]
        f = flip(g)
        @test isdual(f) == true
        # Dualizing the labels reorders them, so `flip` re-sorts to restore canonical form;
        # the data lengths follow the resort.
        @test sectors(f) == [U1(1), U1(-1)]
        @test datalengths(f) == [2, 3]
    end

    @testset "equality and hashing" begin
        g = fusedgradedrange([U1(0) => 2, U1(1) => 3])
        @test g == fusedgradedrange([U1(0) => 2, U1(1) => 3])
        @test hash(g) == hash(fusedgradedrange([U1(0) => 2, U1(1) => 3]))
        @test g != dual(g)
        # An empty axis still carries an arrow, so a dual and non-dual empty axis differ.
        @test fusedgradedrange(U1[] .=> Int[]) != dual(fusedgradedrange(U1[] .=> Int[]))
    end

    @testset "conversion to/from GradedOneTo" begin
        g = gradedrange([U1(0) => 2, U1(1) => 3])   # already in fused form
        fg = FusedGradedOneTo(g)
        @test fg isa FusedGradedOneTo{U1}
        @test sectors(fg) == [U1(0), U1(1)]
        @test datalengths(fg) == [2, 3]

        back = GradedOneTo(fg)
        @test back isa GradedOneTo{U1}
        @test sectors(back) == sectors(fg)
        @test datalengths(back) == datalengths(fg)
        @test isdual(back) == isdual(fg)

        # The conversion checks rather than normalizes: a GradedOneTo not already in fused
        # form is rejected.
        @test_throws ArgumentError FusedGradedOneTo(gradedrange([U1(1) => 3, U1(0) => 2]))  # unsorted
        @test_throws Exception FusedGradedOneTo(gradedrange([U1(0) => 2, U1(0) => 1]))      # repeated
    end

    @testset "eachblockaxis / eachsectoraxis apply the arrow" begin
        g = fusedgradedrange([U1(0) => 2, U1(1) => 3])
        @test GradedArrays.eachsectoraxis(g) == [U1(0), U1(1)]
        @test GradedArrays.eachsectoraxis(dual(g)) == [conj(U1(0)), conj(U1(1))]
    end

    @testset "GradedOneTo is also an AbstractGradedOneTo" begin
        @test gradedrange([U1(0) => 2]) isa AbstractGradedOneTo
        @test fusedgradedrange([U1(0) => 2]) isa AbstractGradedOneTo
    end

    @testset "show" begin
        # Match structurally (the sector type may print module-qualified depending on scope).
        g = fusedgradedrange([U1(0) => 2, U1(1) => 3])
        @test occursin("fusedgradedrange([", repr(g))
        @test occursin("=> 2", repr(g)) && occursin("=> 3", repr(g))
        @test !startswith(repr(g), "dual(")
        @test startswith(repr(dual(g)), "dual(fusedgradedrange([")
    end
end
