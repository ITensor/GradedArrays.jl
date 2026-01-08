using GradedArrays: U1, dual, gradedrange, isdual
using NamedDimsArrays: denamed, named
using Test: @test, @testset

@testset "GradedArraysNamedDimsArraysExt" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    nr = named(r, "i")
    nr_dual = dual(nr)
    @test isdual(nr_dual)
    @test isdual(denamed(nr_dual))
end
