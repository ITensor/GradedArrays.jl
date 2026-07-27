using Aqua: Aqua
using GradedArrays: GradedArrays
using TensorAlgebra: TensorAlgebra
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # `to_range` is deliberately extended on bare `TensorKitSectors.Sector`-keyed pairs (type
    # piracy, since GradedArrays owns neither `to_range` nor `TKS.Sector`) so raw sectors work as
    # axis descriptors. `treat_as_own` allowlists that one method; the piracy is tracked for
    # rehoming onto a GradedArrays-owned entry point.
    Aqua.test_piracies(GradedArrays; treat_as_own = [TensorAlgebra.to_range])
end
