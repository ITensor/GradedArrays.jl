using Aqua: Aqua
using GradedArrays: GradedArrays
using TensorAlgebra: TensorAlgebra
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # `to_range` is deliberately extended on pairs keyed by a bare `TensorKitSectors.Sector`, and
    # by a tuple or named tuple of sectors (type piracy, since GradedArrays owns neither
    # `to_range` nor those key types) so that a sector spelling usable with `Sector` is also usable
    # as an axis descriptor. `treat_as_own` allowlists the whole function, so it covers each of
    # those methods; the piracy is tracked for rehoming onto a GradedArrays-owned entry point.
    Aqua.test_piracies(GradedArrays; treat_as_own = [TensorAlgebra.to_range])
end
