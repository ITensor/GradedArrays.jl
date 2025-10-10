module GradedArrays

include("gradedunitrange_interface.jl")

include("sectorrange.jl")
include("sectorunitrange.jl")
include("gradedunitrange.jl")

include("namedtuple_operations.jl")
include("sectorproduct.jl")

include("fusion.jl")
include("gradedarray.jl")
include("tensoralgebra.jl")
include("factorizations.jl")

export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export dag,
    dual,
    flip,
    gradedrange,
    isdual,
    sector,
    sector_multiplicities,
    sector_multiplicity,
    sectorrange,
    sectors,
    sector_type,
    space_isequal,
    ungrade

end
