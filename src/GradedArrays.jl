module GradedArrays

include("gradedunitrange_interface.jl")
include("symmetry_style.jl")

include("sectorunitrange.jl")
include("gradedunitrange.jl")

include("abstractsector.jl")
include("sector_definitions/fib.jl")
include("sector_definitions/ising.jl")
include("sector_definitions/o2.jl")
include("sector_definitions/trivial.jl")
include("sector_definitions/su.jl")
include("sector_definitions/su2k.jl")
include("sector_definitions/u1.jl")
include("sector_definitions/zn.jl")
include("namedtuple_operations.jl")
include("sector_product.jl")

include("fusion.jl")
include("gradedarray.jl")
include("tensoralgebra.jl")
include("factorizations.jl")

export SU2,
    U1,
    Z,
    dag,
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
