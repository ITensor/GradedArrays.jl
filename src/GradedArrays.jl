module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export SectorRange, SectorDelta
export SectorUnitRange, SectorOneTo, SectorArray, SectorMatrix
export GradedUnitRange, GradedOneTo, GradedArray
export gradedrange

export dual, flip, gradedrange, isdual,
    sector, sector_multiplicities, sector_multiplicity,
    sectorrange, sectors, sector_type,
    space_isequal, ungrade

# imports
# -------
import KroneckerArrays: ×
using BlockArrays: BlockArrays, Block, blocksize
using BlockSparseArrays: @view!, AbstractBlockSparseArray, BlockOneTo, blockrange
using KroneckerArrays
using KroneckerArrays: AbstractKroneckerArray, CartesianProductUnitRange
using LinearAlgebra: LinearAlgebra, Adjoint
using SparseArraysBase: isstored
using TypeParameterAccessors: type_parameters, unspecify_type_parameters

include("sectorrange.jl")
include("sectorarray.jl")
include("gradedarray.jl")

include("sectorproduct.jl")

include("fusion.jl")
include("tensoralgebra.jl")
include("factorizations.jl")

end
