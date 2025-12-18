module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export SectorRange, SectorDelta
export SectorUnitRange, SectorOneTo, SectorArray, SectorMatrix
export GradedUnitRange, GradedOneTo, GradedArray
export gradedrange

export sector_type

export dag, dual, flip, gradedrange, isdual,
    sector, sector_multiplicities, sector_multiplicity,
    sectorrange, sectors, sector_type,
    space_isequal, ungrade

# imports
# -------
using LinearAlgebra: LinearAlgebra, Adjoint
using KroneckerArrays
using KroneckerArrays: AbstractKroneckerArray, CartesianProductUnitRange
using SparseArraysBase: SparseArraysBase, isstored
using BlockArrays: BlockArrays, Block
using BlockSparseArrays: AbstractBlockSparseArray, blockrange, @view!

include("sectorrange.jl")
include("sectorarray.jl")
include("gradedarray.jl")

include("sectorproduct.jl")

include("fusion.jl")
include("tensoralgebra.jl")
include("factorizations.jl")


end
