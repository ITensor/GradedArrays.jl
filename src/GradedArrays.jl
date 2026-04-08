module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export SectorRange, SectorDelta, SectorIndices, GradedIndices
export SectorArray, SectorMatrix
export AbstractGradedArray, AbelianArray, FusedSectorMatrix
export gradedrange

export dual, flip, gradedrange, isdual,
    labels,
    sector, sector_multiplicities, sector_multiplicity,
    sectorrange, sectors, sector_type

# imports
# -------
import FunctionImplementations as FI
using BlockArrays:
    BlockArrays, AbstractBlockVector, Block, BlockVector, blocklength, blocklengths, blocks
using BlockSparseArrays:
    BlockSparseArrays, @view!, eachblockaxis, eachblockstoredindex, mortar_axis, view!
using KroneckerArrays: KroneckerArrays, kroneckerfactors, ×
using LinearAlgebra: LinearAlgebra, Adjoint, mul!
using TensorAlgebra: TensorAlgebra, BlockedTuple, FusionStyle, matricize, permutedimsadd!,
    permutedimsopadd!, trivialbiperm, tryflattenlinear
using TensorKitSectors: TensorKitSectors as TKS
using TypeParameterAccessors: type_parameters, unspecify_type_parameters

include("sectorrange.jl")
include("sectorindices.jl")
include("gradedindices.jl")
include("sectorarray.jl")
include("abstractgradedarray.jl")
include("abelianarray.jl")

include("fusedsectormatrix.jl")

include("sectorproduct.jl")

include("broadcast.jl")
include("fusion.jl")
include("tensoralgebra.jl")

end
