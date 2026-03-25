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
    ungrade

# imports
# -------
import FunctionImplementations as FI
using ArrayLayouts: ArrayLayouts
using BlockArrays: BlockArrays, AbstractBlockArray, AbstractBlockVector,
    AbstractBlockedUnitRange, Block, BlockIndexRange, BlockedOneTo, blockisequal, blocks,
    blocksize, eachblockaxes1
using BlockSparseArrays: BlockSparseArrays, @view!, AbstractBlockSparseArray,
    AnyAbstractBlockSparseArray, BlockOneTo, BlockSparseArray, BlockUnitRange, blockrange,
    blockreshape, blocktype, eachblockstoredindex, sparsemortar
using KroneckerArrays: KroneckerArrays, AbstractKroneckerArray, CartesianProductUnitRange,
    cartesianrange, kroneckerfactors, unproduct, ×
using LinearAlgebra: LinearAlgebra, Adjoint, mul!
using SparseArraysBase: isstored
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockedTuple, FusionStyle,
    ReshapeFusion, matricize, matricize_axes, tensor_product_axis, trivialbiperm,
    tuplemortar, unmatricize
using TensorKitSectors: TensorKitSectors as TKS
using TypeParameterAccessors:
    similartype, type_parameters, unspecify_type_parameters, unwrap_array_type

include("sectorrange.jl")
include("sectorarray.jl")
include("gradedarray.jl")
include("broadcast.jl")

include("sectorproduct.jl")

include("fusion.jl")
include("tensoralgebra.jl")
include("factorizations.jl")

end
