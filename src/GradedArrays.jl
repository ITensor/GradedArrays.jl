module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export SectorRange, SectorOneTo, GradedOneTo
export AbstractSectorDelta, AbelianSectorDelta, SectorIdentity
export AbstractSectorArray,
    AbelianSectorArray, AbelianSectorVector, AbelianSectorMatrix,
    SectorMatrix
export AbstractGradedArray, AbstractGradedMatrix
export AbelianGradedArray, AbelianGradedVector, AbelianGradedMatrix
export FusedGradedMatrix

export dual, flip, gradedrange, isdual,
    data, dataaxes, dataaxes1, datalength, datalengths,
    eachdataaxis, eachsectoraxis,
    sector, sectoraxes, sectoraxes1, sectorlength, sectorlengths,
    sectors, sectortype,
    Data

# imports
# -------
import FunctionImplementations as FI
using BlockArrays: BlockArrays, AbstractBlockArray, AbstractBlockVector,
    AbstractBlockedUnitRange, Block, BlockIndexRange, BlockVector, BlockedOneTo,
    blockedrange, blocklasts, blocklength, blocklengths, blocks, eachblockaxes1
using BlockSparseArrays: BlockSparseArrays, blockdiagindices, blockstoredlength,
    eachblockaxis, eachblockstoredindex, mortar_axis
using KroneckerArrays: KroneckerArrays, kroneckerfactors, ×, ⊗
using LinearAlgebra: LinearAlgebra, Adjoint, mul!
using SparseArraysBase: SparseArraysBase
using TensorAlgebra: TensorAlgebra, BlockedTuple, FusionStyle, bipermutedimsopadd!,
    matricize, matricize_axes, permutedimsadd!, tensor_product_axis, trivial_axis,
    trivialbiperm, tryflattenlinear, unmatricize
using TensorKitSectors: TensorKitSectors as TKS
using TypeParameterAccessors: type_parameters, unspecify_type_parameters

include("sectorrange.jl")
include("data.jl")
include("sectoroneto.jl")
include("gradedoneto.jl")
include("abstractsectordelta.jl")
include("abstractsectorarray.jl")
include("abeliansectordelta.jl")
include("abeliansectorarray.jl")
include("sectoridentity.jl")
include("sectormatrix.jl")
include("abstractgradedarray.jl")
include("abeliangradedarray.jl")

include("fusedgradedmatrix.jl")

include("sectorproduct.jl")

include("broadcast.jl")
include("fusion.jl")
include("tensoralgebra.jl")

end
