module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export SectorRange, SectorOneTo, GradedOneTo
export AbstractSectorDelta, AbelianSectorDelta, SectorIdentity
export AbstractSectorArray,
    AbelianSectorArray, AbelianSectorVector, AbelianSectorMatrix,
    SectorMatrix, SectorVector
export AbstractGradedArray, AbstractGradedMatrix
export AbelianGradedArray, AbelianGradedVector, AbelianGradedMatrix
export FusedGradedMatrix, FusedGradedVector
export GradedBlockAlgorithm

export dual, flip, gradedrange, isdual,
    data, dataaxes, dataaxes1, datalength, datalengths,
    eachdataaxis, eachsectoraxis,
    sector, sectoraxes, sectoraxes1, sectorlength, sectorlengths,
    sectors, sectortype,
    Data

# imports
# -------
using BlockArrays: BlockArrays, AbstractBlockArray, AbstractBlockVector,
    AbstractBlockedUnitRange, Block, BlockIndexRange, BlockVector, BlockedOneTo,
    blockedrange, blocklasts, blocklength, blocklengths, blocks, eachblockaxes1
using BlockSparseArrays: BlockSparseArrays, blockdiagindices, blockstoredlength,
    eachblockaxis, eachblockstoredindex, mortar_axis
using Dictionaries: Dictionaries, Dictionary, dictionary, gettoken, gettokenvalue
using FunctionImplementations: FunctionImplementations as FI
using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, kron, mul!
using Random: Random, AbstractRNG
using SparseArraysBase: SparseArraysBase
using TensorAlgebra: TensorAlgebra, BlockedTuple, FusionStyle, bipermutedimsopadd!,
    check_input, matricize, matricize_axes, permutedimsadd!, tensor_product_axis,
    trivial_axis, trivialbiperm, tryflattenlinear, unmatricize
using TensorKitSectors: TensorKitSectors as TKS

include("kron.jl")
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
include("fusedgradedvector.jl")

include("sectorproduct.jl")

include("broadcast.jl")
include("fusion.jl")
include("tensoralgebra.jl")

include("matrixalgebrakit.jl")

end
