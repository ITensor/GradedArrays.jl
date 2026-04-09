module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, O2, SU2, Fib, Ising
export SectorRange, SectorOneTo, GradedOneTo
export AbstractSectorDelta, AbelianSectorDelta, SectorIdentity
export AbstractSectorArray, AbelianSectorArray, AbelianSectorMatrix, SectorMatrix
export AbstractGradedArray, AbstractGradedMatrix
export AbelianGradedArray, AbelianGradedVector, AbelianGradedMatrix
export FusedGradedMatrix
export gradedrange

export dual, flip, gradedrange, isdual,
    labels,
    data, dataaxes, dataaxes1,
    sector, sectoraxes, sectoraxes1,
    sector_multiplicities, sector_multiplicity,
    sectorrange, sectors, sector_type

# imports
# -------
import FunctionImplementations as FI
using BlockArrays: BlockArrays, AbstractBlockVector, Block, BlockIndexRange, BlockVector,
    blocklength, blocklengths, blocks
using BlockSparseArrays:
    BlockSparseArrays, eachblockaxis, eachblockstoredindex, mortar_axis, view!
using KroneckerArrays: KroneckerArrays, kroneckerfactors, ×
using LinearAlgebra: LinearAlgebra, Adjoint, mul!
using TensorAlgebra: TensorAlgebra, BlockedTuple, FusionStyle, matricize, matricize_axes,
    permutedimsadd!, permutedimsopadd!, tensor_product_axis, trivial_axis, trivialbiperm,
    tryflattenlinear, unmatricize
using TensorKitSectors: TensorKitSectors as TKS
using TypeParameterAccessors: type_parameters, unspecify_type_parameters

include("sectorrange.jl")
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
