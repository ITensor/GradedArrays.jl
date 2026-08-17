module GradedArrays

# exports
# -------
export TrivialSector, Z, Z2, U1, SU2
export SectorRange, SectorOneTo, AbstractGradedOneTo, GradedOneTo, FusedGradedOneTo
export AbstractSectorDelta, UniqueSectorDelta, SectorIdentity
export AbstractSectorArray,
    UniqueSectorArray, UniqueSectorVector, UniqueSectorMatrix,
    FusedSectorMatrix, FusedSectorVector
export GradedArray
export FusedGradedMatrix, FusedGradedVector, FusedGradedDiagonal
export GradedBlockAlgorithm

export codomain, domain,
    dual, flip, gradedrange, fusedgradedrange, isdual,
    data, dataaxes, dataaxes1, datalength, datalengths,
    eachdataaxis, eachsectoraxis,
    sector, sectoraxes, sectoraxes1, sectorlength, sectorlengths,
    sectors, sectortype,
    Data

if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public with_scalar_indexing, with_block_indexing"))
end

# imports
# -------
using BlockArrays: BlockArrays, AbstractBlockVector, AbstractBlockedUnitRange, Block,
    BlockIndexRange, BlockVector, BlockedArray, BlockedOneTo, block, blockedrange,
    blocklasts, blocklength, blocklengths, blocks, eachblockaxes1
using Dictionaries: Dictionaries, Dictionary, dictionary, gettoken, gettokenvalue
using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, dot, kron, mul!
using Random: Random, AbstractRNG, rand!, randn!
using TensorAlgebra: TensorAlgebra, TensorAlgebra as TA, BiTuple, MatricizeStyle,
    bipartition, bipermutedims!, bipermutedimsopadd!, check_input, dual, flattenlinear,
    isdual, matricize, permutedimsadd!, scale!, unmatricize, zero!
using TensorKitSectors: TensorKitSectors as TKS
using VectorInterface: VectorInterface as VI

# Self-alias for calling our own functions qualified where a local argument shadows the bare name
# (e.g. a `sectordata` constructor argument shadowing the `sectordata` accessor).
const GA = GradedArrays

include("compat.jl")
include("indexingguards.jl")
include("kron.jl")
include("blocksparseinterface.jl")
include("sparsearrays.jl")
include("sectorrange.jl")
include("data.jl")
include("sectoroneto.jl")
include("abstractgradedoneto.jl")
include("gradedoneto.jl")
include("fusedgradedoneto.jl")
include("tensorkit.jl")
include("abstractsectordelta.jl")
include("abstractsectorarray.jl")
include("uniquesectordelta.jl")
include("uniquesectorarray.jl")
include("sectoridentity.jl")
include("sectoronesvector.jl")
include("fusedsectormatrix.jl")
include("abstractfusedgradedarray.jl")

include("sectordata.jl")
include("fusedgradedmatrix.jl")
include("fusedgradedvector.jl")
# The graded broadcast-style lattice is defined here, before `FusedGradedDiagonal` (which adds its
# own `FusedGradedDiagonalStyle` under the lattice) and `GradedArray` (which adds `GradedStyle`).
include("broadcast.jl")
include("fusedgradeddiagonal.jl")
include("adjointfusedgradedarray.jl")
include("fusedgradedblocks.jl")

include("sectorproduct.jl")

include("fusion.jl")
include("tensoralgebra.jl")
include("cat.jl")

include("matrixalgebrakit.jl")

include("gradedarray.jl")
# Shared graded-array interface + `VectorInterface` (both name `GradedArray`, so they come after it).
include("gradedarrayinterface.jl")
include("vectorinterface.jl")
include("gradedconstructors.jl")

end
