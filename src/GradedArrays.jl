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
using BlockArrays: BlockArrays, AbstractBlockVector, AbstractBlockedUnitRange, Block,
    BlockIndexRange, BlockVector, BlockedOneTo, blockedrange, blocklasts, blocklength,
    blocklengths, blocks, eachblockaxes1
using Dictionaries: Dictionaries, Dictionary, dictionary, gettoken, gettokenvalue
using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, dot, kron, mul!
using Preferences: @load_preference, @set_preferences!
using Random: Random, AbstractRNG, rand!, randn!
using SparseArraysBase: SparseArraysBase, AbstractSparseMatrix
using TensorAlgebra: TensorAlgebra, TensorAlgebra as TA, FusionStyle, bipartition,
    bipermutedims!, bipermutedimsopadd!, check_input, dual, flattenlinear, isdual,
    matricize, permutedimsadd!, scale!, unmatricize, zero!
using TensorKitSectors: TensorKitSectors as TKS
using VectorInterface: VectorInterface as VI

# The graded backend `allocate_graded` builds, chosen in one place via a compile-time preference.
# Defaults to the block-sparse `AbelianGradedArray`; set to "fusion" (see `set_graded_backend!`) to
# build the always-fused `FusionArray` everywhere. A `@load_preference` constant (baked in at
# precompile), so `@static if graded_backend == …` keeps the backend branches type-stable with the
# unused branch eliminated. Defined here, before the includes, so it is available to `@static` in any
# included file. Temporary switch to develop `FusionArray` toward parity with `AbelianGradedArray`.
const graded_backend = @load_preference("graded_backend", "abelian")
graded_backend in ("abelian", "fusion") ||
    error(
    "graded_backend preference must be \"abelian\" or \"fusion\", got $(repr(graded_backend))"
)

include("kron.jl")
include("blocksparseinterface.jl")
include("sectorrange.jl")
include("data.jl")
include("sectoroneto.jl")
include("gradedoneto.jl")
include("tensorkit.jl")
include("abstractsectordelta.jl")
include("abstractsectorarray.jl")
include("abeliansectordelta.jl")
include("abeliansectorarray.jl")
include("sectoridentity.jl")
include("sectoronesvector.jl")
include("sectormatrix.jl")
include("abstractgradedarray.jl")
include("abeliangradedarray.jl")

include("fusedgradedmatrix.jl")
include("fusedgradedvector.jl")
include("fusedgradedblocks.jl")

include("sectorproduct.jl")

include("broadcast.jl")
include("fusion.jl")
include("tensoralgebra.jl")
include("vectorinterface.jl")
include("cat.jl")

include("matrixalgebrakit.jl")

include("fusionarray.jl")
include("fusionmap.jl")

end
