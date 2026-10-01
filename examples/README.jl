# # GradedArrays.jl
#
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/GradedArrays.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/GradedArrays.jl/dev/)
# [![Build Status](https://github.com/ITensor/GradedArrays.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/GradedArrays.jl/actions/workflows/Tests.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/ITensor/GradedArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/GradedArrays.jl)
# [![Code Style](https://img.shields.io/badge/code_style-ITensor-purple)](https://github.com/ITensor/ITensorFormatter.jl)
# [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# ## Support
#
# {CCQ_LOGO}
#
# GradedArrays.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

# ## Installation instructions

# This package resides in the `ITensor/ITensorRegistry` local registry.
# In order to install, simply add that registry through your package manager.
# This step is only required once.
#=
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
=#
# or:
#=
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
=#
# if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

# Then, the package can be added as usual through the package manager:

#=
```julia
julia> Pkg.add("GradedArrays")
```
=#

# ## Examples

# A graded axis is built from `sector => multiplicity` pairs, one block per pair. This one is
# graded by `U(1)` charge, with a one-dimensional zero-charge block and a two-dimensional
# charge-one block.

using GradedArrays: U1, dual, gradedrange, isdual, sectors
g = gradedrange([U1(0) => 1, U1(1) => 2])

# The sectors come back one per block.

sectors(g)

# An axis also carries an arrow saying whether it transforms in a representation or in its dual.
# `dual` flips the arrow and leaves the sectors alone.

isdual(g), isdual(dual(g))

# Calling an array constructor on graded axes gives an array that stores only the
# symmetry-allowed blocks.

a = randn(g, dual(g))

#=
For the symmetries that are available, see
[Symmetry sectors](https://itensor.github.io/GradedArrays.jl/dev/user_interface/sectors/). For
building axes and arrays over them, see
[Graded arrays](https://itensor.github.io/GradedArrays.jl/dev/user_interface/graded_arrays/).
=#
