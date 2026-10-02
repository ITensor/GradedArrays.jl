# GradedArrays.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/GradedArrays.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/GradedArrays.jl/dev/)
[![Build Status](https://github.com/ITensor/GradedArrays.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/GradedArrays.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/GradedArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/GradedArrays.jl)
[![Code Style](https://img.shields.io/badge/code_style-ITensor-purple)](https://github.com/ITensor/ITensorFormatter.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Support

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
  <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
</picture>


GradedArrays.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

## Installation instructions

This package resides in the `ITensor/ITensorRegistry` local registry.
In order to install, simply add that registry through your package manager.
This step is only required once.
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
or:
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

Then, the package can be added as usual through the package manager:

```julia
julia> Pkg.add("GradedArrays")
```

## Examples

A `GradedArray` is an array over spaces graded by symmetry sectors, and it stores only the
blocks the symmetry allows. Build the spaces from `sector => multiplicity` pairs and pass
them to the standard Julia array constructors. `dual` gives the dual of a space.

````julia
using GradedArrays: U1, dual, gradedrange
g = gradedrange([U1(0) => 1, U1(1) => 2])
a = randn(g, dual(g))
````

`zeros`, `ones`, and `fill` work the same way, and allocate only the allowed blocks.

````julia
zeros(g, dual(g))
````

A `GradedArray` behaves like any other array. Scale one,

````julia
2 * a
````

add two over the same spaces,

````julia
a + randn(g, dual(g))
````

or permute the dimensions.

````julia
permutedims(a, (2, 1))
````

For the symmetries that are available, see
[Symmetry sectors](https://itensor.github.io/GradedArrays.jl/dev/user_interface/sectors/). For
building graded spaces and arrays over them, see
[Graded arrays](https://itensor.github.io/GradedArrays.jl/dev/user_interface/graded_arrays/).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

