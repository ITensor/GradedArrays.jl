# Graded arrays

```@meta
CurrentModule = GradedArrays
```

## Graded axes

[`gradedrange`](@ref) builds a graded axis from `sector => multiplicity` pairs. Each pair
contributes one block, whose length is the multiplicity times the sector's dimension.

```@example gradedarrays
using GradedArrays: U1, gradedrange
g = gradedrange([U1(0) => 1, U1(1) => 2])
```

Read the sectors back with [`sectors`](@ref), which gives one per block in block order.

```@example gradedarrays
using GradedArrays: sectors
sectors(g)
```

```@docs; canonical=false
gradedrange
GradedOneTo
sectors
```

## Duality

A graded axis carries an arrow saying whether it transforms in a representation or in its
dual. `dual` flips the arrow and leaves the sectors alone, `conj` is an alternative spelling
of it, and `isdual` asks which way it points. GradedArrays re-exports all three from
TensorAlgebra, so their docstrings live there and they are shown here by example rather than
in the [Reference](@ref).

```@example gradedarrays
using GradedArrays: dual, isdual
dg = dual(g)
isdual(g), isdual(dg)
```

```@example gradedarrays
conj(g) == dual(g)
```

Duality belongs to the graded axis, not to the sectors it carries, so the sectors come back
unchanged.

```@example gradedarrays
sectors(dg) == sectors(g)
```

## Arrays over graded axes

Calling a `Base` array constructor on graded axes gives a [`GradedArray`](@ref), which stores
only the symmetry-allowed blocks.

```@example gradedarrays
a = randn(g, dual(g))
```

## Codomain and domain

A graded array partitions its legs into a codomain and a domain, so it reads as a map between
spaces, and stores the block diagonal matrix that bipartitioning gives. Both legs landed in the
codomain above, which is why the stored form there is a single column. `matricize` is that
regrouping on its own, fusing the codomain legs to the rows and the domain legs to the columns.

```@example gradedarrays
using TensorAlgebra: matricize
matricize(a, (1,), (2,))
```

The split can also be given up front, as a codomain tuple and a domain tuple. Domain axes are
passed facing the same way as codomain ones and are dualized for you, so `randn((g,), (g,))`
has the same axes as `randn(g, dual(g))` and differs only in how they are grouped. It is
stored as the matrix directly.

```@example gradedarrays
b = randn((g,), (g,))
```

```@example gradedarrays
axes(b) == axes(a)
```

When printing, by convention domain axes are implicitly dual, shown the way they were passed
rather than the way `axes` returns them. The format and conventions are compatible with those of
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

```@docs; canonical=false
GradedArray
FusedGradedMatrix
FusedGradedVector
```
