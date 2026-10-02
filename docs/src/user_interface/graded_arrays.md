# Graded arrays

```@meta
CurrentModule = GradedArrays
```

## Graded spaces

[`gradedrange`](@ref) builds a graded space from `sector => multiplicity` pairs.

```@example gradedarrays
using GradedArrays: U1, gradedrange
g = gradedrange([U1(0) => 1, U1(1) => 2])
```

The total length of the space is 3, and [`sectors`](@ref) returns the list of sectors.

```@example gradedarrays
using GradedArrays: sectors
length(g), sectors(g)
```

```@docs; canonical=false
gradedrange
GradedOneTo
sectors
```

## Duality

A graded space carries a duality, and `dual` flips it.

```@example gradedarrays
using GradedArrays: dual, isdual
dg = dual(g)
isdual(g), isdual(dg)
```

`conj` is an alternative for `dual`.

```@example gradedarrays
conj(g) == dual(g)
```

## Arrays over graded spaces

Array constructors accept graded spaces and return a [`GradedArray`](@ref), which only stores
the symmetry-allowed blocks.

```@example gradedarrays
a = randn(g, dual(g))
```

`zeros`, `ones`, and `fill` work the same way.

```@example gradedarrays
zeros(g, dual(g))
```

```@example gradedarrays
fill(2.0, g, dual(g))
```

You can also specify an element type.

```@example gradedarrays
randn(ComplexF64, g, dual(g))
```

## Codomain and domain

A graded array partitions its indices into a codomain and a domain, and stores the block
diagonal matrix corresponding to that bipartitioning. `matricize` fuses the codomain and domain
according to a specified bipartitioning.

```@example gradedarrays
using TensorAlgebra: matricize
matricize(a, (1,), (2,))
```

You can specify the codomain/domain split in graded array constructors, where the domain is
implicitly dualized. For example, `randn((g,), (g,))` has axes `(g, dual(g))` with one axis in
the codomain and one in the domain.

```@example gradedarrays
b = randn((g,), (g,))
```

```@example gradedarrays
axes(b) == axes(a)
```

When printing, by convention domain axes are implicitly dual. The format and conventions are
compatible with those of [TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

```@docs; canonical=false
GradedArray
FusedGradedMatrix
FusedGradedVector
```
