# Symmetry sectors

```@meta
CurrentModule = GradedArrays
```

A sector is an irreducible label of a symmetry, such as a `U(1)` charge or an `SU(2)` spin. It
is also the range of the degrees of freedom that label spans, so `length` of a sector is its
dimension.

```@example sectors
using GradedArrays: SU2, U1
length(U1(1)), length(SU2(1//2))
```

Sectors grade an axis, making it a graded axis. See [Graded arrays](@ref) for building graded
axes and the arrays over them.

## Available sectors

| Sector | Symmetry | Example |
| :----- | :------- | :------ |
| [`Trivial`](@ref) | none | `Trivial()` |
| [`Z`](@ref), [`Z2`](@ref) | cyclic group of order `N` | `Z{3}(1)`, `Z2(1)` |
| [`U1`](@ref) | `U(1)` | `U1(-1)` |
| [`SU2`](@ref) | `SU(2)` | `SU2(1//2)` |
| [`SU`](@ref) | `SU(N)` | `SU{3}(1, 1)` |
| [`CU1`](@ref) | `U(1) ⋊ C`, also called `O(2)` | `CU1(1)` |
| [`fZ2`](@ref) | fermion parity | `fZ2(true)` |
| [`fU1`](@ref) | fermion number | `fU1(1)` |
| [`fSU2`](@ref) | fermion spin | `fSU2(1//2)` |
| [`TensorKitSector`](@ref) | a wrapped TensorKitSectors sector | `TensorKitSector(c)` |

The fermionic ones carry the parity their charge forces, so each takes only the charge.

```@example sectors
using GradedArrays: fSU2, fU1, fZ2
fU1(1), fSU2(1//2), fZ2(true)
```

[`TensorKitSector`](@ref) wraps a TensorKitSectors sector and reinterprets it as a sector here,
which is what supports the symmetries TensorKitSectors defines that have none of their own, such
as the anyons. [`Sector`](@ref) converts any TensorKitSectors
sector, to the GradedArrays name when there is one and to `TensorKitSector` when there is
not.

```@example sectors
using GradedArrays: Sector
using TensorKitSectors: FibonacciAnyon, SU2Irrep
Sector(SU2Irrep(1//2)), Sector(FibonacciAnyon(:τ))
```

```@docs; canonical=false
Sector
Trivial
Z
Z2
U1
SU2
SU
CU1
fZ2
fU1
fSU2
TensorKitSector
```

## Products of sectors

[`sectorproduct`](@ref) combines symmetries, and `×` is the same thing infix. `Trivial` is its
unit and drops out of any product.

```@example sectors
using GradedArrays: SU2, Sector, U1, sectorproduct, ×
sectorproduct(U1(1), SU2(1//2))
```

```@example sectors
U1(1) × SU2(1//2)
```

`sectorproduct`/`×` also works on types, for example `fU1` is an alias for `U1 × fZ2`.

```@example sectors
using GradedArrays: fZ2
U1 × fZ2
```

[`Sector`](@ref) gives another way to make the same product.

```@example sectors
Sector(U1(1), SU2(1//2)) == U1(1) × SU2(1//2)
```

A product can also name its factors instead of ordering them.

```@example sectors
Sector(; charge = U1(1), spin = SU2(1//2))
```

Anything [`Sector`](@ref) accepts can be passed where a sector is expected and is converted
there, so a product needs no `Sector` call of its own.

```@example sectors
using GradedArrays: gradedrange
gradedrange([
    (charge = U1(0), spin = SU2(0)) => 1, (charge = U1(1), spin = SU2(1//2)) => 2,
])
```

```@docs; canonical=false
sectorproduct
```
