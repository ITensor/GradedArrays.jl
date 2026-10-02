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

Sectors grade a space, making it a graded space. See [Graded arrays](@ref) for building
graded spaces and the arrays over them.

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

The fermionic ones carry the parity their charge forces, so each takes only the charge.

```@example sectors
using GradedArrays: fSU2, fU1, fZ2
fU1(1), fSU2(1//2), fZ2(true)
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
```

## Products of sectors

[`sectorproduct`](@ref) creates products of symmetry sectors, which can also be written as
`×`. `Trivial` is its unit and drops out of any product.

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

[`Sector`](@ref) is used to convert to sector types in functions such as `gradedrange`:

```@example sectors
using GradedArrays: gradedrange
gradedrange([
    (charge = U1(0), spin = SU2(0)) => 1, (charge = U1(1), spin = SU2(1//2)) => 2,
])
```

```@docs; canonical=false
sectorproduct
```

## TensorKitSectors compatibility

The sectors here correspond to the ones in
[TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl), and convert both
ways: [`Sector`](@ref) takes a TensorKitSectors sector, and `TensorKitSectors.Sector` takes one
of these.

```@example sectors
using GradedArrays: Sector
using TensorKitSectors: TensorKitSectors, SU2Irrep
Sector(SU2Irrep(1//2)), TensorKitSectors.Sector(SU2(1//2))
```

Fusion rules and the rest of a sector's topological data come from that conversion rather than
from definitions of their own, so the two packages agree on them by construction.

`SU` is the exception. It can be constructed and compared on its own, but everything that needs
`SU(N)` representation theory, its dimension and its fusion rule included, comes from
[SUNRepresentations.jl](https://github.com/QuantumKitHub/SUNRepresentations.jl) through a
package extension. A graded space over `SU` sectors needs that package loaded.
