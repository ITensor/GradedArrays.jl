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

Sectors are what grade an axis. See [Graded arrays](@ref) for building axes and arrays out of
them.

## Available sectors

| Sector | Symmetry | Example |
| :----- | :------- | :------ |
| [`Trivial`](@ref) | none | `Trivial()` |
| [`Z`](@ref), [`Z2`](@ref) | cyclic group of order `N` | `Z{3}(1)`, `Z2(1)` |
| [`U1`](@ref) | `U(1)` | `U1(-1)` |
| [`SU2`](@ref) | `SU(2)` | `SU2(1//2)` |
| [`SU`](@ref) | `SU(N)` | `SU{3}(1, 1)` |
| [`CU1`](@ref) | `U(1) ⋊ C`, also written `O(2)` | `CU1(1)` |
| [`fZ2`](@ref) | fermion parity | `fZ2(true)` |
| [`fU1`](@ref) | fermion number | `fU1(1)` |
| [`fSU2`](@ref) | fermion spin | `fSU2(1//2)` |
| [`TensorKitSector`](@ref) | any symmetry TensorKitSectors defines | `TensorKitSector(c)` |

The fermionic ones carry the parity their charge forces, so each takes only the charge.

```@example sectors
using GradedArrays: fSU2, fU1, fZ2
fU1(1), fSU2(1//2), fZ2(true)
```

Anything TensorKitSectors defines but GradedArrays does not name, such as the anyons, is
available through [`TensorKitSector`](@ref).

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

[`sectorproduct`](@ref), written `×`, combines symmetries. `Trivial` is its unit and drops out
of any product.

```@example sectors
using GradedArrays: SU2, Sector, U1, ×
U1(1) × SU2(1//2)
```

[`Sector`](@ref) of two or more sectors is the same product, so take whichever reads better at
the call site.

```@example sectors
Sector(U1(1), SU2(1//2)) == U1(1) × SU2(1//2)
```

A product can name its factors instead of ordering them, which is what you want once there are
several and position stops being memorable. Positional and named products do not mix.

```@example sectors
Sector(; charge = U1(1), spin = SU2(1//2))
```

The product also works on types, which is how a fused symmetry gets a name of its own. `fU1` is
defined as `U1 × fZ2`.

```@docs; canonical=false
sectorproduct
```
