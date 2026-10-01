# Changelog

```@meta
CurrentModule = GradedArrays
```

## [0.17.0](https://github.com/ITensor/GradedArrays.jl/compare/v0.16.6...v0.17.0) - 2026-10-01

Gives the sector types a hierarchy of their own and moves duality off of them and onto the
axis.

### Breaking changes

- `SectorRange` is removed, and the sector types are no longer aliases of it. `U1` was
  `SectorRange{TensorKitSectors.U1Irrep}` and is now a GradedArrays type of its own, as are the
  rest, under a new [`Sector`](@ref) supertype. A sector carries no arrow, so duality lives on
  the axis, which is where `dual` and `isdual` already applied
  ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
- `TrivialSector` is now [`Trivial`](@ref), and it equals only itself. It was
  `SectorRange{TensorKitSectors.Trivial}`, and comparing it against another symmetry's trivial
  sector now returns `false`, so `Trivial() == U1(0)` where it used to hold. A `U1`-graded space
  with a single zero-charge block is not an ungraded space
  ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
- `sectors` is exported rather than `public`. It is a new name in the exported surface, so code
  doing `using GradedArrays` alongside another package that exports `sectors`, TensorKit among
  them, has to qualify it ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).

### Non-breaking changes

- [`Sector`](@ref) is the single entry point for building a sector from anything that specifies
  one, a TensorKitSectors sector and a product of sectors included
  ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
- [`SU`](@ref), [`CU1`](@ref), [`fZ2`](@ref), [`fU1`](@ref) and [`fSU2`](@ref) are named
  sectors, covering `SU(N)`, `O(2)`, and the three fermionic symmetries
  ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
- [`sectorproduct`](@ref), written `×`, combines symmetries positionally or by name, and works
  on types so a fused symmetry can be named
  ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
- [`TensorKitSector`](@ref) carries any symmetry TensorKitSectors defines but GradedArrays does
  not name, such as the anyons ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
- `GradedOneTo`, `FusedGradedMatrix` and `FusedGradedVector` are `public`
  ([#285](https://github.com/ITensor/GradedArrays.jl/pull/285)).
