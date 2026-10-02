# Fermionic symmetries: a charge together with the parity that charge forces. TensorKitSectors
# has the same two as `FermionNumber` and `FermionSpin`, which it also aliases `fU₁` and `fSU₂`;
# these are those aliases in ASCII, over GradedArrays' own sector types.

"""
    const fU1 = U1 × fZ2
    fU1(n::Integer) -> fU1

Fermion number: a `U1` charge together with its parity, which is odd exactly when the charge is
odd. The parity follows from the charge, so the constructor takes only the charge.

See also [`U1`](@ref), [`fZ2`](@ref) and [`fSU2`](@ref).
"""
const fU1 = U1 × fZ2
fU1(n::Integer) = Sector(U1(n), fZ2(isodd(n)))
sectortype_repr(::Type{fU1}) = "fU1"

"""
    const fSU2 = SU2 × fZ2
    fSU2(j::Real) -> fSU2

Fermion spin: an `SU2` spin together with its parity, which is odd exactly when `2j` is odd. The
parity follows from the spin, so the constructor takes only the spin.

See also [`SU2`](@ref), [`fZ2`](@ref) and [`fU1`](@ref).
"""
const fSU2 = SU2 × fZ2
fSU2(j::Real) = (s = SU2(j); Sector(s, fZ2(isodd(twice(s.j)))))
sectortype_repr(::Type{fSU2}) = "fSU2"

# An alias shows under its own name only for a value it actually produces. Any other product of
# the same two symmetries is not a fermionic sector, and shows as its components.
function Base.show(io::IO, s::fU1)
    n = only(sector_labels(first(arguments(s))))
    isinteger(n) && fU1(Int(n)) == s || return @invoke show(io::IO, s::TupleSectorProduct)
    return print(io, "fU1(", Int(n), ")")
end
function Base.show(io::IO, s::fSU2)
    j = only(sector_labels(first(arguments(s))))
    fSU2(j) == s || return @invoke show(io::IO, s::TupleSectorProduct)
    return print(io, "fSU2(", j, ")")
end
