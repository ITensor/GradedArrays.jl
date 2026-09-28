"""
    Sector

A sector: an irreducible label of a symmetry, and the range of the degrees of freedom that
label spans, so `length` is the sector's dimension.

A sector carries no arrow. Duality lives on [`OrientedSector`](@ref) and on the axis types,
which is the convention `GradedOneTo` already follows.

Concrete sectors are GradedArrays' own types storing their own label, so `typeof(U1(0))` is
`U1`. `TensorKitSectors` supplies the fusion rules, ordering and symbols, reached through
[`tensorkitsector`](@ref), and constructors route through it to reuse its validation.
"""
abstract type Sector <: AbstractUnitRange{Int} end

"""
    tensorkitsector(s::Sector) -> TensorKitSectors.Sector

The TensorKitSectors sector `s` stands for. Every concrete [`Sector`](@ref) defines this, and
gets the fusion rules, ordering and symbols through it.
"""
function tensorkitsector end

"""
    tensorkitsectortype(::Type{<:Sector}) -> Type{<:TensorKitSectors.Sector}

The type-level counterpart of [`tensorkitsector`](@ref), which the fusion and braiding traits
dispatch through. Every concrete [`Sector`](@ref) defines it.
"""
tensorkitsectortype(s::Sector) = tensorkitsectortype(typeof(s))

"""
    TensorKitSector(c::TensorKitSectors.Sector)

Any TensorKitSectors sector as a [`Sector`](@ref). This is the escape hatch for the symmetries
GradedArrays does not give a name of its own, such as the anyons.
"""
struct TensorKitSector{I <: TKS.Sector} <: Sector
    sector::I
end
tensorkitsector(s::TensorKitSector) = s.sector
tensorkitsectortype(::Type{TensorKitSector{I}}) where {I} = I

"""
    to_sector(c::TensorKitSectors.Sector) -> Sector

The GradedArrays sector for a TensorKitSectors one, the inverse of [`tensorkitsector`](@ref).
A symmetry GradedArrays names itself comes back as that name, anything else as a
[`TensorKitSector`](@ref).
"""
to_sector(s::Sector) = s
to_sector(c::TKS.Sector) = TensorKitSector(c)

sectortype(x) = sectortype(typeof(x))
sectortype(S::Type{<:Sector}) = S
sectortype(T::Type) = throw(MethodError(sectortype, T))

# ===================================  Base interface  =====================================

Base.length(s::Sector) = TKS.dim(tensorkitsector(s))
Base.OneTo(s::Sector) = Base.OneTo(length(s))
Base.first(s::Sector) = first(Base.OneTo(s))
Base.last(s::Sector) = last(Base.OneTo(s))
Base.axes(s::Sector) = (s,)

# Ordering and equality run through TensorKitSectors so that two sectors GradedArrays spells
# differently, such as a named type and the same symmetry reached as a `TensorKitSector`,
# still compare as the sector they both denote.
Base.isless(s1::Sector, s2::Sector) = isless(tensorkitsector(s1), tensorkitsector(s2))
Base.isless(s1::Sector, c2::TKS.Sector) = isless(tensorkitsector(s1), c2)
Base.isless(c1::TKS.Sector, s2::Sector) = isless(c1, tensorkitsector(s2))
Base.:(==)(s1::Sector, s2::Sector) = tensorkitsector(s1) == tensorkitsector(s2)
Base.:(==)(s1::Sector, c2::TKS.Sector) = tensorkitsector(s1) == c2
Base.:(==)(c1::TKS.Sector, s2::Sector) = c1 == tensorkitsector(s2)
Base.isequal(s1::Sector, s2::Sector) = isequal(tensorkitsector(s1), tensorkitsector(s2))
Base.hash(s::Sector, h::UInt) = hash(tensorkitsector(s), h)

Base.show(io::IO, s::Sector) = print(io, sector_name(s))

# =================================  Sectors interface  ====================================

"""
    label(s::Sector)

The value labelling the sector, such as the charge of a `U1` or the spin of an `SU2`.
"""
function label end

trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
    return gradedrange([trivial(sectortype(axis_type)) => 1])  # always returns nondual
end
trivial(type::Type) = error("`trivial` not defined for type $(type).")
trivial(S::Type{<:Sector}) = to_sector(one(tensorkitsectortype(S)))
trivial(::Type{I}) where {I <: TKS.Sector} = one(I)

istrivial(s::Sector) = isone(tensorkitsector(s))
istrivial(x) = (x == trivial(x))

to_gradedrange(s::Sector) = gradedrange([s => 1])
to_gradedrange(c::TKS.Sector) = to_gradedrange(to_sector(c))

function nsymbol(s1::Sector, s2::Sector, s3::Sector)
    return TKS.Nsymbol(tensorkitsector(s1), tensorkitsector(s2), tensorkitsector(s3))
end

twist(s::Sector) = TKS.twist(tensorkitsector(s))

"""
    charge_conjugate(s::Sector) -> Sector

The conjugate sector, TensorKitSectors' `dual`. This is a different operation from
[`dual`](@ref), which flips a sector's arrow and so returns an [`OrientedSector`](@ref).
"""
charge_conjugate(s::Sector) = to_sector(TKS.dual(tensorkitsector(s)))

# A total version of `TensorKitSectors.fermionparity`. TKS defines it only for
# `FermionParity`, `NamedSector`, and `ProductSector`, and its `ProductSector` method maps
# `fermionparity` over the components, so it errors on a bosonic component such as the
# `U1Irrep` in `FermionNumber = U1Irrep ⊠ FermionParity`. We delegate to TKS where it is
# defined, add the bosonic-irrep case a plain group irrep is bosonic, hence even parity, and
# decompose product sectors over their components.
fermionparity(s::Sector) = fermionparity(tensorkitsector(s))
fermionparity(c::TKS.Sector) = TKS.fermionparity(c)
fermionparity(::TKS.AbstractIrrep) = false
fermionparity(c::TKS.ProductSector) = mapreduce(fermionparity, ⊻, c.sectors)

# ===============================  Fusion rule interface  ==================================

for trait in (
        :FusionStyle,
        :BraidingStyle,
        :sectorscalartype,
        :fusionscalartype,
        :braidingscalartype,
    )
    @eval TKS.$trait(::Type{S}) where {S <: Sector} = TKS.$trait(tensorkitsectortype(S))
end

function fusion_rule(s1::Sector, s2::Sector)
    a = tensorkitsector(s1)
    b = tensorkitsector(s2)
    fstyle = TKS.FusionStyle(typeof(s1)) & TKS.FusionStyle(typeof(s2))
    fstyle === TKS.UniqueFusion() && return to_sector(only(TKS.otimes(a, b)))
    return gradedrange(
        vec([to_sector(c) => Int(TKS.Nsymbol(a, b, c)) for c in TKS.otimes(a, b)])
    )
end

# =============================  TensorProducts interface  =================================

tensor_product(s::Sector) = s
tensor_product(s1::Sector, s2::Sector) = fusion_rule(s1, s2)
function tensor_product(c1::TKS.Sector, c2::TKS.Sector)
    return tensor_product(to_sector(c1), to_sector(c2))
end
tensor_product(s1::Sector, c2::TKS.Sector) = tensor_product(s1, to_sector(c2))
tensor_product(c1::TKS.Sector, s2::Sector) = tensor_product(to_sector(c1), s2)

# ==================================  OrientedSector  ======================================

"""
    OrientedSector(s::Sector, isdual::Bool)

A sector together with an arrow. This is where a bare sector acquires a duality, and what
[`dual`](@ref) of a sector returns.
"""
struct OrientedSector{S <: Sector} <: AbstractUnitRange{Int}
    sector::S
    isdual::Bool
end
OrientedSector(s::Sector) = OrientedSector(s, false)
OrientedSector(s::OrientedSector) = s

sector(s::OrientedSector) = s.sector
sector(s::Sector) = s
sectortype(::Type{OrientedSector{S}}) where {S} = S
tensorkitsector(s::OrientedSector) = tensorkitsector(sector(s))
tensorkitsectortype(::Type{OrientedSector{S}}) where {S} = tensorkitsectortype(S)

TensorAlgebra.isdual(s::OrientedSector) = s.isdual
TensorAlgebra.isdual(::Sector) = false

# `dual` flips the arrow and leaves the label alone, so a bare sector has to gain an arrow to
# carry the result. `flip` is the composite of the arrow flip with charge conjugation.
TensorAlgebra.dual(s::Sector) = OrientedSector(s, true)
TensorAlgebra.dual(s::OrientedSector) = OrientedSector(sector(s), !isdual(s))
flip(s::OrientedSector) = OrientedSector(charge_conjugate(sector(s)), !isdual(s))
flip(s::Sector) = OrientedSector(charge_conjugate(s), true)
flip_dual(s::OrientedSector) = isdual(s) ? flip(s) : s
flip_dual(s::Sector) = s
nondual(s::OrientedSector) = OrientedSector(sector(s), false)
nondual(s::Sector) = s
Base.conj(s::Sector) = dual(s)
Base.conj(s::OrientedSector) = dual(s)

# An arrow does not change whether a sector is trivial, its fermion parity, or its twist.
istrivial(s::OrientedSector) = istrivial(sector(s))
fermionparity(s::OrientedSector) = fermionparity(sector(s))
twist(s::OrientedSector) = twist(sector(s))
to_gradedrange(s::OrientedSector) = GradedOneTo([sector(s)], [1], isdual(s))
function to_sector(s::OrientedSector)
    return throw(
        ArgumentError(
            "a graded axis stores non-dual sectors, pass the arrow through the axis instead of `$(s)`"
        )
    )
end

"""
    AnySector

A sector with or without an arrow, for the places that accept either. `splitarrows` takes a
tuple of them apart into the sectors and the arrows, which is how a type stores them.
"""
const AnySector = Union{Sector, OrientedSector}
splitarrows(ss::Tuple{Vararg{AnySector}}) = (map(sector, ss), map(isdual, ss))

Base.length(s::OrientedSector) = length(sector(s))
Base.OneTo(s::OrientedSector) = Base.OneTo(length(s))
Base.first(s::OrientedSector) = first(Base.OneTo(s))
Base.last(s::OrientedSector) = last(Base.OneTo(s))
Base.axes(s::OrientedSector) = (s,)

function Base.:(==)(s1::OrientedSector, s2::OrientedSector)
    return sector(s1) == sector(s2) && isdual(s1) == isdual(s2)
end
Base.:(==)(s1::OrientedSector, s2::Sector) = !isdual(s1) && sector(s1) == s2
Base.:(==)(s1::Sector, s2::OrientedSector) = s2 == s1
function Base.isequal(s1::OrientedSector, s2::OrientedSector)
    return isequal(sector(s1), sector(s2)) && isequal(isdual(s1), isdual(s2))
end
Base.isless(s1::OrientedSector, s2::OrientedSector) = isless(sector(s1), sector(s2))
Base.hash(s::OrientedSector, h::UInt) = hash(sector(s), hash(isdual(s), h))

function Base.show(io::IO, s::OrientedSector)
    # Print duals as `dual(...)` rather than using a trailing `'`: Julia already uses `'` for
    # adjoint of ranges (`(1:4)'` returns a 1×N adjoint matrix), so reusing `'` here is
    # ambiguous.
    isdual(s) || return show(io, sector(s))
    print(io, "dual(")
    show(io, sector(s))
    print(io, ")")
    return nothing
end

trivial(::Type{OrientedSector{S}}) where {S} = OrientedSector(trivial(S))
function fusion_rule(s1::OrientedSector, s2::OrientedSector)
    return fusion_rule(sector(flip_dual(s1)), sector(flip_dual(s2)))
end
tensor_product(s::OrientedSector) = sector(flip_dual(s))
# Fusing folds each arrow into its sector, so the result is bare whatever the arguments were.
# The mixed arities are what keeps a `reduce` closed: after the first step one side is already
# bare while the rest are still oriented.
tensor_product(s1::OrientedSector, s2::OrientedSector) = fusion_rule(s1, s2)
tensor_product(s1::OrientedSector, s2::Sector) = fusion_rule(sector(flip_dual(s1)), s2)
tensor_product(s1::Sector, s2::OrientedSector) = fusion_rule(s1, sector(flip_dual(s2)))

# =====================================  Sectors  ==========================================

"""
    TrivialSector()

The sector of the trivial symmetry, the unit of every fusion.
"""
struct TrivialSector <: Sector end
tensorkitsector(::TrivialSector) = TKS.Trivial()
tensorkitsectortype(::Type{TrivialSector}) = TKS.Trivial
label(::TrivialSector) = nothing
to_sector(::TKS.Trivial) = TrivialSector()

# The trivial sector fuses with anything, so it promotes to the other sector's type.
Base.promote_rule(::Type{TrivialSector}, ::Type{S}) where {S <: Sector} = S
Base.convert(::Type{S}, ::TrivialSector) where {S <: Sector} = trivial(S)

# TensorKitSectors has no ordering or fusion between `Trivial` and a real irrep, so the generic
# methods above cannot answer these. The same-type methods break the ambiguity the mixed ones
# would otherwise create.
Base.:(==)(::TrivialSector, ::TrivialSector) = true
Base.:(==)(::TrivialSector, s::Sector) = istrivial(s)
Base.:(==)(s::Sector, ::TrivialSector) = istrivial(s)
Base.isless(::TrivialSector, ::TrivialSector) = false
Base.isless(::TrivialSector, s::Sector) = isless(trivial(typeof(s)), s)
Base.isless(s::Sector, ::TrivialSector) = isless(s, trivial(typeof(s)))
fusion_rule(s::TrivialSector, ::TrivialSector) = s
function fusion_rule(::TrivialSector, s::Sector)
    return TKS.FusionStyle(typeof(s)) === TKS.UniqueFusion() ? s : to_gradedrange(s)
end
function fusion_rule(s::Sector, ::TrivialSector)
    return TKS.FusionStyle(typeof(s)) === TKS.UniqueFusion() ? s : to_gradedrange(s)
end

"""
    Z{N}(n::Integer)

An irreducible representation of the cyclic group of order `N`.
"""
struct Z{N} <: Sector
    n::Int8
    # Constructed through TensorKitSectors so the modular reduction and the `N < 64` bound are
    # checked in one place rather than restated here.
    Z{N}(n::Integer) where {N} = new{N}(TKS.ZNIrrep{N}(n).n)
end
const Z2 = Z{2}
tensorkitsector(s::Z{N}) where {N} = TKS.ZNIrrep{N}(s.n)
tensorkitsectortype(::Type{Z{N}}) where {N} = TKS.ZNIrrep{N}
to_sector(c::TKS.ZNIrrep{N}) where {N} = Z{N}(c.n)
label(s::Z) = Int(s.n)
modulus(::Z{N}) where {N} = N

"""
    U1(charge::Real)

An irreducible representation of `U(1)`, labelled by its charge.
"""
struct U1 <: Sector
    charge::HalfInt
    U1(charge::Real) = new(TKS.U1Irrep(charge).charge)
end
tensorkitsector(s::U1) = TKS.U1Irrep(s.charge)
tensorkitsectortype(::Type{U1}) = TKS.U1Irrep
to_sector(c::TKS.U1Irrep) = U1(c.charge)
label(s::U1) = s.charge

"""
    SU2(j::Real)

An irreducible representation of `SU(2)`, labelled by its spin.
"""
struct SU2 <: Sector
    j::HalfInt
    # Constructed through TensorKitSectors so its non-negative half-integer check applies.
    SU2(j::Real) = new(TKS.SU2Irrep(j).j)
end
tensorkitsector(s::SU2) = TKS.SU2Irrep(s.j)
tensorkitsectortype(::Type{SU2}) = TKS.SU2Irrep
to_sector(c::TKS.SU2Irrep) = SU2(c.j)
label(s::SU2) = s.j

"""
    fZ2(isodd::Bool)

Fermion parity, the fermionic analog of `Z2`.
"""
struct fZ2 <: Sector
    isodd::Bool
end
tensorkitsector(s::fZ2) = TKS.FermionParity(s.isodd)
tensorkitsectortype(::Type{fZ2}) = TKS.FermionParity
to_sector(c::TKS.FermionParity) = fZ2(c.isodd)
label(s::fZ2) = Int(s.isodd)

label(s::TensorKitSector) = sector_label(tensorkitsector(s))
function sector_label(c::TKS.Sector)
    return map(f -> getfield(c, f), fieldnames(typeof(c)))
end
# A product sector's one field is already the component tuple, so the generic method above would
# wrap it in a second tuple. Its label is its components.
sector_label(c::TKS.ProductSector) = map(to_sector, c.sectors)

# The GradedArrays name of a sector type, falling back to the TensorKitSectors type name for a
# symmetry reached through `TensorKitSector`.
sector_typename(::Type{S}) where {S <: Sector} = string(nameof(S))
sector_typename(::Type{Z{N}}) where {N} = "Z{$N}"
sector_typename(::Type{TensorKitSector{I}}) where {I} = string(nameof(I))

# Constructor-form display of a sector value: `name(label)`.
function sector_name(s::Sector)
    l = label(s)
    return string(sector_typename(typeof(s)), '(', isnothing(l) ? "" : sprint(show, l), ')')
end

# A `TKS.ProductSector` has no name of its own, so it shows as its components rather than
# leaking `ProductSector` into the display. Components go through `to_sector` so each prints
# under its GradedArrays name. Add a method per named product sector, as `FermionNumber` does.
function sector_name(s::TensorKitSector{<:TKS.ProductSector})
    parts = map(c -> sector_name(to_sector(c)), tensorkitsector(s).sectors)
    return string('(', join(parts, " × "), ')')
end
function sector_name(
        s::TensorKitSector{TKS.ProductSector{Tuple{TKS.U1Irrep, TKS.FermionParity}}}
    )
    c = tensorkitsector(s)
    q = c.sectors[1].charge
    # Recover the alias only for a value `FermionNumber` actually produces; any other
    # `(U1, FermionParity)` product shows as its components.
    isinteger(q) && TKS.FermionNumber(Int(q)) == c ||
        return @invoke sector_name(s::TensorKitSector{<:TKS.ProductSector})
    return "FermionNumber($(Int(q)))"
end
