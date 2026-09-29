"""
    Sector

A sector: an irreducible label of a symmetry, and the range of the degrees of freedom that
label spans, so `length` is the sector's dimension.

A sector carries no arrow. Duality lives on [`OrientedSector`](@ref) and on the axis types,
which is the convention `GradedOneTo` already follows.

Concrete sectors are GradedArrays' own types storing their own label, so `typeof(U1(0))` is
`U1`. `TensorKitSectors` supplies the fusion rules, ordering and symbols, and each direction
between the two packages is spelled as the constructor of the type being asked for: every
concrete sector defines `TensorKitSectors.Sector(s)` for the upstream sector it stands for, and
`Sector` takes an upstream sector going the other way. Construction routes through upstream to
reuse its validation, and `tensorkit_sectortype` is the type-level counterpart.

    Sector(s::Sector) -> Sector
    Sector(c::TensorKitSectors.Sector) -> Sector
    Sector(s1, s2, srest...) -> TupleSectorProduct
    Sector(t::Tuple) -> TupleSectorProduct
    Sector(nt::NamedTuple) -> NamedSectorProduct
    Sector(; kws...) -> NamedSectorProduct
    Sector() -> TrivialSector

Calling `Sector` normalizes a sector specification, and is the one place that defines what
counts as one. A sector comes back as itself, a `TensorKitSectors` sector as the GradedArrays
name for that symmetry or else a [`TensorKitSector`](@ref), two or more sectors as the product
over them, and no sectors at all as [`TrivialSector`](@ref). Everything that takes a sector from
a caller, `gradedrange` and the array constructors included, routes through here, so the
accepted spellings are the same everywhere.
"""
abstract type Sector <: AbstractUnitRange{Int} end

"""
    tensorkit_sectortype(::Type{<:Sector}) -> Type{<:TensorKitSectors.Sector}

The type-level counterpart of `TensorKitSectors.Sector(s)`, which the fusion and braiding
traits dispatch through. Every concrete [`Sector`](@ref) defines it.
"""
tensorkit_sectortype(s::Sector) = tensorkit_sectortype(typeof(s))

# A sector converts by handing its labels to its counterpart's own constructor, which takes them
# in the same order and is the spelling the sector's display prints. A type therefore declares
# the counterpart and its labels, and this direction of the conversion follows.
TKS.Sector(s::Sector) = tensorkit_sectortype(s)(sector_labels(s)...)

"""
    TensorKitSector(c::TensorKitSectors.Sector)

Any TensorKitSectors sector as a [`Sector`](@ref). This is the escape hatch for the symmetries
GradedArrays does not give a name of its own, such as the anyons.
"""
struct TensorKitSector{I <: TKS.Sector} <: Sector
    sector::I
end
TKS.Sector(s::TensorKitSector) = s.sector
tensorkit_sectortype(::Type{TensorKitSector{I}}) where {I} = I

Sector(s::Sector) = s
Sector(c::TKS.Sector) = TensorKitSector(c)

sectortype(x) = sectortype(typeof(x))
sectortype(S::Type{<:Sector}) = S
sectortype(T::Type) = throw(MethodError(sectortype, T))

# ===================================  Base interface  =====================================

Base.length(s::Sector) = TKS.dim(TKS.Sector(s))
Base.OneTo(s::Sector) = Base.OneTo(length(s))
Base.first(s::Sector) = first(Base.OneTo(s))
Base.last(s::Sector) = last(Base.OneTo(s))
Base.axes(s::Sector) = (s,)

# Ordering and equality run through TensorKitSectors so that two sectors GradedArrays spells
# differently, such as a named type and the same symmetry reached as a `TensorKitSector`,
# still compare as the sector they both denote.
function Base.isless(s1::Sector, s2::Sector)
    return isless(TKS.Sector(s1), TKS.Sector(s2))
end
Base.:(==)(s1::Sector, s2::Sector) = TKS.Sector(s1) == TKS.Sector(s2)
# Comparison does not cross the library boundary: a sector here and the TensorKitSectors sector
# it converts to are values of two libraries' types, and `Sector` is how you move between them.
# Defining `==` across would also oblige `hash` to agree, which it cannot, since it takes one
# operand and so has to read one library's notion of identity. Products make that concrete: a
# named product equals one that leaves a trivially-valued name out, while the `NamedSector`s they
# convert to differ, so equating either with its own converted form would not even be transitive.
# `isequal` delegates to `==` and must never reimplement it. Reimplementing is what let the two
# drift, so that hash-based containers disagreed with `==` about padded products. Leaving it
# undefined is not an option: a `Sector` is an `AbstractUnitRange{Int}`, so it would inherit
# Base's `AbstractArray` method rather than the scalar `isequal(x, y) = x == y` fallback, and
# agree only by way of a sector being its own axis.
Base.isequal(s1::Sector, s2::Sector) = s1 == s2
Base.hash(s::Sector, h::UInt) = hash(TKS.Sector(s), h)

# =================================  Sectors interface  ====================================

"""
    sector_labels(s::Sector)

The labels of the sector, as a tuple in the order the constructor takes them: the charge of a
`U1`, the spin of an `SU2`, the Dynkin labels of an `SU`. A `TrivialSector` has nothing
labelling it and gives an empty tuple.

Splatting them back into the constructor gives the sector again, which is the spelling the
display of a sector prints. These are not the struct's fields: a stored label is normalized
where the type has a canonical form for it, and a [`TensorKitSector`](@ref) reaches through to
the labels of the sector it wraps, whose type is also the constructor its display names.

A `SectorProduct` has none of its own, since what it holds are sectors rather than the values
labelling one, and it displays as those components.
"""
function sector_labels end

trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
    return gradedrange([trivial(sectortype(axis_type)) => 1])  # always returns nondual
end
trivial(type::Type) = error("`trivial` not defined for type $(type).")
trivial(S::Type{<:Sector}) = Sector(one(tensorkit_sectortype(S)))
trivial(::Type{I}) where {I <: TKS.Sector} = one(I)

istrivial(s::Sector) = isone(TKS.Sector(s))
istrivial(x) = (x == trivial(x))

to_gradedrange(s::Sector) = gradedrange([s => 1])
to_gradedrange(c::TKS.Sector) = to_gradedrange(Sector(c))

# A method of upstream's function rather than a parallel name, matching how `TKS.Sector`,
# `TKS.FusionStyle` and `TKS.BraidingStyle` are extended for these types. A GradedArrays sector is
# not a `TKS.Sector`, so the inner call lands on upstream's own methods and cannot recurse.
function TKS.Nsymbol(s1::Sector, s2::Sector, s3::Sector)
    return TKS.Nsymbol(
        TKS.Sector(s1),
        TKS.Sector(s2),
        TKS.Sector(s3)
    )
end

twist(s::Sector) = TKS.twist(TKS.Sector(s))

"""
    dual_sector(s::Sector) -> Sector

The conjugate sector, which is what TensorKitSectors calls `dual`. Here `dual` is the arrow
operation instead, since a [`Sector`](@ref) is itself the range of its degrees of freedom and
so `dual` of one is the dual space, returning an [`OrientedSector`](@ref).
"""
dual_sector(s::Sector) = Sector(TKS.dual(TKS.Sector(s)))

# A total version of `TensorKitSectors.fermionparity`. TKS defines it only for
# `FermionParity`, `NamedSector`, and `ProductSector`, and its `ProductSector` method maps
# `fermionparity` over the components, so it errors on a bosonic component such as the
# `U1Irrep` in `FermionNumber = U1Irrep ⊠ FermionParity`. We delegate to TKS where it is
# defined, add the bosonic-irrep case a plain group irrep is bosonic, hence even parity, and
# decompose product sectors over their components.
fermionparity(s::Sector) = fermionparity(TKS.Sector(s))
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
    @eval TKS.$trait(::Type{S}) where {S <: Sector} = TKS.$trait(tensorkit_sectortype(S))
end

function fusion_rule(s1::Sector, s2::Sector)
    a = TKS.Sector(s1)
    b = TKS.Sector(s2)
    fstyle = TKS.FusionStyle(typeof(s1)) & TKS.FusionStyle(typeof(s2))
    fstyle === TKS.UniqueFusion() && return Sector(only(TKS.otimes(a, b)))
    return gradedrange(
        vec([Sector(c) => Int(TKS.Nsymbol(a, b, c)) for c in TKS.otimes(a, b)])
    )
end

# =============================  TensorProducts interface  =================================

tensor_product(s::Sector) = s
tensor_product(s1::Sector, s2::Sector) = fusion_rule(s1, s2)
function tensor_product(c1::TKS.Sector, c2::TKS.Sector)
    return tensor_product(Sector(c1), Sector(c2))
end
tensor_product(s1::Sector, c2::TKS.Sector) = tensor_product(s1, Sector(c2))
tensor_product(c1::TKS.Sector, s2::Sector) = tensor_product(Sector(c1), s2)

# ==================================  OrientedSector  ======================================

"""
    OrientedSector(s::Sector, isdual::Bool)

A sector together with an arrow. This is where a bare sector acquires a duality, and what
`dual` of a sector returns.
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
TKS.Sector(s::OrientedSector) = TKS.Sector(sector(s))
tensorkit_sectortype(::Type{OrientedSector{S}}) where {S} = tensorkit_sectortype(S)

TensorAlgebra.isdual(s::OrientedSector) = s.isdual
TensorAlgebra.isdual(::Sector) = false

# `dual` flips the arrow and leaves the label alone, so a bare sector has to gain an arrow to
# carry the result. `flip` is the composite of the arrow flip with charge conjugation.
TensorAlgebra.dual(s::Sector) = OrientedSector(s, true)
TensorAlgebra.dual(s::OrientedSector) = OrientedSector(sector(s), !isdual(s))
flip(s::OrientedSector) = OrientedSector(dual_sector(sector(s)), !isdual(s))
flip(s::Sector) = OrientedSector(dual_sector(s), true)
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
function Sector(s::OrientedSector)
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

The sector of the trivial group, and so the sectortype of a space carrying no symmetry. It is
the unit object of the category of ordinary vector spaces, and the unit of `sectorproduct`.

It is not another symmetry's trivial sector: `U1(0)` is the zero-charge irrep *of* U(1), and a
`U1`-graded space with a single zero-charge block is not an ungraded space. Ask `istrivial`
whether a sector is its own symmetry's trivial one.
"""
struct TrivialSector <: Sector end
tensorkit_sectortype(::Type{TrivialSector}) = TKS.Trivial
sector_labels(::TrivialSector) = ()
Sector(::TKS.Trivial) = TrivialSector()

# TensorKitSectors has no ordering or equality between `Trivial` and a real irrep, so the
# generic methods above cannot answer these. A `TrivialSector` equals only itself, and a product
# with no content: it denotes the absence of a symmetry rather than any particular symmetry's
# trivial sector. Equating it with all of those made `==` intransitive, since `U1(0)` and
# `SU2(0)` would each equal it while differing from each other. `istrivial` asks that question.
# Ordering it below every other sector keeps `isless` a total order in which no two distinct
# sectors come out order-equivalent. `convert` and `promote_rule` are deliberately absent too:
# `convert` must preserve value, which `TrivialSector` into `trivial(S)` no longer does, and
# `trivial(S)` already spells that conversion where it is wanted.
Base.:(==)(::TrivialSector, ::TrivialSector) = true
Base.:(==)(::TrivialSector, ::Sector) = false
Base.:(==)(::Sector, ::TrivialSector) = false
Base.isless(::TrivialSector, ::TrivialSector) = false
Base.isless(::TrivialSector, ::Sector) = true
Base.isless(::Sector, ::TrivialSector) = false
fusion_rule(s::TrivialSector, ::TrivialSector) = s
# The unit fuses as the other operand's own trivial sector, so the ordinary fusion answers this
# and gives the outcome in whichever form it gives outcomes for that sector.
fusion_rule(::TrivialSector, s::Sector) = fusion_rule(trivial(s), s)
fusion_rule(s::Sector, ::TrivialSector) = fusion_rule(s, trivial(s))

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
"""
    const Z2 = Z{2}

The irreducible representations of the cyclic group of order two, labelled `0` and `1`.

See also [`Z`](@ref) and [`fZ2`](@ref), which is the fermionic counterpart.
"""
const Z2 = Z{2}
tensorkit_sectortype(::Type{Z{N}}) where {N} = TKS.ZNIrrep{N}
Sector(c::TKS.ZNIrrep{N}) where {N} = Z{N}(c.n)
sector_labels(s::Z) = (Int(s.n),)
# The order of the cyclic group. It is a property of the type rather than a label, so it is not
# something `sector_labels` can answer, and the type form is the one worth having.
modulus(s::Z) = modulus(typeof(s))
modulus(::Type{Z{N}}) where {N} = N

"""
    U1(charge::Real)

An irreducible representation of `U(1)`, labelled by its charge.
"""
struct U1 <: Sector
    charge::HalfInt
    U1(charge::Real) = new(TKS.U1Irrep(charge).charge)
end
tensorkit_sectortype(::Type{U1}) = TKS.U1Irrep
Sector(c::TKS.U1Irrep) = U1(c.charge)
sector_labels(s::U1) = (s.charge,)

"""
    SU2(j::Real)

An irreducible representation of `SU(2)`, labelled by its spin.
"""
struct SU2 <: Sector
    j::HalfInt
    # Constructed through TensorKitSectors so its non-negative half-integer check applies.
    SU2(j::Real) = new(TKS.SU2Irrep(j).j)
end
tensorkit_sectortype(::Type{SU2}) = TKS.SU2Irrep
Sector(c::TKS.SU2Irrep) = SU2(c.j)
sector_labels(s::SU2) = (s.j,)

"""
    CU1(j::Real, s::Integer = ifelse(j > zero(j), 2, 0))

An irreducible representation of `U(1) ⋊ C`, also written `O(2)`: the `U(1)` charge `j` together
with the representation `s` of charge conjugation. For `j > 0` the only value is `s = 2`, the
two-dimensional representation. For `j == 0` there are two, `s = 0` and `s = 1`, the trivial and
non-trivial representations of the conjugation.
"""
struct CU1 <: Sector
    j::HalfInt
    s::Int
    # Constructed through TensorKitSectors so its check on the allowed `(j, s)` pairs applies.
    function CU1(j::Real, s::Integer = ifelse(j > zero(j), 2, 0))
        c = TKS.CU1Irrep(j, s)
        return new(c.j, c.s)
    end
end
tensorkit_sectortype(::Type{CU1}) = TKS.CU1Irrep
Sector(c::TKS.CU1Irrep) = CU1(c.j, c.s)
sector_labels(s::CU1) = (s.j, s.s)

"""
    SU{N}(a::Vararg{Int, M})

An irreducible representation of `SU(N)`, labelled either by its `N - 1` Dynkin labels or by
its `N`-component highest weight: `SU{3}(1, 1)` and `SU{3}(2, 1, 0)` are both the adjoint. The
rank is always spelled out, since the number of labels alone does not fix it.

The Dynkin labels are the canonical form, being the same for every weight denoting the same
representation, and are what [`sector_labels`](@ref) gives back and what is stored, as
`SUNIrrep` also stores. Two `SU`s are therefore equal exactly when they are the same
representation, which `SUNIrrep` did not guarantee while it stored a weight and compared it
verbatim.

[`SU2`](@ref) is a separate type labelling the same symmetry by its spin, and unlike `SU{2}` it
needs no extension. An `SU` is constructed and compared without `SUNRepresentations`, but its
dimension, ordering and fusion all need that package, and are defined in the extension.
"""
struct SU{N} <: Sector
    # The Dynkin labels, with a trailing zero. There are `N - 1` of them, a length no single type
    # parameter can state, and the zero costs the slot a weight's own trailing zero would have
    # taken anyway. Storing what `SUNIrrep` stores means converting either way copies rather
    # than computes, and makes two `SU`s equal exactly when they are the same representation,
    # since Dynkin labels do not depend on which weight was given.
    a::NTuple{N, Int}
    # Unconstrained in the number of labels so that giving the wrong number of them is an
    # argument error rather than a missing method.
    function SU{N}(a::Vararg{Int}) where {N}
        N >= 2 || throw(ArgumentError("`SU{N}` needs an N of at least 2, got $(N)"))
        if length(a) == N - 1
            all(>=(0), a) ||
                throw(ArgumentError("a Dynkin label must be non-negative, got $(a)"))
            return new{N}((a..., 0))
        end
        length(a) == N || throw(
            ArgumentError(
                "`SU{$(N)}` takes $(N - 1) Dynkin labels or $(N) weight components, \
                got $(length(a))"
            )
        )
        issorted(a; rev = true) ||
            throw(ArgumentError("a highest weight must be non-increasing, got $(a)"))
        return new{N}((ntuple(i -> a[i] - a[i + 1], Val(N - 1))..., 0))
    end
end
# The rank does not follow from the number of labels, so there is nothing to infer it from.
function SU(::Vararg{Int})
    throw(ArgumentError("`SU` needs its rank spelled out, as in `SU{3}(1, 1)`"))
end
sector_labels(s::SU) = Base.front(s.a)
trivial(::Type{SU{N}}) where {N} = SU{N}(ntuple(_ -> 0, Val(N - 1))...)
istrivial(s::SU) = all(iszero, sector_labels(s))
# Defined here rather than left to the generic methods, which convert to `SUNIrrep` and so would
# need the extension. Dynkin labels are canonical, so comparing them is comparing the
# representation, and a differing `N` gives tuples of differing length and so compares false.
Base.:(==)(s1::SU, s2::SU) = s1.a == s2.a
Base.hash(s::SU, h::UInt) = hash(s.a, hash(:SU, h))
sectortype_repr(::Type{SU{N}}) where {N} = "SU{$(N)}"

"""
    fZ2(isodd::Bool)

Fermion parity, the fermionic analog of `Z2`.
"""
struct fZ2 <: Sector
    isodd::Bool
end
tensorkit_sectortype(::Type{fZ2}) = TKS.FermionParity
Sector(c::TKS.FermionParity) = fZ2(c.isodd)
sector_labels(s::fZ2) = (s.isodd,)
# A parity reads as a charge rather than as a flag, the way `Z{2}` and the other sectors display,
# so the `Bool` prints as `0` or `1`.
Base.show(io::IO, s::fZ2) = print(io, "fZ2(", Int(s.isodd), ')')

sector_labels(s::TensorKitSector) = sector_labels(TKS.Sector(s))
# Upstream has no accessor of its own, and spells a sector out by looping its fields.
sector_labels(c::TKS.Sector) = map(f -> getfield(c, f), fieldnames(typeof(c)))
# A product sector's one field is already the component tuple, so the generic method above would
# wrap it in a second tuple. Its labels are its components.
sector_labels(c::TKS.ProductSector) = map(Sector, c.sectors)

# The GradedArrays name of a sector type, for the constructor-form display below. Upstream's
# `type_repr` is the same idea for its own types, and a symmetry reached through
# `TensorKitSector` defers to it rather than to a name we re-derive.
sectortype_repr(::Type{S}) where {S <: Sector} = string(nameof(S))
sectortype_repr(::Type{Z{N}}) where {N} = "Z{$N}"
sectortype_repr(::Type{TensorKitSector{I}}) where {I} = TKS.type_repr(I)

# Constructor-form display of a sector value: `name(sector_labels...)`. Upstream spells an
# `AbstractIrrep` out the same way, as the constructor call that rebuilds it.
function Base.show(io::IO, s::Sector)
    print(io, sectortype_repr(typeof(s)), '(')
    for (k, v) in enumerate(sector_labels(s))
        k > 1 && print(io, ", ")
        show(io, v)
    end
    return print(io, ')')
end

# A `TKS.ProductSector` has no name of its own, so it shows as its components rather than
# leaking `ProductSector` into the display. Components go through `Sector` so each prints under
# its GradedArrays name. Only an explicitly wrapped product reaches these, since `Sector`
# unwraps a `ProductSector` into a `TupleSectorProduct`.
function Base.show(io::IO, s::TensorKitSector{<:TKS.ProductSector})
    print(io, '(')
    join(io, (sprint(show, Sector(c); context = io) for c in TKS.Sector(s).sectors), " × ")
    return print(io, ')')
end
function Base.show(
        io::IO, s::TensorKitSector{TKS.ProductSector{Tuple{TKS.U1Irrep, TKS.FermionParity}}}
    )
    c = TKS.Sector(s)
    q = c.sectors[1].charge
    # Recover the alias only for a value `FermionNumber` actually produces; any other
    # `(U1, FermionParity)` product shows as its components.
    isinteger(q) && TKS.FermionNumber(Int(q)) == c ||
        return @invoke show(io::IO, s::TensorKitSector{<:TKS.ProductSector})
    return print(io, "FermionNumber(", Int(q), ")")
end
