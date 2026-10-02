# A sector with an arrow, where a bare `Sector`'s arrow is unset. Both it and `OrientedSector`
# answer `sector` and `isdual`, and that pair is the identity: comparison, hashing and ordering
# are defined once here rather than per type, and anything taking a sector whose arrow is
# optional dispatches on this.
abstract type AbstractOrientedSector <: AbstractUnitRange{Int} end

"""
    Sector

An irreducible label of a symmetry, and the range of the degrees of freedom that label spans,
so `length` is the sector's dimension.

    Sector(s::Sector)
    Sector(c::TensorKitSectors.Sector)
    Sector(s1, s2, srest...)
    Sector(t::Tuple)
    Sector(nt::NamedTuple)
    Sector(; kws...)

Two or more sectors give the product over them, positional or named. Everything that takes a
sector from a caller, `gradedrange` and the array constructors included, routes through here.
"""
abstract type Sector <: AbstractOrientedSector end

# The type-level counterpart of `TensorKitSectors.Sector(s)`, which the fusion and braiding
# traits dispatch through. Every concrete `Sector` defines it.
tensorkit_sectortype(s::Sector) = tensorkit_sectortype(typeof(s))

# The `Sector` a TensorKitSectors sector converts to, the inverse of `tensorkit_sectortype`.
# Read off the `Sector` constructor rather than restated, so a sector type that defines the
# conversion needs no second declaration here, and one that wants to state it anyway can
# define this method.
gradedarrays_sectortype(c::TKS.Sector) = gradedarrays_sectortype(typeof(c))
function gradedarrays_sectortype(::Type{I}) where {I <: TKS.Sector}
    S = Base.promote_op(Sector, I)
    isconcretetype(S) || throw(
        ArgumentError("no single `Sector` type for $(I), `Sector` of one infers as $(S)")
    )
    return S
end

# A sector converts by handing its labels to its counterpart's own constructor, which takes them
# in the same order and is the spelling the sector's display prints. A type therefore declares
# the counterpart and its labels, and this direction of the conversion follows.
TKS.Sector(s::Sector) = tensorkit_sectortype(s)(sector_labels(s)...)

"""
    TensorKitSector(c::TensorKitSectors.Sector)

Wrap a TensorKitSectors sector and reinterpret it as a [`Sector`](@ref). This is what supports
the symmetries TensorKitSectors defines that have no GradedArrays sector of their own, such as
the anyons.
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
# Terminates the recursion through `sectortype(x)`, which would otherwise be a fixed point once
# it reached `DataType`. The argument is wrapped in a tuple because that is what `MethodError`
# reports as the call's arguments, and `showerror` throws on anything else.
sectortype(T::Type) = throw(MethodError(sectortype, (T,)))

# ===================================  Base interface  =====================================

Base.length(s::Sector) = TKS.dim(TKS.Sector(s))
Base.length(x::AbstractOrientedSector) = length(sector(x))
Base.OneTo(x::AbstractOrientedSector) = Base.OneTo(length(x))
Base.first(x::AbstractOrientedSector) = first(Base.OneTo(x))
Base.last(x::AbstractOrientedSector) = last(Base.OneTo(x))
Base.axes(x::AbstractOrientedSector) = (x,)

# Ordering and equality run through TensorKitSectors so that two sectors GradedArrays spells
# differently, such as a named type and the same symmetry reached as a `TensorKitSector`,
# still compare as the sector they both denote.
function Base.isless(s1::Sector, s2::Sector)
    return isless(TKS.Sector(s1), TKS.Sector(s2))
end
Base.:(==)(s1::Sector, s2::Sector) = TKS.Sector(s1) == TKS.Sector(s2)
Base.hash(s::Sector, h::UInt) = hash(TKS.Sector(s), h)

# Identity is `(sector, isdual)`, so equality and hashing compare that pair, and the two bare
# methods above are their base case. An unset arrow contributes nothing, which is what makes a
# non-dual `OrientedSector` interchangeable with the bare sector it wraps, as `flip(dual(s)) == s`
# requires. Comparison stays between two of our own and never crosses to the TensorKitSectors
# sector one converts to, because `hash` takes a single operand and so has to commit to one
# library's notion of identity. `isequal` is written out at all because these are
# `AbstractUnitRange`s and would otherwise inherit Base's elementwise `AbstractArray` method
# instead of the scalar fallback.
function Base.:(==)(a::AbstractOrientedSector, b::AbstractOrientedSector)
    return sector(a) == sector(b) && isdual(a) == isdual(b)
end
Base.isequal(a::AbstractOrientedSector, b::AbstractOrientedSector) = a == b
# Ordering only orders the sectors, and needs a shared arrow to do it: sectors get sorted to put a
# graded axis in canonical form, and an axis has one arrow for all of its blocks, so there is no
# meaningful order between a sector and a dual one.
function Base.isless(a::AbstractOrientedSector, b::AbstractOrientedSector)
    isdual(a) == isdual(b) || throw(
        ArgumentError("cannot order sectors with different arrows: $(a) and $(b)")
    )
    return isless(sector(a), sector(b))
end
function Base.hash(x::AbstractOrientedSector, h::UInt)
    return isdual(x) ? hash(:dual, hash(sector(x), h)) : hash(sector(x), h)
end

# =================================  Sectors interface  ====================================

# The sector's labels, in the order its constructor takes them, which is the order its fields
# are in. These go straight to the TensorKitSectors constructor, so they are the stored values
# rather than widened ones.
sector_labels(s::Sector) = map(f -> getfield(s, f), fieldnames(typeof(s)))

trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
    return gradedrange([trivial(sectortype(axis_type)) => 1])  # always returns nondual
end
trivial(type::Type) = error("`trivial` not defined for type $(type).")
trivial(S::Type{<:Sector}) = Sector(one(tensorkit_sectortype(S)))
trivial(::Type{I}) where {I <: TKS.Sector} = one(I)

istrivial(s::Sector) = isone(TKS.Sector(s))
istrivial(x) = (x == trivial(x))

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

# The conjugate sector, which is what TensorKitSectors calls `dual`. Here `dual` is the arrow
# operation instead, since a `Sector` is itself the range of its degrees of freedom and
# so `dual` of one is the dual space, returning an `OrientedSector`.
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

# A sector together with an arrow. This is where a bare sector acquires a duality, and what
# `dual` of a sector returns.
struct OrientedSector{S <: Sector} <: AbstractOrientedSector
    sector::S
    isdual::Bool
end
OrientedSector(x::AbstractOrientedSector) = OrientedSector(sector(x), isdual(x))

# The two accessors every `AbstractOrientedSector` answers.
sector(s::OrientedSector) = s.sector
sector(s::Sector) = s
TensorAlgebra.isdual(s::OrientedSector) = s.isdual
TensorAlgebra.isdual(::Sector) = false

sectortype(::Type{OrientedSector{S}}) where {S} = S
tensorkit_sectortype(::Type{OrientedSector{S}}) where {S} = tensorkit_sectortype(S)

# An unset arrow is what makes a non-dual `OrientedSector` equal to the sector it wraps, so it also
# makes it convert to that sector, which lets it stand in for the sector wherever a bare one is
# required. A dual sector is equal to no bare sector, so it does not convert.
function Base.convert(::Type{S}, x::OrientedSector) where {S <: Sector}
    isdual(x) && throw(
        ArgumentError("a dual sector does not convert to a bare sector: $(x)")
    )
    return convert(S, sector(x))
end

# Everything below reads the pair and so serves both types. `dual` flips the arrow and leaves the
# label alone, so a bare sector gains an arrow to carry the result. `flip` composes that with
# charge conjugation. An arrow does not change whether a sector is trivial, its fermion parity,
# or its twist, so those forward to the sector, whose own methods are the base case.
TensorAlgebra.dual(x::AbstractOrientedSector) = OrientedSector(sector(x), !isdual(x))
flip(x::AbstractOrientedSector) = OrientedSector(dual_sector(sector(x)), !isdual(x))
flip_dual(x::AbstractOrientedSector) = isdual(x) ? flip(x) : x
nondual(s::OrientedSector) = OrientedSector(sector(s), false)
nondual(s::Sector) = s
Base.conj(x::AbstractOrientedSector) = dual(x)
TKS.Sector(x::AbstractOrientedSector) = TKS.Sector(sector(x))
istrivial(x::AbstractOrientedSector) = istrivial(sector(x))
fermionparity(x::AbstractOrientedSector) = fermionparity(sector(x))
twist(x::AbstractOrientedSector) = twist(sector(x))
to_gradedrange(x::AbstractOrientedSector) = GradedOneTo([sector(x)], [1], isdual(x))

# `splitarrows` takes a tuple of them apart into the sectors and the arrows, which is how a type
# stores them.
splitarrows(ss::Tuple{Vararg{AbstractOrientedSector}}) = (map(sector, ss), map(isdual, ss))

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
# Fusing folds each arrow into its sector, so the result is bare whatever the arguments were,
# which is also what keeps a `reduce` closed: after the first step one side is already bare while
# the rest are still oriented.
function fusion_rule(a::AbstractOrientedSector, b::AbstractOrientedSector)
    return fusion_rule(sector(flip_dual(a)), sector(flip_dual(b)))
end
tensor_product(x::AbstractOrientedSector) = sector(flip_dual(x))
function tensor_product(a::AbstractOrientedSector, b::AbstractOrientedSector)
    return fusion_rule(a, b)
end

# =====================================  Sectors  ==========================================

"""
    Trivial()

The sector of the trivial group, and so the sectortype of a space carrying no symmetry. It is
the unit of `sectorproduct`.

It is not another symmetry's trivial sector: `U1(0)` is the zero-charge irrep *of* U(1), and a
`U1`-graded space with a single zero-charge block is not an ungraded space.
"""
struct Trivial <: Sector end
tensorkit_sectortype(::Type{Trivial}) = TKS.Trivial
Sector(::TKS.Trivial) = Trivial()

# TensorKitSectors has no ordering or equality between its own `Trivial` and a real irrep, so the
# generic methods above cannot answer these. `Trivial` denotes the absence of a symmetry rather
# than any symmetry's trivial sector, so it equals only itself and orders below everything: if it
# equalled every `trivial(S)`, then `U1(0)` and `SU2(0)` would both equal it while differing from
# each other. `istrivial` asks that question, and `trivial(S)` spells the conversion that
# `convert` and `promote_rule` deliberately do not, since `convert` has to preserve value.
Base.:(==)(::Trivial, ::Trivial) = true
Base.:(==)(::Trivial, ::Sector) = false
Base.:(==)(::Sector, ::Trivial) = false
Base.isless(::Trivial, ::Trivial) = false
Base.isless(::Trivial, ::Sector) = true
Base.isless(::Sector, ::Trivial) = false
fusion_rule(s::Trivial, ::Trivial) = s
# The unit fuses as the other operand's own trivial sector, so the ordinary fusion answers this
# and gives the outcome in whichever form it gives outcomes for that sector.
fusion_rule(::Trivial, s::Sector) = fusion_rule(trivial(s), s)
fusion_rule(s::Sector, ::Trivial) = fusion_rule(s, trivial(s))

"""
    Z{N}(n::Integer)

An irreducible representation of the cyclic group of order `N`.
"""
struct Z{N} <: Sector
    n::Int8
    # The label is reduced modulo `N` by TensorKitSectors rather than restated here, but the
    # bound on `N` is ours to report: upstream's message names a type of its own that has no
    # counterpart here, so it would send a reader looking for something that does not exist.
    function Z{N}(n::Integer) where {N}
        1 <= N <= 128 || throw(
            ArgumentError("`Z{N}` needs an `N` between 1 and 128, got $(N)")
        )
        return new{N}(TKS.ZNIrrep{N}(n).n)
    end
end
"""
    const Z2 = Z{2}

The irreducible representations of the cyclic group of order two, labelled `0` and `1`.

See also [`Z`](@ref) and [`fZ2`](@ref), which is the fermionic counterpart.
"""
const Z2 = Z{2}
tensorkit_sectortype(::Type{Z{N}}) where {N} = TKS.ZNIrrep{N}
Sector(c::TKS.ZNIrrep{N}) where {N} = Z{N}(c.n)
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

"""
    CU1(j::Real, s::Integer = ifelse(j > zero(j), 2, 0))

An irreducible representation of `U(1) ⋊ C`, also called `O(2)`: the `U(1)` charge `j` together
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

"""
    SU{N}(a::Vararg{Int})

An irreducible representation of `SU(N)`, labelled either by its `N - 1` Dynkin labels or by its
`N`-component highest weight: `SU{3}(1, 1)` and `SU{3}(2, 1, 0)` are both the adjoint.

Dimensions and fusion need `SUNRepresentations`.

See also [`SU2`](@ref).
"""
struct SU{N, M} <: Sector
    # The `N - 1` Dynkin labels, in the byte storage `SUNIrrep{N, M}` uses, so the two types are
    # the same size and a label reaches upstream's constructor unconverted. The byte range is the
    # bound this reports. `M` is `N - 1`, a parameter only because a field type cannot be computed
    # from `N`.
    a::NTuple{M, UInt8}
    function SU{N, M}(a::NTuple{M, Integer}) where {N, M}
        N >= 2 || throw(ArgumentError("`SU{N}` needs an N of at least 2, got $(N)"))
        M == N - 1 || throw(
            ArgumentError("`SU{$(N)}` has $(N - 1) Dynkin labels, got $(M)")
        )
        all(x -> 0 <= x <= typemax(UInt8), a) || throw(
            ArgumentError(
                "a Dynkin label must be between 0 and $(Int(typemax(UInt8))), got $(a)"
            )
        )
        return new{N, M}(map(UInt8, a))
    end
end
# Either the `N - 1` Dynkin labels or the `N` components of a highest weight. Unconstrained in the
# number of labels so that giving the wrong number of them is an argument error rather than a
# missing method.
function SU{N}(a::Vararg{Int}) where {N}
    N >= 2 || throw(ArgumentError("`SU{N}` needs an N of at least 2, got $(N)"))
    length(a) == N - 1 && return SU{N, N - 1}(a)
    length(a) == N || throw(
        ArgumentError(
            "`SU{$(N)}` takes $(N - 1) Dynkin labels or $(N) weight components, \
            got $(length(a))"
        )
    )
    issorted(a; rev = true) ||
        throw(ArgumentError("a highest weight must be non-increasing, got $(a)"))
    return SU{N, N - 1}(ntuple(i -> a[i] - a[i + 1], Val(N - 1)))
end
SU{N, M}(a::Vararg{Int}) where {N, M} = SU{N, M}(a)
# The rank does not follow from the number of labels, so there is nothing to infer it from.
function SU(::Vararg{Int})
    throw(ArgumentError("`SU` needs its rank spelled out, as in `SU{3}(1, 1)`"))
end
# The one field is already the label tuple, so the generic read would wrap it in a second tuple.
sector_labels(s::SU) = s.a
# `SUNIrrep` takes its labels as a tuple rather than as varargs, so the generic conversion's splat
# does not resolve. Handing the stored tuple over whole is also the cheaper crossing, since both
# sides store an `NTuple{M, UInt8}`.
TKS.Sector(s::SU) = tensorkit_sectortype(s)(sector_labels(s))
trivial(::Type{<:SU{N}}) where {N} = SU{N, N - 1}(ntuple(_ -> 0, Val(N - 1)))
istrivial(s::SU) = all(iszero, sector_labels(s))
# Defined here rather than left to the generic methods, which convert to `SUNIrrep` and so would
# need the extension. Dynkin labels are canonical, so comparing them is comparing the
# representation, and a differing `N` gives tuples of differing length and so compares false.
Base.:(==)(s1::SU, s2::SU) = s1.a == s2.a
Base.hash(s::SU, h::UInt) = hash(s.a, hash(:SU, h))
sectortype_repr(::Type{<:SU{N}}) where {N} = "SU{$(N)}"

"""
    fZ2(isodd::Bool)

Fermion parity, the fermionic analog of `Z2`.
"""
struct fZ2 <: Sector
    isodd::Bool
end
tensorkit_sectortype(::Type{fZ2}) = TKS.FermionParity
Sector(c::TKS.FermionParity) = fZ2(c.isodd)

sector_labels(s::TensorKitSector) = sector_labels(TKS.Sector(s))
# Upstream has no accessor of its own, and spells a sector out by looping its fields.
sector_labels(c::TKS.Sector) = map(f -> getfield(c, f), fieldnames(typeof(c)))
# A product sector's one field is already the component tuple, so the generic method above would
# wrap it in a second tuple. Its labels are its components.
sector_labels(c::TKS.ProductSector) = map(Sector, c.sectors)

# The GradedArrays name of a sector type, for the constructor-form display below. Upstream's
# `type_repr` is the same idea for its own types.
sectortype_repr(::Type{S}) where {S <: Sector} = string(nameof(S))
sectortype_repr(::Type{Z{N}}) where {N} = "Z{$N}"
# `Z2` is the one order with an exported alias of its own, so it shows under that name.
sectortype_repr(::Type{Z2}) = "Z2"

# A label as a reader should see it, since a stored `UInt8` would otherwise `show` as `0x01` and
# a `Bool` parity as `true`. A `HalfInt` spin is not an `Integer`, so it keeps its `1/2` form.
pretty_sector_label(x) = x
pretty_sector_label(x::Integer) = Int(x)

# Constructor-form display of a sector value: `name(sector_labels...)`. Upstream spells an
# `AbstractIrrep` out the same way, as the constructor call that rebuilds it.
function Base.show(io::IO, s::Sector)
    print(io, sectortype_repr(typeof(s)), '(')
    for (k, v) in enumerate(sector_labels(s))
        k > 1 && print(io, ", ")
        show(io, pretty_sector_label(v))
    end
    return print(io, ')')
end

# The wrapping call around upstream's own spelling of the sector it wraps. The wrapper belongs in
# the display because the printed form has to rebuild this value, and the native sector for the
# same symmetry is a different one: `==` on a product and a non-product is `false` by
# construction, so the native spelling would read back as something this does not equal. The
# inside is upstream's spelling rather than one re-derived from the fields, which for
# `FibonacciAnyon` would print its `isunit` flag as a number instead of the label that rebuilds
# it. One case the round trip does not survive: upstream spells a product of irreps
# `Irrep[U₁ × U₁](1, 2)`, whose `×` is theirs rather than ours, so reading that back needs theirs
# in scope. Only an explicitly wrapped sector reaches this, since `Sector` unwraps anything with
# a native counterpart, a `ProductSector` into a `TupleSectorProduct` included.
function Base.show(io::IO, s::TensorKitSector)
    print(io, "TensorKitSector(")
    show(io, TKS.Sector(s))
    return print(io, ")")
end
