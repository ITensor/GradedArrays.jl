# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

# =====================================  Definition  =======================================

"""
    SectorProduct

The Cartesian product of two or more sectors, itself a [`Sector`](@ref). Its arguments are
bare sectors, so like any other sector it carries no arrow of its own.

Abstract, with `TupleSectorProduct` and `NamedSectorProduct` as its two concrete forms. Build
one by calling `Sector`, which picks the form matching what it is given, or `sectorproduct` to
multiply sectors you already hold.
"""
abstract type SectorProduct <: Sector end

"""
    TupleSectorProduct(arguments::Tuple)

A `SectorProduct` whose arguments are positional. A position identifies a factor only relative
to this product, so the arity is part of the sector's identity and there is no sense in which a
factor can be left out.

Takes its arguments as given. Call `Sector` to build one from a specification that still needs
normalizing.
"""
struct TupleSectorProduct{Arguments <: Tuple} <: SectorProduct
    arguments::Arguments
end

"""
    NamedSectorProduct(arguments::NamedTuple)

A `SectorProduct` whose arguments are named. A name identifies a symmetry across products, so a
symmetry this one does not name is that symmetry's trivial sector, which is what lets products
over different sets of symmetries be compared and fused.

Sorts its arguments by name, an invariant the type relies on, but otherwise takes them as
given. Call `Sector` to build one from a specification that still needs normalizing.
"""
struct NamedSectorProduct{Arguments <: NamedTuple} <: SectorProduct
    arguments::Arguments
    function NamedSectorProduct(nt::NamedTuple)
        sorted = sort_keys(nt)
        return new{typeof(sorted)}(sorted)
    end
end

# `Sector` is the single entry point for turning a specification into a sector, so these are
# what define which spellings a product can be written as. Each normalizes its arguments
# through `Sector` in turn, which is what lets TensorKitSectors sectors appear inside them.
# Two or more factors, left untyped: the arity floor keeps this from competing with the
# arity-1 methods, so the element types do not have to be restated here, and a bad element
# fails as `Sector(element)` the same way it does inside the container methods.
Sector(s1, s2, srest...) = Sector((s1, s2, srest...))
Sector(t::Tuple) = TupleSectorProduct(map(Sector, t))
Sector(nt::NamedTuple) = NamedSectorProduct(map(Sector, nt))
# An explicit empty container is still a specification of that shape, so `Sector(())` and
# `Sector((;))` keep their forms. Keeping the container methods uniform matters because the
# argument is often built by the caller, and an `args` that happens to come out empty should not
# change which form comes back. Passing no factors at all specifies no symmetry instead, which
# is `TrivialSector`. That cannot be its own method, since `Sector(; kws...)` already claims the
# no-argument signature, so it is a branch on the keywords instead, resolved at compile time
# because their `NamedTuple` type is concrete.
function Sector(; kws...)
    nt = values(kws)
    isempty(nt) && return TrivialSector()
    return Sector(nt)
end

# Fusion and the n-symbol work argument by argument, so a mixed call has to read its bare operand
# as the one-factor positional product for the duration. `Sector` deliberately does not build
# one-factor products, so the wrapping is spelled out here instead. Positional always: a bare
# sector names no symmetry, so there is nothing to match against a named product's keys, and the
# promotion rejects the pairing.
to_sectorproduct(s::SectorProduct) = s
to_sectorproduct(s::Sector) = TupleSectorProduct((s,))

arguments(s::SectorProduct) = getfield(s, :arguments)
arguments_type(::Type{<:TupleSectorProduct{T}}) where {T} = T
arguments_type(::Type{<:NamedSectorProduct{T}}) where {T} = T

label(s::SectorProduct) = map(label, arguments(s))

# The TensorKitSectors counterparts, used for ordering and by anything that reaches for a
# sector's TensorKitSectors form. Upstream draws the same positional/named distinction, so each
# kind maps onto its own counterpart and both directions keep the arguments and their names.
# Fusion still does not go through these, since neither upstream form can express the mismatched
# argument sets `promote_sector` brings together.
TKS.Sector(s::TupleSectorProduct) = TKS.ProductSector(map(TKS.Sector, arguments(s)))
TKS.Sector(s::NamedSectorProduct) = TKS.NamedSector(map(TKS.Sector, arguments(s)))

# Coming back the other way, an upstream product is the upstream form of a product and returns
# one rather than being wrapped. Without this a block label read back out of TensorKit does not
# compare equal to the sector the axis was built from.
Sector(c::TKS.ProductSector) = Sector(Tuple(c))
Sector(c::TKS.NamedSector) = Sector(NamedTuple(c))

function tensorkit_sectortype(::Type{P}) where {P <: TupleSectorProduct}
    T = arguments_type(P)
    return TKS.ProductSector{Tuple{map(tensorkit_sectortype, fieldtypes(T))...}}
end
function tensorkit_sectortype(::Type{P}) where {P <: NamedSectorProduct}
    T = arguments_type(P)
    return TKS.NamedSector{
        NamedTuple{fieldnames(T), Tuple{map(tensorkit_sectortype, fieldtypes(T))...}},
    }
end

# =================================  Sectors interface  ====================================

function TKS.FusionStyle(::Type{P}) where {P <: SectorProduct}
    return mapreduce(
        TKS.FusionStyle, &, fieldtypes(arguments_type(P)); init = TKS.UniqueFusion()
    )
end
function TKS.BraidingStyle(::Type{P}) where {P <: SectorProduct}
    return mapreduce(
        TKS.BraidingStyle,
        &,
        fieldtypes(arguments_type(P));
        init = TKS.Bosonic()
    )
end

Base.length(s::SectorProduct) = prod(length, arguments(s); init = 1)

# Fermion parity and twist of a product are the xor and the product of its arguments'. Taking
# them argument by argument also covers the empty product, which has no `ProductSector` form to
# delegate to.
fermionparity(s::SectorProduct) = mapreduce(fermionparity, ⊻, arguments(s); init = false)
twist(s::SectorProduct) = prod(twist, arguments(s); init = 1)

# use map instead of broadcast to support both Tuple and NamedTuple
function charge_conjugate(s::SectorProduct)
    return Sector(map(charge_conjugate, arguments(s)))
end

function trivial(::Type{P}) where {P <: TupleSectorProduct}
    return TupleSectorProduct(map(trivial, fieldtypes(arguments_type(P))))
end
function trivial(::Type{P}) where {P <: NamedSectorProduct}
    NT = arguments_type(P)
    return NamedSectorProduct(NT(map(trivial, fieldtypes(NT))))
end
istrivial(s::SectorProduct) = all(istrivial, arguments(s))

# ===============================  Fusion rule interface  ==================================

# The fusion of two products is the product of its arguments' fusions, so it is built argument
# by argument. A product with no arguments constrains no symmetry, so rather than promoting it, it
# takes the other operand's shape by fusing that operand with its own trivial sector.
function fusion_rule(s1::SectorProduct, s2::SectorProduct)
    isempty(arguments(s1)) && isempty(arguments(s2)) && return s1
    isempty(arguments(s1)) && return fusion_rule(trivial(s2), s2)
    isempty(arguments(s2)) && return fusion_rule(s1, trivial(s1))
    s1′, s2′ = promote_sector(s1, s2)
    fstyle = TKS.FusionStyle(typeof(s1′)) & TKS.FusionStyle(typeof(s2′))
    fstyle === TKS.UniqueFusion() &&
        return Sector(map(fusion_rule, arguments(s1′), arguments(s2′)))
    return gradedrange([s => TKS.Nsymbol(s1′, s2′, s) for s in fusion_products(s1′, s2′)])
end
fusion_rule(s1::SectorProduct, s2::Sector) = fusion_rule(s1, TupleSectorProduct((s2,)))
fusion_rule(s1::Sector, s2::SectorProduct) = fusion_rule(TupleSectorProduct((s1,)), s2)
# `TrivialSector` has its own methods against any `Sector`, which the two above would otherwise be
# ambiguous with. They fuse it the same way the branches above do, rather than reading it as a
# product: it denotes no symmetry, so it has no arguments to promote.
fusion_rule(s::SectorProduct, ::TrivialSector) = fusion_rule(s, trivial(s))
fusion_rule(::TrivialSector, s::SectorProduct) = fusion_rule(trivial(s), s)

# Every sector the fusion of `s1` and `s2` can produce, as the Cartesian product of its
# arguments' fusion outcomes. Both arguments must already be canonicalized.
function fusion_products(s1::SectorProduct, s2::SectorProduct)
    argument_sectors = map(arguments(s1), arguments(s2)) do a1, a2
        return sectors(to_gradedrange(fusion_rule(a1, a2)))
    end
    return vec(
        map(Iterators.product(values(argument_sectors)...)) do args
            return rebuild_arguments(s1, args)
        end
    )
end
rebuild_arguments(::TupleSectorProduct, args::Tuple) = TupleSectorProduct(args)
function rebuild_arguments(s::NamedSectorProduct, args::Tuple)
    return NamedSectorProduct(arguments_type(typeof(s))(args))
end

# multiple dispatch through explicit loop
for T1 in (:SectorProduct, :Sector),
        T2 in (:SectorProduct, :Sector),
        T3 in (:SectorProduct, :Sector)

    T1 === T2 === T3 && continue
    @eval function TKS.Nsymbol(s1::$T1, s2::$T2, s3::$T3)
        return TKS.Nsymbol(
            to_sectorproduct(s1), to_sectorproduct(s2), to_sectorproduct(s3)
        )
    end
end
function TKS.Nsymbol(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
    isempty(arguments(s1)) && isempty(arguments(s2)) && return istrivial(s3) ? 1 : 0
    isempty(arguments(s1)) && return TKS.Nsymbol(trivial(s2), s2, s3)
    isempty(arguments(s2)) && return TKS.Nsymbol(s1, trivial(s1), s3)
    isempty(arguments(s3)) && return TKS.Nsymbol(s1, s2, trivial(s1))

    s1′, s2′, s3′ = promote_sector(s1, s2, s3)
    return prod(
        splat(TKS.Nsymbol), zip(arguments(s1′), arguments(s2′), arguments(s3′));
        init = 1
    )
end

# ===================================  Base interface  =====================================

# Equality and hashing read one notion of content, so they cannot come apart. A positional
# product's content is all of its arguments, because a position identifies a factor only within
# that product and so the arity is part of the identity. A named product's content is its
# non-trivial arguments, because a name identifies a symmetry across products and a symmetry a
# product does not name is that symmetry's trivial sector. That asymmetry is exactly why named
# products compare across different sets of symmetries and positional ones cannot.
#
# `promote_sector` is deliberately not used here, for two reasons. It throws on operands it cannot
# bring to a common argument set, which is right for fusion but wrong for equality: comparing two
# sectors has to answer `false` rather than raise. And it takes two operands, while `hash` takes
# one and so has nothing to promote against, which would leave `hash` unable to follow the
# procedure it has to agree with.

function Base.:(==)(a::TupleSectorProduct, b::TupleSectorProduct)
    length(arguments(a)) == length(arguments(b)) || return false
    return all(splat(==), zip(arguments(a), arguments(b)))
end

function Base.:(==)(a::NamedSectorProduct, b::NamedSectorProduct)
    aa, bb = arguments(a), arguments(b)
    for k in keys(aa)
        v = aa[k]
        istrivial(v) && continue
        (haskey(bb, k) && v == bb[k]) || return false
    end
    for k in keys(bb)
        istrivial(bb[k]) && continue
        (haskey(aa, k) && !istrivial(aa[k])) || return false
    end
    return true
end

# Everything a product is not equal to. It is not its own argument, since `Sector` builds no
# one-factor product and there is nothing to round-trip. It is not a product of the other
# indexing, since the two describe different objects. And it is not `TrivialSector`, which is
# equal only to itself, the same way `U1(0)` is not it: asking whether a sector denotes no
# symmetry is `istrivial`'s job, not equality's.
Base.:(==)(::SectorProduct, ::Sector) = false
Base.:(==)(::Sector, ::SectorProduct) = false
Base.:(==)(::SectorProduct, ::SectorProduct) = false

# `(SectorProduct, TrivialSector)` is ambiguous between `==(::SectorProduct, ::Sector)` above and
# `==(::Sector, ::TrivialSector)`, which agree; these state that answer.
Base.:(==)(::SectorProduct, ::TrivialSector) = false
Base.:(==)(::TrivialSector, ::SectorProduct) = false

# `hash` has no second operand to promote against, so it hashes directly whatever content `==`
# compares: every argument of a positional product, and only the non-trivially-valued name and
# value pairs of a named one, in the sorted order they are stored in.
function Base.hash(s::TupleSectorProduct, h::UInt)
    return hash(
        :TupleSectorProduct,
        foldl((acc, a) -> hash(a, acc), arguments(s); init = h)
    )
end
function Base.hash(s::NamedSectorProduct, h::UInt)
    args = arguments(s)
    for k in keys(args)
        v = args[k]
        istrivial(v) || (h = hash(k, hash(v, h)))
    end
    return hash(:NamedSectorProduct, h)
end

# Within one axis every sector shares a type, and there the order must match how TensorKit orders
# the corresponding `GradedSpace`, so it delegates upstream. Products of different types never
# share an axis and only need a deterministic order, for which the arity and then the type
# itself serve.
function Base.isless(s1::P, s2::P) where {P <: SectorProduct}
    return isless(TKS.Sector(s1), TKS.Sector(s2))
end
function Base.isless(s1::SectorProduct, s2::SectorProduct)
    s1 == s2 && return false
    n1, n2 = length(arguments(s1)), length(arguments(s2))
    n1 == n2 || return n1 < n2
    return isless(string(typeof(s1)), string(typeof(s2)))
end
Base.isless(::SectorProduct, ::Sector) = false
Base.isless(::Sector, ::SectorProduct) = true
Base.isless(::TrivialSector, ::SectorProduct) = true
Base.isless(::SectorProduct, ::TrivialSector) = false

# Print the spelling that reconstructs the sector. Two or more positional factors get the infix
# form, which round-trips through `×`; everything else gets the explicit `Sector` form, since
# `×` cannot spell a named product, a one-factor product or an empty one.
function Base.show(io::IO, s::TupleSectorProduct)
    args = arguments(s)
    if length(args) < 2
        # The tuple is spelled out rather than passed as arguments, because `Sector` reads a lone
        # sector as itself and no arguments as `TrivialSector`, so neither of those round-trips.
        print(io, "Sector((")
        isempty(args) || (show(io, only(args)); print(io, ","))
        return print(io, "))")
    end
    print(io, "(")
    join(io, (sprint(show, a; context = io) for a in args), " × ")
    return print(io, ")")
end

function Base.show(io::IO, s::NamedSectorProduct)
    args = arguments(s)
    # Same reason as above: `Sector(;)` is a call with no keywords, which is `TrivialSector`.
    isempty(args) && return print(io, "Sector((;))")
    print(io, "Sector(;")
    for (i, (k, v)) in enumerate(pairs(args))
        print(io, i == 1 ? " " : ", ", k, " = ")
        show(io, v)
    end
    return print(io, ")")
end

# =================================  Cartesian Product  ====================================

"""
    sectorproduct(ss...)
    ×(ss...)

The Cartesian product of the symmetries of `ss`, as a `SectorProduct`. Each argument is
normalized with [`Sector`](@ref) first, so anything that specifies a sector can be multiplied.

A positional product absorbs factors positionally and a named one absorbs them by name.
Multiplying the two together is an error, because a product cannot be indexed both ways.
`TrivialSector` is the unit and drops out of any product, as it does in TensorKitSectors.
"""
function sectorproduct end
const × = sectorproduct

# The product over no sectors is the unit, matching `Sector()`.
# The type-level product, so a fused symmetry can be named as `const fU1 = U1 × fZ2`. Going
# through the value-level product on each symmetry's trivial sector means the unit rule and the
# flattening cannot drift from the value level, since they are the value level.
function sectorproduct(S1::Type{<:Sector}, Srest::Type{<:Sector}...)
    return typeof(sectorproduct(trivial(S1), map(trivial, Srest)...))
end

sectorproduct() = TrivialSector()

# Arity 1 is normalization and nothing more: a lone sector is already the product over itself,
# so no one-factor product is ever built.
sectorproduct(s::Sector) = s
sectorproduct(s) = Sector(s)

# Anything not already a sector is normalized here, then the methods below take over. Those
# stay specific, so an unhandled pair raises rather than recursing back through this one.
sectorproduct(s1, s2) = sectorproduct(Sector(s1), Sector(s2))
sectorproduct(s1, s2, s3, srest...) = foldl(sectorproduct, (s1, s2, s3, srest...))

# Stripping the unit here rather than by dispatch keeps `TrivialSector` out of the methods
# below, which would otherwise need a method per pairing to stay unambiguous with them. Both
# branches are resolved at compile time, since the argument types are concrete.
function sectorproduct(s1::Sector, s2::Sector)
    s1 isa TrivialSector && return s2
    s2 isa TrivialSector && return s1
    return _sectorproduct(s1, s2)
end

# A bare sector enters a product positionally, so these absorb into a `TupleSectorProduct`.
_sectorproduct(s1::Sector, s2::Sector) = TupleSectorProduct((s1, s2))
_sectorproduct(p::TupleSectorProduct, s::Sector) = TupleSectorProduct((arguments(p)..., s))
_sectorproduct(s::Sector, p::TupleSectorProduct) = TupleSectorProduct((s, arguments(p)...))
function _sectorproduct(p1::TupleSectorProduct, p2::TupleSectorProduct)
    return TupleSectorProduct((arguments(p1)..., arguments(p2)...))
end

# A named product absorbs by name, and so only from another named product.
function _sectorproduct(p1::NamedSectorProduct, p2::NamedSectorProduct)
    isdisjoint(keys(arguments(p1)), keys(arguments(p2))) ||
        throw(ArgumentError("names of a SectorProduct must be distinct"))
    return NamedSectorProduct(merge(arguments(p1), arguments(p2)))
end

# A named product multiplied by anything positional. The last two are each ambiguous between the
# first two, and agree with them, so they state that answer.
function _sectorproduct(a::NamedSectorProduct, b::Sector)
    return throw(
        ArgumentError(
            "cannot multiply $(a) by $(b): one is indexed by name and the other by position, " *
                "so the product would have to be indexed both ways"
        )
    )
end
function _sectorproduct(a::Sector, b::NamedSectorProduct)
    return throw(
        ArgumentError(
            "cannot multiply $(a) by $(b): one is indexed by name and the other by position, " *
                "so the product would have to be indexed both ways"
        )
    )
end
function _sectorproduct(a::NamedSectorProduct, b::TupleSectorProduct)
    return throw(
        ArgumentError(
            "cannot multiply $(a) by $(b): one is indexed by name and the other by position, " *
                "so the product would have to be indexed both ways"
        )
    )
end
function _sectorproduct(a::TupleSectorProduct, b::NamedSectorProduct)
    return throw(
        ArgumentError(
            "cannot multiply $(a) by $(b): one is indexed by name and the other by position, " *
                "so the product would have to be indexed both ways"
        )
    )
end

# ===========================  Promotion  ==================================================

# Bring two products onto a common argument set so they can be fused. Positional arguments are
# already matched, by position, so there is nothing to fill in and all that is left is to reject
# operands that are not elements of the same group. Named ones are brought onto the union of
# their keys, filling a key one side lacks with that symmetry's trivial sector. Between the two
# indexings there is no correspondence to bring about at all, which the fallback rejects.

function promote_sector(s1::SectorProduct, s2::SectorProduct)
    return throw(
        ArgumentError(
            "cannot combine $(s1) and $(s2): one is indexed by name and the other by position, " *
                "so there is no correspondence between their arguments"
        )
    )
end

function promote_sector(s1::TupleSectorProduct, s2::TupleSectorProduct)
    arguments_type(typeof(s1)) === arguments_type(typeof(s2)) || throw(
        ArgumentError(
            "cannot combine $(s1) and $(s2): positional arguments identify a symmetry by slot, " *
                "so the two must agree slot for slot on which symmetries they are elements of"
        )
    )
    return (s1, s2)
end

Base.@assume_effects :foldable function _sorted_union(::Val{K1}, ::Val{K2}) where {K1, K2}
    return Tuple(sort(union(K1, K2)))
end

function promote_sector(
        s1::NamedSectorProduct{<:NamedTuple{K1}}, s2::NamedSectorProduct{<:NamedTuple{K2}}
    ) where {K1, K2}
    allkeys = _sorted_union(Val(K1), Val(K2))
    for k in allkeys
        si1 = get(arguments(s1), k, TrivialSector())
        si2 = get(arguments(s2), k, TrivialSector())
        si1 isa TrivialSector ||
            si2 isa TrivialSector ||
            (typeof(si1) == typeof(si2)) ||
            throw(
            ArgumentError(
                "Cannot canonicalize SectorProduct with different non-trivial arguments"
            )
        )
    end
    s1′ = NamedSectorProduct(
        NamedTuple{allkeys}(
            ntuple(length(allkeys)) do i
                k = allkeys[i]
                arg = get(arguments(s1), k, TrivialSector())
                if arg isa TrivialSector
                    return trivial(getproperty(arguments(s2), k))
                else
                    return arg
                end
            end
        )
    )
    s2′ = NamedSectorProduct(
        NamedTuple{allkeys}(
            ntuple(length(allkeys)) do i
                k = allkeys[i]
                arg = get(arguments(s2), k, TrivialSector())
                if arg isa TrivialSector
                    return trivial(getproperty(arguments(s1), k))
                else
                    return arg
                end
            end
        )
    )
    return s1′, s2′
end

function promote_sector(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
    s1′, s2′ = promote_sector(s1, s2)
    s1″, s3′ = promote_sector(s1′, s3)
    s2″, s3″ = promote_sector(s2′, s3′)
    return s1″, s2″, s3″
end

@generated function sort_keys(nt::NamedTuple{N}) where {N}
    return :(NamedTuple{$(Tuple(sort(collect(N))))}(nt))
end
