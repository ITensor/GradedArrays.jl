# =====================================  Definition  =======================================

# The Cartesian product of sectors, itself a `Sector`. Its arguments are bare sectors.
abstract type SectorProduct <: Sector end

# A `SectorProduct` with positional arguments. The arity is part of the sector's identity, so no
# factor can be left out.
struct TupleSectorProduct{Arguments <: Tuple} <: SectorProduct
    arguments::Arguments
end

# A `SectorProduct` with named arguments, sorted by name. A symmetry it does not name is that
# symmetry's trivial sector, so products over different sets of symmetries can be compared and
# fused.
struct NamedSectorProduct{Arguments <: NamedTuple} <: SectorProduct
    arguments::Arguments
    function NamedSectorProduct(nt::NamedTuple)
        sorted = sort_keys(nt)
        return new{typeof(sorted)}(sorted)
    end
end

# The positional form takes two factors before its `Vararg` so it does not compete with the
# one-argument methods. A positional product as a factor would nest, which TensorKitSectors does
# not support, so it is rejected in favor of `sectorproduct`.
Sector(s1, s2, srest...) = Sector((s1, s2, srest...))
function Sector(t::Tuple)
    args = map(Sector, t)
    any(a -> a isa TupleSectorProduct, args) && throw(
        ArgumentError(
            "a positional product cannot be a factor of another positional product, use " *
                "`sectorproduct` (or `×`) to multiply products instead"
        )
    )
    return TupleSectorProduct(args)
end
Sector(nt::NamedTuple) = NamedSectorProduct(map(Sector, nt))
Sector(; kws...) = Sector(values(kws))

# Fusion works argument by argument, so a bare operand is read as a one-factor positional product.
to_sectorproduct(s::SectorProduct) = s
to_sectorproduct(s::Sector) = TupleSectorProduct((s,))

arguments(s::SectorProduct) = getfield(s, :arguments)
arguments_type(::Type{<:TupleSectorProduct{T}}) where {T} = T
arguments_type(::Type{<:NamedSectorProduct{T}}) where {T} = T

# Fusion does not go through these, since they cannot express the mismatched argument sets
# `promote_sector` handles.
TKS.Sector(s::TupleSectorProduct) = TKS.ProductSector(map(TKS.Sector, arguments(s)))
TKS.Sector(s::NamedSectorProduct) = TKS.NamedSector(map(TKS.Sector, arguments(s)))

# Otherwise a sector read back from TensorKit would not equal the one it was built from.
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

# Argument by argument, which also covers the empty product.
fermionparity(s::SectorProduct) = mapreduce(fermionparity, ⊻, arguments(s); init = false)
twist(s::SectorProduct) = prod(twist, arguments(s); init = 1)

# use map instead of broadcast to support both Tuple and NamedTuple
function dual_sector(s::SectorProduct)
    return Sector(map(dual_sector, arguments(s)))
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

# An empty product fuses as the trivial sector of the other operand.
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
# Disambiguate against the `Trivial` methods.
fusion_rule(s::SectorProduct, ::Trivial) = fusion_rule(s, trivial(s))
fusion_rule(::Trivial, s::SectorProduct) = fusion_rule(trivial(s), s)

# Every sector the fusion of `s1` and `s2` can produce. Both must already be promoted.
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

# A positional product compares all of its arguments, while a named product compares only its
# non-trivial ones. `promote_sector` is not used since it throws on incompatible operands.

# Iterating over a `zip` of the heterogeneous argument tuples would box each argument.
function Base.:(==)(a::TupleSectorProduct, b::TupleSectorProduct)
    return arguments(a) == arguments(b)
end

# Same names on both sides, so compare the values without boxing them.
Base.:(==)(a::P, b::P) where {P <: NamedSectorProduct} = arguments(a) == arguments(b)

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

Base.:(==)(::SectorProduct, ::Sector) = false
Base.:(==)(::Sector, ::SectorProduct) = false
Base.:(==)(::SectorProduct, ::SectorProduct) = false

# Disambiguate against `==(::Sector, ::Trivial)`.
Base.:(==)(::SectorProduct, ::Trivial) = false
Base.:(==)(::Trivial, ::SectorProduct) = false

# Hashes what `==` compares.
function Base.hash(s::TupleSectorProduct, h::UInt)
    return hash(
        :TupleSectorProduct,
        foldl((acc, a) -> hash(a, acc), arguments(s); init = h)
    )
end
function Base.hash(s::NamedSectorProduct, h::UInt)
    args = arguments(s)
    # Folded rather than looped over the names, which would box each value.
    h = foldl(map(tuple, keys(args), values(args)); init = h) do h, (k, v)
        return istrivial(v) ? h : hash(k, hash(v, h))
    end
    return hash(:NamedSectorProduct, h)
end

# Sectors of one type must be ordered as in TensorKit's `GradedSpace`. Products of different types
# never share an axis, so they only need some deterministic order.
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
Base.isless(::Trivial, ::SectorProduct) = true
Base.isless(::SectorProduct, ::Trivial) = false

# Print the spelling that rebuilds the sector, using `×` when there are two or more factors.
function Base.show(io::IO, s::TupleSectorProduct)
    args = arguments(s)
    if length(args) < 2
        # `Sector` reads a lone sector as itself, so the tuple is needed.
        print(io, "Sector((")
        isempty(args) || (show(io, only(args)); print(io, ","))
        return print(io, "))")
    end
    print(io, "(")
    join(io, (sprint(show, a; context = io) for a in args), " × ")
    return print(io, ")")
end

# A bare `NamedTuple` rather than a `Sector` call, to keep axis displays short. It still reads
# back in, since the graded constructors accept it.
function Base.show(io::IO, s::NamedSectorProduct)
    args = arguments(s)
    isempty(args) && return print(io, "(;)")
    print(io, "(")
    for (i, (k, v)) in enumerate(pairs(args))
        i > 1 && print(io, ", ")
        print(io, k, " = ")
        show(io, v)
    end
    # A one-field `NamedTuple` needs the trailing comma to read as one.
    length(args) == 1 && print(io, ",")
    return print(io, ")")
end

# Print a product type as the product of its factor types, or its alias if it has one.
function show_sectorproduct_type(io::IO, P::Type{<:TupleSectorProduct})
    isnothing(Base.make_typealias(P)) || return show(io, P)
    args = map(A -> sprint(show, A; context = io), fieldtypes(arguments_type(P)))
    if length(args) == 1
        # The empty product has no type-level spelling, so it keeps the default.
        print(io, "×((", only(args), ",))")
    elseif isempty(args)
        show(io, P)
    else
        join(io, args, " × ")
    end
    return nothing
end
function show_sectorproduct_type(io::IO, P::Type{<:NamedSectorProduct})
    isnothing(Base.make_typealias(P)) || return show(io, P)
    nt = arguments_type(P)
    fields = [
        "$(k) = $(sprint(show, V; context = io))"
            for (k, V) in zip(fieldnames(nt), fieldtypes(nt))
    ]
    if length(fields) == 1
        print(io, "×((; ", only(fields), "))")
    elseif isempty(fields)
        show(io, P)
    else
        join(io, ("(; $(field))" for field in fields), " × ")
    end
    return nothing
end

# =================================  Cartesian Product  ====================================

"""
    sectorproduct(ss...)
    ×(ss...)

The Cartesian product of the symmetries of `ss`. Each argument is normalized with
[`Sector`](@ref) first, so anything that specifies a sector can be multiplied.

A positional product absorbs factors positionally and a named one absorbs them by name.
Multiplying the two together is an error. `Trivial` is the unit and drops out of any product.

Sector types multiply too, which is how a fused symmetry gets a name of its own, as in
`const fU1 = U1 × fZ2`. A `NamedTuple` of sector types names the factors, and a `Tuple` or
`NamedTuple` is also how to spell a product of one of them: `(; charge = U1) × (; parity = fZ2)`,
`×((U1,))`. An empty container carries no type to name a symmetry with, so `×(())` and `×((;))`
stay the empty products' values.
"""
function sectorproduct end
const × = sectorproduct

# A symmetry named at the type level: a sector type, or a non-empty container of them. Empty
# containers are excluded since they already specify the empty products as values.
const SectorTypeSpec = Union{
    Type{<:Sector}, Tuple{Type, Vararg{Type}},
    NamedTuple{<:Any, <:Tuple{Type, Vararg{Type}}},
}
trivial(t::Tuple{Type, Vararg{Type}}) = Sector(map(trivial, t))
trivial(nt::NamedTuple{<:Any, <:Tuple{Type, Vararg{Type}}}) = Sector(map(trivial, nt))

# Through the value-level product of the trivial sectors, so the two levels agree.
function sectorproduct(S1::SectorTypeSpec, Srest::SectorTypeSpec...)
    return typeof(sectorproduct(trivial(S1), map(trivial, Srest)...))
end

# The unit. `Sector()` instead gives the empty named product.
sectorproduct() = Trivial()

# No one-factor product is ever built.
sectorproduct(s::Sector) = s
sectorproduct(s) = Sector(s)

sectorproduct(s1, s2) = sectorproduct(Sector(s1), Sector(s2))
sectorproduct(s1, s2, s3, srest...) = foldl(sectorproduct, (s1, s2, s3, srest...))

# Strip the unit here rather than by dispatch, which would need extra methods to avoid ambiguities.
function sectorproduct(s1::Sector, s2::Sector)
    s1 isa Trivial && return s2
    s2 isa Trivial && return s1
    return _sectorproduct(s1, s2)
end

_sectorproduct(s1::Sector, s2::Sector) = TupleSectorProduct((s1, s2))
_sectorproduct(p::TupleSectorProduct, s::Sector) = TupleSectorProduct((arguments(p)..., s))
_sectorproduct(s::Sector, p::TupleSectorProduct) = TupleSectorProduct((s, arguments(p)...))
function _sectorproduct(p1::TupleSectorProduct, p2::TupleSectorProduct)
    return TupleSectorProduct((arguments(p1)..., arguments(p2)...))
end

function _sectorproduct(p1::NamedSectorProduct, p2::NamedSectorProduct)
    isdisjoint(keys(arguments(p1)), keys(arguments(p2))) ||
        throw(ArgumentError("names of a SectorProduct must be distinct"))
    return NamedSectorProduct(merge(arguments(p1), arguments(p2)))
end

# The last two disambiguate the first two.
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

# Bring two products onto a common argument set so they can be fused. Named products are brought
# onto the union of their names, filling a missing name with that symmetry's trivial sector.

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
        si1 = get(arguments(s1), k, Trivial())
        si2 = get(arguments(s2), k, Trivial())
        si1 isa Trivial ||
            si2 isa Trivial ||
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
                arg = get(arguments(s1), k, Trivial())
                if arg isa Trivial
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
                arg = get(arguments(s2), k, Trivial())
                if arg isa Trivial
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
