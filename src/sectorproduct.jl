# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

# =====================================  Definition  =======================================

"""
    SectorProduct(sectors...)
    SectorProduct(sectors::Tuple)
    SectorProduct(sectors::NamedTuple)

The Cartesian product of two or more sectors, itself a [`Sector`](@ref). Its arguments are
bare sectors, so like any other sector it carries no arrow of its own.

A `NamedTuple` of arguments names the factors, which lets products over different sets of
symmetries be compared and fused: a missing argument acts as that symmetry's trivial sector.
"""
struct SectorProduct{Arguments} <: Sector
    arguments::Arguments
    global _SectorProduct(l) = new{typeof(l)}(l)
end

SectorProduct(t::Tuple) = _SectorProduct(map(to_sector, t))
SectorProduct(nt::NamedTuple) = _SectorProduct(map(to_sector, sort_keys(nt)))
SectorProduct(; kws...) = SectorProduct((; kws...))

SectorProduct(s::SectorProduct) = s
SectorProduct(ss::Union{Sector, TKS.Sector}...) = SectorProduct(ss)

arguments(s::SectorProduct) = getfield(s, :arguments)
arguments_type(::Type{SectorProduct{T}}) where {T} = T

function to_sector(nt::NamedTuple{<:Any, <:Tuple{Vararg{Union{Sector, TKS.Sector}}}})
    return SectorProduct(nt)
end

label(s::SectorProduct) = map(label, arguments(s))

# The TensorKitSectors counterpart, used for ordering and by anything that reaches for a
# sector's TensorKitSectors form. It is one-way: a `ProductSector` records neither the argument
# names nor a missing argument, so `to_sector` cannot rebuild the `SectorProduct` and returns a
# `TensorKitSector` instead. Fusion therefore does not go through it either, since a
# `ProductSector` cannot express the mismatched argument sets `arguments_canonicalize` aligns.
function tensorkitsector(s::SectorProduct)
    return TKS.ProductSector(map(tensorkitsector, values(arguments(s))))
end
function tensorkitsectortype(::Type{SectorProduct{T}}) where {T}
    return TKS.ProductSector{Tuple{map(tensorkitsectortype, fieldtypes(T))...}}
end

# =================================  Sectors interface  ====================================

function TKS.FusionStyle(::Type{SectorProduct{T}}) where {T}
    return mapreduce(TKS.FusionStyle, &, fieldtypes(T); init = TKS.UniqueFusion())
end
function TKS.BraidingStyle(::Type{SectorProduct{T}}) where {T}
    return mapreduce(TKS.BraidingStyle, &, fieldtypes(T); init = TKS.Bosonic())
end

Base.length(s::SectorProduct) = prod(length, arguments(s); init = 1)

# Fermion parity and twist of a product are the xor and the product of its arguments'. Taking
# them argument by argument also covers the empty product, which has no `ProductSector` form to
# delegate to.
fermionparity(s::SectorProduct) = mapreduce(fermionparity, ⊻, arguments(s); init = false)
twist(s::SectorProduct) = prod(twist, arguments(s); init = 1)

# use map instead of broadcast to support both Tuple and NamedTuple
function charge_conjugate(s::SectorProduct)
    return SectorProduct(map(charge_conjugate, arguments(s)))
end

function trivial(::Type{SectorProduct{T}}) where {T <: Tuple}
    return SectorProduct(map(trivial, fieldtypes(T)))
end
function trivial(::Type{SectorProduct{NT}}) where {NT <: NamedTuple}
    return SectorProduct(NT(map(trivial, fieldtypes(NT))))
end
istrivial(s::SectorProduct) = all(istrivial, arguments(s))

# A product with no arguments at all is trivial for every symmetry, so it fuses with anything.
is_global_trivial(::Sector) = false
is_global_trivial(::TrivialSector) = true
is_global_trivial(s::SectorProduct) = isempty(arguments(s))

# ===============================  Fusion rule interface  ==================================

# The fusion of two products is the product of its arguments' fusions, so it is built argument
# by argument.
function fusion_rule(s1::SectorProduct, s2::SectorProduct)
    is_global_trivial(s1) && is_global_trivial(s2) && return s1
    is_global_trivial(s1) && return fusion_rule(trivial(s2), s2)
    is_global_trivial(s2) && return fusion_rule(s1, trivial(s1))
    s1′, s2′ = arguments_canonicalize(s1, s2)
    fstyle = TKS.FusionStyle(typeof(s1′)) & TKS.FusionStyle(typeof(s2′))
    fstyle === TKS.UniqueFusion() &&
        return SectorProduct(map(fusion_rule, arguments(s1′), arguments(s2′)))
    return gradedrange([s => nsymbol(s1′, s2′, s) for s in fusion_products(s1′, s2′)])
end
fusion_rule(s1::SectorProduct, s2::Sector) = fusion_rule(s1, SectorProduct(s2))
fusion_rule(s1::Sector, s2::SectorProduct) = fusion_rule(SectorProduct(s1), s2)
# `TrivialSector` has its own methods against any `Sector`, which these would otherwise be
# ambiguous with.
fusion_rule(s1::SectorProduct, s2::TrivialSector) = fusion_rule(s1, SectorProduct(s2))
fusion_rule(s1::TrivialSector, s2::SectorProduct) = fusion_rule(SectorProduct(s1), s2)

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
rebuild_arguments(::SectorProduct{<:Tuple}, args::Tuple) = SectorProduct(args)
function rebuild_arguments(::SectorProduct{NT}, args::Tuple) where {NT <: NamedTuple}
    return SectorProduct(NT(args))
end

# multiple dispatch through explicit loop
for T1 in (:SectorProduct, :Sector),
        T2 in (:SectorProduct, :Sector),
        T3 in (:SectorProduct, :Sector)

    T1 === T2 === T3 && continue
    @eval function nsymbol(s1::$T1, s2::$T2, s3::$T3)
        return nsymbol(SectorProduct(s1), SectorProduct(s2), SectorProduct(s3))
    end
end
function nsymbol(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
    is_global_trivial(s1) && is_global_trivial(s2) && return istrivial(s3) ? 1 : 0
    is_global_trivial(s1) && return nsymbol(trivial(s2), s2, s3)
    is_global_trivial(s2) && return nsymbol(s1, trivial(s1), s3)
    is_global_trivial(s3) && return nsymbol(s1, s2, trivial(s1))

    s1_can, s2_can, s3_can = arguments_canonicalize(s1, s2, s3)
    return prod(
        splat(nsymbol), zip(arguments(s1_can), arguments(s2_can), arguments(s3_can));
        init = 1
    )
end

# ===================================  Base interface  =====================================

function Base.:(==)(a::SectorProduct, b::SectorProduct)
    isempty(arguments(a)) && return istrivial(b)
    isempty(arguments(b)) && return istrivial(a)
    a′, b′ = arguments_canonicalize(a, b)
    return all(splat(==), zip(arguments(a′), arguments(b′)))
end
Base.:(==)(a::SectorProduct, b::Sector) = a == SectorProduct(b)
Base.:(==)(a::Sector, b::SectorProduct) = SectorProduct(a) == b
Base.:(==)(a::SectorProduct, b::TrivialSector) = a == SectorProduct(b)
Base.:(==)(a::TrivialSector, b::SectorProduct) = SectorProduct(a) == b
Base.:(==)(a::SectorProduct, b::TKS.Sector) = a == SectorProduct(b)
Base.:(==)(a::TKS.Sector, b::SectorProduct) = SectorProduct(a) == b

# Order product sectors the way TensorKit orders `ProductSector`s: by total degree first, then
# lexicographically, not the plain lexicographic order a tuple comparison gives. This keeps a
# `SectorProduct` axis sorted the same way its TensorKit `GradedSpace` is. Canonicalizing first
# aligns the argument shapes, so both sides build the same `ProductSector` type.
function Base.isless(s1::SectorProduct, s2::SectorProduct)
    isempty(arguments(s1)) && isempty(arguments(s2)) && return false
    isempty(arguments(s1)) && return trivial(s2) < s2
    isempty(arguments(s2)) && return s1 < trivial(s1)
    s1′, s2′ = arguments_canonicalize(s1, s2)
    return isless(tensorkitsector(s1′), tensorkitsector(s2′))
end
Base.isless(s1::SectorProduct, s2::Sector) = s1 < SectorProduct(s2)
Base.isless(s1::Sector, s2::SectorProduct) = SectorProduct(s1) < s2
Base.isless(s1::SectorProduct, s2::TrivialSector) = s1 < SectorProduct(s2)
Base.isless(s1::TrivialSector, s2::SectorProduct) = SectorProduct(s1) < s2

function Base.show(io::IO, s::SectorProduct)
    (length(arguments(s)) < 2) && print(io, "sector")
    print(io, "(")
    symbol = ""
    for (k, v) in pairs(arguments(s))
        print(io, symbol)
        sector_show(io, k, v)
        symbol = " × "
    end
    return print(io, ")")
end

sector_show(io::IO, ::Int, v) = show(io, v)
function sector_show(io::IO, k::Symbol, v)
    print(io, '(', k, '=')
    show(io, v)
    print(io, ",)")
    return nothing
end

# =================================  Cartesian Product  ====================================

# Multi-argument sector product, expressed as a left fold over the binary methods.
# The binary methods below stay specific so an unhandled pair is a `MethodError`
# rather than recursing back into the fold.
×(x) = x
×(x, y, z, zs...) = foldl(×, (x, y, z, zs...))
const sectorproduct = ×

×(s::Sector) = SectorProduct(s)
×(s1::Sector, s2::Sector) = ×(SectorProduct(s1), SectorProduct(s2))
×(c1::TKS.Sector, c2::TKS.Sector) = ×(to_sector(c1), to_sector(c2))

function ×(p1::SectorProduct{<:Tuple}, p2::SectorProduct{<:Tuple})
    return SectorProduct(arguments(p1)..., arguments(p2)...)
end
function ×(
        p1::SectorProduct{<:NamedTuple},
        p2::SectorProduct{<:NamedTuple}
    )
    isdisjoint(keys(arguments(p1)), keys(arguments(p2))) ||
        throw(ArgumentError("keys of SectorProducts must be distinct"))
    return SectorProduct(merge(arguments(p1), arguments(p2)))
end
function ×(a::SectorProduct, b::SectorProduct)
    isempty(arguments(a)) && return b
    isempty(arguments(b)) && return a
    throw(MethodError(×, typeof.((a, b))))
end

×(nt1::NamedTuple) = to_sector(nt1)
×(nt1::NamedTuple, nt2::NamedTuple) = ×(to_sector(nt1), to_sector(nt2))
×(c1::NamedTuple, c2::Sector) = ×(to_sector(c1), c2)
×(c1::Sector, c2::NamedTuple) = ×(c1, to_sector(c2))

function ×(pairs::Pair...)
    keys = Symbol.(first.(pairs))
    vals = last.(pairs)
    return ×(NamedTuple{keys}(vals))
end

function ×(r1::SectorOneTo, r2::SectorOneTo)
    isdual(r1) == isdual(r2) || throw(ArgumentError("SectorProduct duality must match"))
    new_datalength = datalength(r1) * datalength(r2)
    return SectorOneTo(sector(r1) × sector(r2), isdual(r1), new_datalength)
end

# ===========================  Canonicalize arguments  =====================================

# Align two products onto a common argument set so they can be compared and fused: an argument
# one of them is missing, or holds as `TrivialSector`, is filled in with the other's trivial
# sector for that symmetry.

function arguments_canonicalize(s1::SectorProduct{<:Tuple}, s2::SectorProduct{<:Tuple})
    lmin = min(length(arguments(s1)), length(arguments(s2)))
    for i in 1:lmin
        arguments(s1)[i] isa TrivialSector ||
            arguments(s2)[i] isa TrivialSector ||
            typeof(arguments(s1)[i]) == typeof(arguments(s2)[i]) ||
            throw(
            ArgumentError(
                "Cannot canonicalize SectorProduct with different non-trivial arguments"
            )
        )
    end
    lmax = max(length(arguments(s1)), length(arguments(s2)))
    s1′ = SectorProduct(
        ntuple(lmax) do i
            if i <= length(arguments(s1))
                arg = arguments(s1)[i]
                arg isa TrivialSector || return arg
            end
            return trivial(arguments(s2)[i])
        end
    )
    s2′ = SectorProduct(
        ntuple(lmax) do i
            if i <= length(arguments(s2))
                arg = arguments(s2)[i]
                arg isa TrivialSector || return arg
            end
            return trivial(arguments(s1)[i])
        end
    )
    return s1′, s2′
end

Base.@assume_effects :foldable function _sorted_union(::Val{K1}, ::Val{K2}) where {K1, K2}
    return Tuple(sort(union(K1, K2)))
end

function arguments_canonicalize(
        s1::SectorProduct{<:NamedTuple{K1}}, s2::SectorProduct{<:NamedTuple{K2}}
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
    s1′ = SectorProduct(
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
    s2′ = SectorProduct(
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

function arguments_canonicalize(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
    s1′, s2′ = arguments_canonicalize(s1, s2)
    s1″, s3′ = arguments_canonicalize(s1′, s3)
    s2″, s3″ = arguments_canonicalize(s2′, s3′)
    return s1″, s2″, s3″
end

@generated function sort_keys(nt::NamedTuple{N}) where {N}
    return :(NamedTuple{$(Tuple(sort(collect(N))))}(nt))
end
