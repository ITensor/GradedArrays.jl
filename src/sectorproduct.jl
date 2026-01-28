# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

# =====================================  Definition  =======================================
struct SectorProduct{Sectors} <: TKS.Sector
    arguments::Sectors
    global _SectorProduct(l) = new{typeof(l)}(l)
end

const SectorProductRange{T <: SectorProduct} = SectorRange{T}

SectorProduct(t::Tuple) = _SectorProduct(t)
function SectorProduct(nt::NamedTuple)
    arguments = sort_keys(nt)
    return _SectorProduct(arguments)
end
SectorProduct(; kws...) = SectorProduct((; kws...))

SectorProduct(x::TKS.Sector...) = _SectorProduct(x)
SectorProduct(c::SectorProduct) = _SectorProduct(arguments(c))
SectorProduct(c::SectorRange...) = _SectorProduct(map(label, c))
# SectorProduct(::TKS.Trivial) = _SectorProduct((;))  # empty tuple

arguments(s::SectorProduct) = getfield(s, :arguments)
arguments_type(::Type{SectorProduct{T}}) where {T} = T

function arguments(r::SectorProductRange)
    return map(SectorRange, arguments(label(r)))
end

function to_sector(nt::NamedTuple{<:Any, T}) where {T <: Tuple{Vararg{TKS.Sector}}}
    return SectorRange(SectorProduct(nt))
end
function to_sector(nt::NamedTuple{<:Any, T}) where {T <: Tuple{Vararg{SectorRange}}}
    return SectorRange(SectorProduct(NamedTuple(k => label(v) for (k, v) in pairs(nt))))
end
to_sector(nt::@NamedTuple{}) = to_sector(_SectorProduct(nt))

# =================================  Sectors interface  ====================================

function TKS.FusionStyle(::Type{SectorProduct{T}}) where {T}
    return mapreduce(TKS.FusionStyle, &, fieldtypes(T); init = TKS.UniqueFusion())
end
function TKS.BraidingStyle(::Type{SectorProduct{T}}) where {T}
    return mapreduce(TKS.BraidingStyle, &, fieldtypes(T); init = TKS.Bosonic())
end

TKS.dim(s::SectorProduct) = prod(TKS.dim, arguments(s); init = 1)

# use map instead of broadcast to support both Tuple and NamedTuple
TKS.dual(s::SectorProduct) = SectorProduct(map(TKS.dual, arguments(s)))

function TKS.unit(::Type{SectorProduct{T}}) where {T <: Tuple}
    return SectorProduct(map(TKS.unit, fieldtypes(T)))
end
function TKS.unit(::Type{SectorProduct{NT}}) where {NT <: NamedTuple}
    return SectorProduct(NT(map(TKS.unit, fieldtypes(NT))))
end
Base.isone(s::SectorProduct) = all(isone, arguments(s))

is_global_trivial(s::TKS.Sector) = false
is_global_trivial(s::SectorProduct) = isempty(arguments(s))
is_global_trivial(s::TKS.Trivial) = true

function TKS.otimes(s1::SectorProduct, s2::SectorProduct)
    is_global_trivial(s1) && is_global_trivial(s2) && return (SectorProduct((;)),)
    is_global_trivial(s1) && return TKS.otimes(one(s2), s2)
    is_global_trivial(s2) && return TKS.otimes(s1, one(s1))
    return TKS.otimes(arguments_canonicalize(s1, s2)...)
end
TKS.otimes(s1::SectorProduct, s2::TKS.Sector) = TKS.otimes(s1, SectorProduct(s2))
TKS.otimes(s1::TKS.Sector, s2::SectorProduct) = TKS.otimes(SectorProduct(s1), s2)

function TKS.otimes(s1::I, s2::I) where {I <: SectorProduct{<:Tuple}}
    isempty(arguments(s1)) && return (s2,)
    arg_otimes = map(TKS.otimes, arguments(s1), arguments(s2))
    prod_otimes = Iterators.map(splat(SectorProduct), Iterators.product(arg_otimes...))
    return TKS.SectorSet{I}(prod_otimes)
end
function TKS.otimes(s1::I, s2::I) where {T <: NamedTuple, I <: SectorProduct{T}}
    isempty(arguments(s1)) && return (s2,)
    arg_otimes = map(TKS.otimes, arguments(s1), arguments(s2))
    prod_otimes = Iterators.map(Iterators.product(arg_otimes...)) do factors
        return SectorProduct(T(factors))
    end
    return TKS.SectorSet{I}(prod_otimes)
end

# multiple dispatch through explicit loop
const _TKSSector = TKS.Sector
for T1 in (:SectorProduct, :_TKSSector),
        T2 in (:SectorProduct, :_TKSSector),
        T3 in (:SectorProduct, :_TKSSector)

    T1 === T2 === T3 && continue
    @eval function TKS.Nsymbol(s1::$T1, s2::$T2, s3::$T3)
        return TKS.Nsymbol(SectorProduct(s1), SectorProduct(s2), SectorProduct(s3))
    end
end
function TKS.Nsymbol(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
    is_global_trivial(s1) && is_global_trivial(s2) && return isone(s3) ? 1 : 0
    is_global_trivial(s1) && return TKS.Nsymbol(one(s2), s2, s3)
    is_global_trivial(s2) && return TKS.Nsymbol(s1, one(s1), s3)
    is_global_trivial(s3) && return TKS.Nsymbol(s1, s2, one(s1))

    s1_can, s2_can, s3_can = arguments_canonicalize(s1, s2, s3)
    return prod(
        splat(TKS.Nsymbol), zip(arguments(s1_can), arguments(s2_can), arguments(s3_can)); init = 1
    )
end

# ===================================  Base interface  =====================================

function Base.:(==)(A::SectorProduct, B::SectorProduct)
    isempty(arguments(A)) && return isone(B)
    isempty(arguments(B)) && return isone(A)
    A′, B′ = arguments_canonicalize(A, B)
    return all(splat(==), zip(arguments(A′), arguments(B′)))
end
Base.:(==)(A::SectorProduct, B::TKS.Sector) = A == SectorProduct(B)
Base.:(==)(A::TKS.Sector, B::SectorProduct) = SectorProduct(A) == B

function Base.isless(s1::SectorProduct, s2::SectorProduct)
    isempty(arguments(s1)) && isempty(arguments(s2)) && return false
    isempty(arguments(s1)) && return one(s2) < s2
    isempty(arguments(s2)) && return s1 < one(s1)
    s1′, s2′ = arguments_canonicalize(s1, s2)
    return arguments(s1′) < arguments(s2′)
end
Base.isless(s1::SectorProduct, s2::TKS.Sector) = s1 < SectorProduct(s2)
Base.isless(s1::TKS.Sector, s2::SectorProduct) = SectorProduct(s1) < s2

function Base.isless(s1::SectorProductRange, s2::SectorProductRange)
    isempty(arguments(s1)) && isempty(arguments(s2)) && return false
    isempty(arguments(s1)) && return trivial(s2) < s2
    isempty(arguments(s2)) && return s1 < trivial(s1)
    s1′, s2′ = arguments_canonicalize(s1, s2)
    return arguments(s1′) < arguments(s2′)
end

function Base.show(io::IO, r::SectorProductRange)
    (length(arguments(r)) < 2) && print(io, "sector")
    print(io, "(")
    symbol = ""
    for (k, v) in pairs(arguments(r))
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

const sectorproduct = ×

×(c::SectorRange) = SectorRange(SectorProduct(label(c)))
×(c1::SectorRange, c2::SectorRange) = SectorRange(×(label(c1), label(c2)))
×(c1::TKS.Sector, c2::TKS.Sector) = ×(SectorProduct(c1), SectorProduct(c2))

function ×(p1::SectorProduct{<:Tuple}, p2::SectorProduct{<:Tuple})
    return SectorProduct(arguments(p1)..., arguments(p2)...)
end
function ×(p1::SectorProduct{<:NamedTuple}, p2::SectorProduct{<:NamedTuple})
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
×(c1::NamedTuple, c2::SectorRange) = ×(to_sector(c1), c2)
×(c1::SectorRange, c2::NamedTuple) = ×(c1, to_sector(c2))

function ×(pairs::Pair...)
    keys = Symbol.(first.(pairs))
    vals = last.(pairs)
    return ×(NamedTuple{keys}(vals))
end

function ×(sr1::SectorOneTo, sr2::SectorOneTo)
    isdual(sr1) == isdual(sr2) || throw(ArgumentError("SectorProduct duality must match"))
    return sectorrange(
        sector(sr1) × sector(sr2), sector_multiplicity(sr1) * sector_multiplicity(sr2);
        isdual = isdual(sr1),
    )
end

# TODO: type piracy?
KroneckerArrays.to_product_indices(nt::NamedTuple) =
    KroneckerArrays.to_product_indices(to_sector(nt))

# ===========================  Canonicalize arguments  =====================================

function arguments_canonicalize(s1::SectorProduct{<:Tuple}, s2::SectorProduct{<:Tuple})
    # isempty(arguments(s1)) && return (one(s2), s2)
    # isempty(arguments(s2)) && return (s1, one(s1))
    lmin = min(length(arguments(s1)), length(arguments(s2)))
    for i in 1:lmin
        typeof(arguments(s1)[i]) == TKS.Trivial ||
            typeof(arguments(s2)[i]) == TKS.Trivial ||
            typeof(arguments(s1)[i]) == typeof(arguments(s2)[i]) ||
            throw(
            ArgumentError(
                "Cannot canonicalize SectorProduct with different non-trivial arguments"
            ),
        )
    end
    lmax = max(length(arguments(s1)), length(arguments(s2)))
    s1′ = SectorProduct(
        ntuple(lmax) do i
            if i <= length(arguments(s1))
                si = arguments(s1)[i]
                si != TKS.Trivial() && return si
            end
            return one(arguments(s2)[i])
        end,
    )
    s2′ = SectorProduct(
        ntuple(lmax) do i
            if i <= length(arguments(s2))
                si = arguments(s2)[i]
                si != TKS.Trivial() && return si
            end
            return one(arguments(s1)[i])
        end,
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
        si1 = get(arguments(s1), k, TKS.Trivial())
        si2 = get(arguments(s2), k, TKS.Trivial())
        si1 == TKS.Trivial() ||
            si2 == TKS.Trivial() ||
            (typeof(si1) == typeof(si2)) ||
            throw(
            ArgumentError(
                "Cannot canonicalize SectorProduct with different non-trivial arguments"
            ),
        )
    end
    s1′ = SectorProduct(
        NamedTuple{allkeys}(
            ntuple(length(allkeys)) do i
                k = allkeys[i]
                si = get(arguments(s1), k, TKS.Trivial())
                if si === TKS.Trivial()
                    return one(getproperty(arguments(s2), k))
                else
                    return si
                end
            end,
        )
    )
    s2′ = SectorProduct(
        NamedTuple{allkeys}(
            ntuple(length(allkeys)) do i
                k = allkeys[i]
                si = get(arguments(s2), k, TKS.Trivial())
                if si === TKS.Trivial()
                    return one(getproperty(arguments(s1), k))
                else
                    return si
                end
            end,
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

function arguments_canonicalize(s1::SectorProductRange, s2::SectorProductRange)
    s1′, s2′ = arguments_canonicalize(label(s1), label(s2))
    return SectorRange(s1′), SectorRange(s2′)
end
function arguments_canonicalize(
        s1::SectorProductRange, s2::SectorProductRange, s3::SectorProductRange
    )
    s1′, s2′, s3′ = arguments_canonicalize(label(s1), label(s2), label(s3))
    return SectorRange(s1′), SectorRange(s2′), SectorRange(s3′)
end

@generated function sort_keys(nt::NamedTuple{N}) where {N}
    return :(NamedTuple{$(Tuple(sort(collect(N))))}(nt))
end
