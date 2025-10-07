# This files defines a structure for Cartesian product of 2 or more fusion sectors
# e.g. U(1)×U(1), U(1)×SU2(2)×SU(3)

# =====================================  Definition  =======================================
struct SectorProduct{Sectors} <: AbstractSector
  arguments::Sectors
  global _SectorProduct(l) = new{typeof(l)}(l)
end

const SectorProductRange = SectorRange{SectorProduct}

SectorProduct(t::Tuple) = _SectorProduct(t)
function SectorProduct(nt::NamedTuple)
  arguments = sort_keys(nt)
  return _SectorProduct(arguments)
end
SectorProduct(; kws...) = SectorProduct((; kws...))

SectorProduct(x::AbstractSector...) = _SectorProduct(x)
SectorProduct(c::SectorProduct) = _SectorProduct(arguments(c))
SectorProduct(c::SectorRange...) = _SectorProduct(map(sector, c))

arguments(s::SectorProduct) = s.arguments
arguments_type(::Type{SectorProduct{T}}) where {T} = T

function to_sector(nt::NamedTuple{<:Any,T}) where {T<:Tuple{Vararg{AbstractSector}}}
  return SectorRange(SectorProduct(nt))
end
function to_sector(nt::NamedTuple{<:Any,T}) where {T<:Tuple{Vararg{SectorRange}}}
  return SectorRange(SectorProduct(NamedTuple(k => sector(v) for (k, v) in pairs(nt))))
end
to_sector(nt::@NamedTuple{}) = to_sector(_SectorProduct(nt))

# =================================  Sectors interface  ====================================

function TKS.FusionStyle(::Type{SectorProduct{T}}) where {T}
  return mapreduce(TKS.FusionStyle, &, fieldtypes(T); init=TKS.UniqueFusion())
end
function TKS.BraidingStyle(::Type{SectorProduct{T}}) where {T}
  return mapreduce(TKS.BraidingStyle, &, fieldtypes(T); init=TKS.Bosonic())
end

TKS.dim(s::SectorProduct) = prod(TKS.dim, arguments(s); init=1)

# use map instead of broadcast to support both Tuple and NamedTuple
TKS.dual(s::SectorProduct) = SectorProduct(map(TKS.dual, arguments(s)))

function Base.one(::Type{SectorProduct{T}}) where {T<:Tuple}
  return SectorProduct(map(one, fieldtypes(T)))
end
function Base.one(::Type{SectorProduct{NT}}) where {NT<:NamedTuple}
  return SectorProduct(NT(map(one, fieldtypes(NT))))
end
Base.isone(s::SectorProduct) = all(isone, arguments(s))

function TKS.otimes(s1::SectorProduct, s2::SectorProduct)
  isempty(arguments(s1)) && return (s2,)
  isempty(arguments(s2)) && return (s1,)
  return TKS.otimes(arguments_canonicalize(s1, s2)...)
end
TKS.otimes(s1::SectorProduct, s2::AbstractSector) = TKS.otimes(s1, SectorProduct(s2))
TKS.otimes(s1::AbstractSector, s2::SectorProduct) = TKS.otimes(SectorProduct(s1), s2)

function TKS.otimes(s1::I, s2::I) where {I<:SectorProduct{<:Tuple}}
  isempty(arguments(s1)) && return (s2,)
  arg_otimes = map(TKS.otimes, arguments(s1), arguments(s2))
  prod_otimes = Iterators.map(splat(SectorProduct), Iterators.product(arg_otimes...))
  return TKS.SectorSet{I}(prod_otimes)
end
function TKS.otimes(s1::I, s2::I) where {T<:NamedTuple,I<:SectorProduct{T}}
  isempty(arguments(s1)) && return (s2,)
  arg_otimes = map(TKS.otimes, arguments(s1), arguments(s2))
  prod_otimes = Iterators.map(Iterators.product(arg_otimes...)) do factors
    return SectorProduct(T(factors))
  end
  return TKS.SectorSet{I}(prod_otimes)
end

function TKS.Nsymbol(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
  if isempty(arguments(s1)) && isempty(arguments(s2)) && isempty(arguments(s3))
    return 1
  elseif isempty(arguments(s1)) && isempty(arguments(s2))
    return isone(s3) ? 1 : 0
  elseif isempty(arguments(s1)) && isempty(arguments(s3))
    return isone(s2) ? 1 : 0
  elseif isempty(arguments(s2)) && isempty(arguments(s3))
    return isone(s1) ? 1 : 0
  elseif isempty(arguments(s1))
    return TKS.Nsymbol(one(s2), s2, s3)
  elseif isempty(arguments(s2))
    return TKS.Nsymbol(s1, one(s1), s3)
  elseif isempty(arguments(s3))
    return TKS.Nsymbol(s1, s2, one(s1))
  end
  s1_can, s2_can, s3_can = arguments_canonicalize(s1, s2, s3)
  return prod(
    splat(TKS.Nsymbol), zip(arguments(s1_can), arguments(s2_can), arguments(s3_can)); init=1
  )
end
# multiple dispatch hell :(
function TKS.Nsymbol(s1::SectorProduct, s2::SectorProduct, s3::AbstractSector)
  return TKS.Nsymbol(s1, s2, SectorProduct(s3))
end
function TKS.Nsymbol(s1::SectorProduct, s2::AbstractSector, s3::SectorProduct)
  return TKS.Nsymbol(s1, SectorProduct(s2), s3)
end
function TKS.Nsymbol(s1::AbstractSector, s2::SectorProduct, s3::SectorProduct)
  return TKS.Nsymbol(SectorProduct(s1), s2, s3)
end
function TKS.Nsymbol(s1::SectorProduct, s2::AbstractSector, s3::AbstractSector)
  return TKS.Nsymbol(s1, SectorProduct(s2), SectorProduct(s3))
end
function TKS.Nsymbol(s1::AbstractSector, s2::SectorProduct, s3::AbstractSector)
  return TKS.Nsymbol(SectorProduct(s1), s2, SectorProduct(s3))
end
function TKS.Nsymbol(s1::AbstractSector, s2::AbstractSector, s3::SectorProduct)
  return TKS.Nsymbol(SectorProduct(s1), SectorProduct(s2), s3)
end

# ===================================  Base interface  =====================================

function Base.:(==)(A::SectorProduct, B::SectorProduct)
  isempty(arguments(A)) && return isone(B)
  isempty(arguments(B)) && return isone(A)
  A′, B′ = arguments_canonicalize(A, B)
  return all(splat(==), zip(arguments(A′), arguments(B′)))
end
Base.:(==)(A::SectorProduct, B::AbstractSector) = A == SectorProduct(B)
Base.:(==)(A::AbstractSector, B::SectorProduct) = SectorProduct(A) == B

function Base.isless(s1::SectorProduct, s2::SectorProduct)
  isempty(arguments(s1)) && isempty(arguments(s2)) && return false
  isempty(arguments(s1)) && return one(s2) < s2
  isempty(arguments(s2)) && return s1 < one(s1)
  s1′, s2′ = arguments_canonicalize(s1, s2)
  return arguments(s1′) < arguments(s2′)
end
Base.isless(s1::SectorProduct, s2::AbstractSector) = s1 < SectorProduct(s2)
Base.isless(s1::AbstractSector, s2::SectorProduct) = SectorProduct(s1) < s2

function Base.show(io::IO, r::SectorRange{<:SectorProduct})
  s = sector(r)
  (length(arguments(s)) < 2) && print(io, "sector")
  print(io, "(")
  symbol = ""
  for p in pairs(arguments(s))
    print(io, symbol)
    sector_show(io, p[1], SectorRange(p[2]))
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
×(c::SectorRange) = SectorRange(SectorProduct(sector(c)))
×(c1::SectorRange, c2::SectorRange) = SectorRange(×(sector(c1), sector(c2)))
×(c1::AbstractSector, c2::AbstractSector) = ×(SectorProduct(c1), SectorProduct(c2))

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

×(a, g::AbstractUnitRange) = ×(to_gradedrange(a), g)
×(g::AbstractUnitRange, b) = ×(g, to_gradedrange(b))
×(a::SectorRange, g::AbstractUnitRange) = ×(to_gradedrange(a), g)
×(g::AbstractUnitRange, b::SectorRange) = ×(g, to_gradedrange(b))

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
    sector(sr1) × sector(sr2),
    sector_multiplicity(sr1) * sector_multiplicity(sr2),
    isdual(sr1),
  )
end

function ×(g1::GradedOneTo, g2::GradedOneTo)
  v = map(
    splat(×), Iterators.flatten((Iterators.product(eachblockaxis(g1), eachblockaxis(g2)),),)
  )
  return mortar_axis(v)
end

# ===========================  Canonicalize arguments  =====================================

# some issues to get things type stable so use generated functions :(
@generated function arguments_type_canonicalize(
  ::Type{T1}, ::Type{T2}
) where {T1<:SectorProduct{<:Tuple},T2<:SectorProduct{<:Tuple}}
  F1 = fieldtypes(fieldtypes(T1)[1])
  F2 = fieldtypes(fieldtypes(T2)[1])
  L1 = length(F1)
  L2 = length(F2)
  m = max(L1, L2)
  T = ntuple(m) do i
    if i <= L1 && i <= L2
      if F1[i] == F2[i]
        F1[i]
      elseif F1[i] == TKS.Trivial
        F2[i]
      elseif F2[i] == TKS.Trivial
        F1[i]
      else
        throw(
          ArgumentError(
            "Cannot canonicalize SectorProduct with different non-trivial arguments"
          ),
        )
      end
    elseif i <= L1
      F1[i]
    else
      F2[i]
    end
  end
  return :(SectorProduct{Tuple{$T...}})
end
@generated function arguments_type_canonicalize(
  ::Type{T1}, ::Type{T2}
) where {T1<:SectorProduct{<:NamedTuple},T2<:SectorProduct{<:NamedTuple}}
  K1 = fieldnames(fieldtypes(T1)[1])
  K2 = fieldnames(fieldtypes(T2)[1])
  F1 = fieldtypes(fieldtypes(T1)[1])
  F2 = fieldtypes(fieldtypes(T2)[1])
  allkeys = sort(union(K1, K2))
  allvals = map(allkeys) do k
    i1 = findfirst(==(k), K1)
    i2 = findfirst(==(k), K2)
    if isnothing(i1)
      return F2[i2]
    elseif isnothing(i2)
      return F1[i1]
    else
      if F1[i1] == F2[i2]
        return F1[i1]
      elseif F1[i1] == TKS.Trivial
        return F2[i2]
      elseif F2[i2] == TKS.Trivial
        return F1[i1]
      else
        throw(MethodError(arguments_type_canonicalize, (T1, T2)))
      end
    end
  end
  return :(SectorProduct{NamedTuple{($allkeys...,),Tuple{$(allvals...)}}})
end
function arguments_type_canonicalize(
  ::Type{T1}, ::Type{T2}
) where {T1<:SectorProduct,T2<:SectorProduct}
  length(fieldtypes(arguments_type(T1))) == 0 && return T2
  length(fieldtypes(arguments_type(T2))) == 0 && return T1
  throw(MethodError(arguments_type_canonicalize, (T1, T2)))
end
@inline function arguments_type_canonicalize(
  ::Type{T1}, ::Type{T2}, Ts::Type{T}...
) where {T1<:SectorProduct,T2<:SectorProduct,T<:SectorProduct}
  return arguments_type_canonicalize(arguments_type_canonicalize(T1, T2), Ts...)
end

function arguments_canonicalize(s1::SectorProduct, s2::SectorProduct)
  isempty(arguments(s1)) && return (one(s2), s2)
  isempty(arguments(s2)) && return (s1, one(s1))
  T = arguments_type_canonicalize(typeof(s1), typeof(s2))
  return arguments_canonicalize(T, s1), arguments_canonicalize(T, s2)
end
function arguments_canonicalize(s1::SectorProduct, s2::SectorProduct, s3::SectorProduct)
  T = arguments_type_canonicalize(typeof(s1), typeof(s2), typeof(s3))
  return arguments_canonicalize(T, s1),
  arguments_canonicalize(T, s2),
  arguments_canonicalize(T, s3)
end

function arguments_canonicalize(
  ::Type{SectorProduct{T}}, s::SectorProduct{<:Tuple}
)::SectorProduct{T} where {T<:Tuple}
  b = one(SectorProduct{T})
  a = ntuple(length(arguments(b))) do i
    if i <= length(arguments(s)) && !(arguments(s)[i] isa TKS.Trivial)
      arguments(s)[i]
    else
      arguments(b)[i]
    end
  end
  return SectorProduct(a)::SectorProduct{T}
end
function arguments_canonicalize(
  ::Type{SectorProduct{T}}, s::SectorProduct{<:NamedTuple}
)::SectorProduct{T} where {T<:NamedTuple}
  b = one(SectorProduct{T})
  a = T(
    map(keys(arguments(b))) do k
      si = get(arguments(s), k, TKS.Trivial())
      if si === TKS.Trivial()
        return getproperty(arguments(b), k)
      else
        return si
      end
    end,
  )

  return SectorProduct(a)::SectorProduct{T}
end
