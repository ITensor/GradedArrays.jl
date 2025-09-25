# This file defines the interface for type Sector
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of Sector
using TensorProducts: TensorProducts, ⊗
import TensorKitSectors as TKS
using TensorKitSectors: Sector as AbstractSector

"""
    Sector(sector)
    Sector{T}(sector)

Unit range with elements of type `T` that additionally stores a sector to denote the grading.
Equivalent to `Base.OneTo(length(sector))`.
"""
struct Sector{T,I<:AbstractSector} <: AbstractUnitRange{T}
  sector::I
end

Sector{T,I}(a) where {T,I} = Sector{T,I}(convert(I, a)::I)
Sector(a::Sector) = Sector{Int}(a)

sector(r::Sector) = r.sector
isdual(r::Sector) = r.isdual

# ===================================  Base interface  =====================================

Base.length(r::Sector) = quantum_dimension(r)
Base.isless(r1::Sector, r2::Sector) = isless(sector(r1), sector(r2))
Base.isequal(r1::Sector, r2::Sector) = isequal(sector(r1), sector(r2))
Base.:(==)(r1::Sector, r2::Sector) = sector(r1) == sector(r2)

Base.hash(r::Sector, h::UInt) = hash(r.sector, h)

Base.OneTo(r::Sector{T}) where {T} = Base.OneTo(T(length(r))::T)
Base.first(r::Sector) = first(Base.OneTo(r))
Base.last(r::Sector) = last(Base.OneTo(r))

function Base.show(io::IO, r::Sector{T,I}) where {T,I}
  show(io, typeof(r))
  ioc = IOContext(io, :typeinfo => I)
  print(io, '(')
  show(ioc, r.sector)
  print(io, ')')
end

# =================================  Sectors interface  ====================================

trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
  return gradedrange([trivial(sector_type(axis_type)) => 1])  # always returns nondual
end
function trivial(type::Type)
  return error("`trivial` not defined for type $(type).")
end
trivial(::Type{Sector{T,I}}) where {T,I} = Sector{T,I}(one(I))

istrivial(r::Sector) = isone(sector(r))
istrivial(r) = (r == trivial(r))

sector_label(r::Sector) = sector_label(sector(r))
function sector_label(c::AbstractSector)
  return error("method `sector_label` not defined for type $(typeof(c))")
end

quantum_dimension(g::AbstractUnitRange) = length(g)
quantum_dimension(r::Sector) = quantum_dimension(sector(r))
quantum_dimension(s::AbstractSector) = TKS.dim(s)

# convert to range
to_gradedrange(c::Sector) = gradedrange([c => 1])

function nsymbol(s1::Sector, s2::Sector, s3::Sector)
  return TKS.Nsymbol(sector(s1), sector(s2), sector(s3))
end

dual(r1::Sector) = typeof(r1)(TKS.dual(sector(r1)))

# ===============================  Fusion rule interface  ==================================

TKS.FusionStyle(::Type{<:Sector{<:Any,I}}) where {I} = TKS.FusionStyle(I)
TKS.BraidingStyle(::Type{<:Sector{<:Any,I}}) where {I} = TKS.BraidingStyle(I)

abstract type SymmetryStyle end

struct AbelianStyle <: SymmetryStyle end
struct NotAbelianStyle <: SymmetryStyle end

SymmetryStyle(x) = SymmetryStyle(typeof(x))

# default SymmetryStyle to AbelianStyle
# allows for abelian-like slicing style for GradedUnitRange: assume length(::label) = 1
# and preserve labels in any slicing operation
SymmetryStyle(T::Type) = AbelianStyle()
function SymmetryStyle(::Type{T}) where {T<:Sector}
  if TKS.FusionStyle(T) == TKS.UniqueFusion() && TKS.BraidingStyle(T) == TKS.Bosonic()
    return AbelianStyle()
  else
    return NotAbelianStyle()
  end
end
SymmetryStyle(G::Type{<:AbstractUnitRange}) = SymmetryStyle(sector_type(G))

combine_styles(::AbelianStyle, ::AbelianStyle) = AbelianStyle()
combine_styles(::SymmetryStyle, ::SymmetryStyle) = NotAbelianStyle()

# Suggestion:
# function fusion_rule(r1::Sector{T₁,I}, r2::Sector{T₂,I}) where {T₁,T₂,I}
#   T = promote_type(T₁, T₂)
#   a = sector(r1)
#   b = sector(r2)
#   FusionStyle(I) == UniqueFusion() && return Sector{T}(only(TKS.otimes(a, b)))
#   return gradedrange([Sector{T}(c) => TKS.Nsymbol(a, b, c) for c in TKS.otimes(a, b)])
# end

function fusion_rule(c1::Sector, c2::Sector)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::NotAbelianStyle, r1::C, r2::C) where {C<:Sector}
  a = sector(r1)
  b = sector(r2)
  return gradedrange([C(c) => TKS.Nsymbol(a, b, c) for c in TKS.otimes(a, b)])
end

# abelian case: return Sector
function fusion_rule(::AbelianStyle, c1::C, c2::C) where {C<:Sector}
  return only(sectors(fusion_rule(NotAbelianStyle(), c1, c2)))
end

# =============================  TensorProducts interface  =====--==========================

TensorProducts.tensor_product(s::Sector) = s
TensorProducts.tensor_product(c1::Sector, c2::Sector) = fusion_rule(c1, c2)

# ================================  GradedUnitRanges interface  ==================================

sector_type(S::Type{<:Sector}) = S

# =====================================  Sectors ===========================================

const TrivialSector = Sector{Int,TKS.Trivial}
const Z{N} = Sector{Int,TKS.ZNIrrep{N}}
const Z2 = Z{2}

const U1 = Sector{Int,TKS.U1Irrep}
U1(n::Real) = U1(TKS.U1Irrep(n))
sector_label(c::TKS.U1Irrep) = c.charge

const O2 = Sector{Int,TKS.CU1Irrep}
const SU2 = Sector{Int,TKS.SU2Irrep}
const Fib = Sector{Int,TKS.FibonacciAnyon}
const Ising = Sector{Int,TKS.IsingAnyon}
