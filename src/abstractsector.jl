# This file defines the interface for type Sector
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of Sector
using TensorProducts: TensorProducts, ⊗
import TensorKitSectors as TKS
using TensorKitSectors: Sector as AbstractSector

"""
    SectorRange(sector::AbstractSector)

Unit range with elements of type `Int` that additionally stores a sector to denote the grading.
Equivalent to `Base.OneTo(length(sector))`.
"""
struct SectorRange{I<:AbstractSector} <: AbstractUnitRange{Int}
  sector::I
end

sector(r::SectorRange) = r.sector
sector_type(I::Type{<:SectorRange}) = I

# ===================================  Base interface  =====================================

Base.length(r::SectorRange) = quantum_dimension(r)
Base.isless(r1::SectorRange, r2::SectorRange) = isless(sector(r1), sector(r2))
Base.isequal(r1::SectorRange, r2::SectorRange) = isequal(sector(r1), sector(r2))
Base.:(==)(r1::SectorRange, r2::SectorRange) = sector(r1) == sector(r2)

Base.hash(r::SectorRange, h::UInt) = hash(r.sector, h)

Base.OneTo(r::SectorRange) = Base.OneTo(length(r))
Base.first(r::SectorRange) = first(Base.OneTo(r))
Base.last(r::SectorRange) = last(Base.OneTo(r))

function Base.show(io::IO, r::SectorRange{I}) where {I}
  show(io, typeof(r))
  print(io, '(')
  l = sector_label(sector(r))
  isnothing(l) || show(io, l)
  print(io, ')')
  return nothing
end

# =================================  Sectors interface  ====================================

trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
  return gradedrange([trivial(sector_type(axis_type)) => 1])  # always returns nondual
end
function trivial(type::Type)
  return error("`trivial` not defined for type $(type).")
end
trivial(::Type{SectorRange{I}}) where {I} = SectorRange{I}(one(I))

istrivial(r::SectorRange) = isone(sector(r))
istrivial(r) = (r == trivial(r))

sector_label(r::SectorRange) = sector_label(sector(r))
function sector_label(c::AbstractSector)
  return map(fieldnames(typeof(c))) do f
    return getfield(c, f)
  end
  return c
end

quantum_dimension(g::AbstractUnitRange) = length(g)
quantum_dimension(r::SectorRange) = quantum_dimension(sector(r))
quantum_dimension(s::AbstractSector) = TKS.dim(s)

# convert to range
to_gradedrange(c::SectorRange) = gradedrange([c => 1])
to_gradedrange(c::AbstractSector) = to_gradedrange(SectorRange(c))

function nsymbol(s1::SectorRange, s2::SectorRange, s3::SectorRange)
  return TKS.Nsymbol(sector(s1), sector(s2), sector(s3))
end

dual(r1::SectorRange) = typeof(r1)(TKS.dual(sector(r1)))

# ===============================  Fusion rule interface  ==================================

TKS.FusionStyle(::Type{SectorRange{I}}) where {I} = TKS.FusionStyle(I)
TKS.BraidingStyle(::Type{SectorRange{I}}) where {I} = TKS.BraidingStyle(I)

abstract type SymmetryStyle end

struct AbelianStyle <: SymmetryStyle end
struct NotAbelianStyle <: SymmetryStyle end

SymmetryStyle(x) = SymmetryStyle(typeof(x))

# default SymmetryStyle to AbelianStyle
# allows for abelian-like slicing style for GradedUnitRange: assume length(::label) = 1
# and preserve labels in any slicing operation
SymmetryStyle(T::Type) = AbelianStyle()
function SymmetryStyle(::Type{T}) where {T<:SectorRange}
  if TKS.FusionStyle(T) === TKS.UniqueFusion() && TKS.BraidingStyle(T) === TKS.Bosonic()
    return AbelianStyle()
  else
    return NotAbelianStyle()
  end
end
SymmetryStyle(G::Type{<:AbstractUnitRange}) = SymmetryStyle(sector_type(G))

combine_styles(::AbelianStyle, ::AbelianStyle) = AbelianStyle()
combine_styles(::SymmetryStyle, ::SymmetryStyle) = NotAbelianStyle()

function fusion_rule(r1::C, r2::C) where {C<:SectorRange}
  a = sector(r1)
  b = sector(r2)
  TKS.FusionStyle(C) == TKS.UniqueFusion() && return SectorRange(only(TKS.otimes(a, b)))
  return gradedrange([SectorRange(c) => TKS.Nsymbol(a, b, c) for c in TKS.otimes(a, b)])
end
fusion_rule(r1::SectorRange, r2::SectorRange) = fusion_rule(promote(r1, r2)...)

# =============================  TensorProducts interface  =====--==========================

TensorProducts.tensor_product(s::SectorRange) = s
TensorProducts.tensor_product(c1::SectorRange, c2::SectorRange) = fusion_rule(c1, c2)

# =====================================  Sectors ===========================================

const TrivialSector = SectorRange{TKS.Trivial}
TrivialSector() = TrivialSector(TKS.Trivial())
sector_label(::TKS.Trivial) = nothing

# use promotion to handle trivial sectors in tensor products
Base.promote_rule(::Type{TrivialSector}, ::Type{T}) where {T<:SectorRange} = T
Base.convert(::Type{T}, ::TrivialSector) where {T<:SectorRange} = trivial(T)

const Z{N} = SectorRange{TKS.ZNIrrep{N}}
sector_label(c::TKS.ZNIrrep) = c.n
const Z2 = Z{2}

const U1 = SectorRange{TKS.U1Irrep}
sector_label(c::TKS.U1Irrep) = c.charge

const O2 = SectorRange{TKS.CU1Irrep}
function O2(l::Real)
  j = max(l, zero(l))
  s = if l == 0
    0
  elseif l == -1
    1
  else
    2
  end
  return O2(TKS.CU1Irrep(j, s))
end
function sector_label(c::TKS.CU1Irrep)
  return if c.s == 0
    c.j
  elseif c.s == 1
    oftype(c.j, -1)
  else
    c.j
  end
end

const SU2 = SectorRange{TKS.SU2Irrep}
sector_label(c::TKS.SU2Irrep) = c.j

const Fib = SectorRange{TKS.FibonacciAnyon}
function Fib(s::AbstractString)
  s == "1" && return Fib(:I)
  s == "τ" && return Fib(:τ)
  throw(ArgumentError("Unrecognized input `$s`"))
end
sector_label(c::TKS.FibonacciAnyon) = c.isone ? "1" : "τ"

const Ising = SectorRange{TKS.IsingAnyon}
function Ising(s::AbstractString)
  s in ("1", "σ", "ψ") || throw(ArgumentError("Unrecognized input `$s`"))
  sym = s == "1" ? :I : Symbol(s)
  return Ising(sym)
end
sector_label(c::TKS.IsingAnyon) = Symbol(c.s) == :I ? "1" : String(c.s)
