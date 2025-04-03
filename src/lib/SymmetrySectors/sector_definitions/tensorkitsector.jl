# Generic wrapper
# ---------------
struct TensorKitSector{I<:TensorKitSectors.Sector} <: AbstractSector
  sector::I
end

function SymmetryStyle(::Type{TensorKitSector{I}}) where {I}
  if TensorKitSectors.FusionStyle(I) == TensorKitSectors.UniqueFusion()
    return AbelianStyle()
  else
    return NotAbelianStyle()
  end
end

trivial(::Type{TensorKitSector{I}}) where {I} = TensorKitSector(one(I))
GradedUnitRanges.dual(s::TensorKitSector) = TensorKitSector(TensorKitSectors.dual(s.sector))

quantum_dimension(x::TensorKitSector) = TensorKitSectors.dim(x.sector)

function nsymbol(s1::I, s2::I, s3::I) where {I<:TensorKitSector}
  return TensorKitSectors.Nsymbol(s1.sector, s2.sector, s3.sector)
end

function fusion_rule(::AbelianStyle, c1::I, c2::I) where {I<:TensorKitSector}
  return TensorKitSector(only(TensorKitSectors.otimes(c1.sector, c2.sector)))
end
function fusion_rule(::NotAbelianStyle, c1::I, c2::I) where {I<:TensorKitSector}
  return gradedrange([
    TensorKitSector(c) => nsymbol(c1, c2, TensorKitSector(c)) for
    c in TensorKitSectors.otimes(c1.sector, c2.sector)
  ])
end

Base.:(==)(s1::TensorKitSector, s2::TensorKitSector) = s1.sector == s2.sector
Base.isless(s1::TensorKitSector, s2::TensorKitSector) = isless(s1.sector, s2.sector)

# Specific implementations
# ------------------------
const U1 = TensorKitSector{TensorKitSectors.U1Irrep}
sector_label(s::U1) = s.sector.charge
Base.isless(s1::U1, s2::U1) = s1.sector.charge < s2.sector.charge

const Z{N} = TensorKitSector{TensorKitSectors.ZNIrrep{N}}
sector_label(s::Z) = s.sector.n

const O2 = TensorKitSector{TensorKitSectors.CU1Irrep}
sector_label(s::O2) = s.sector.s == 1 ? Half(-1) : Half(s.sector.j)
function O2(s::Int)
  return s == -1 ? O2(TensorKitSectors.CU1Irrep(0, 1)) : O2(TensorKitSectors.CU1Irrep(s))
end

const SU2 = TensorKitSector{TensorKitSectors.SU2Irrep}
sector_label(s::SU2) = s.sector.j

const Fib = TensorKitSector{TensorKitSectors.FibonacciAnyon}
Fib(s::AbstractString) = Fib(
  if s == "1"
    :I
  elseif s == "τ"
    :τ
  else
    error()
  end,
)

const Ising = TensorKitSector{TensorKitSectors.IsingAnyon}
function sector_label(s::Ising)
  label = s.sector.s
  return if label === :I
    Half(0)
  elseif label === :σ
    Half(1//2)
  else
    Half(1)
  end
end
Ising(s::AbstractString) = Ising(
  if s == "1"
    :I
  elseif s == "σ"
    :σ
  elseif s == "ψ"
    :ψ
  else
    error()
  end,
)
Ising(s::Number) = Ising(
  if s == 0
    :I
  elseif s == 1//2
    :σ
  elseif s == 1
    :ψ
  else
    error()
  end,
)
