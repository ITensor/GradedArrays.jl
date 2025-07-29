# This file defines the interface for type Sector
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of Sector
using TensorProducts: TensorProducts, ⊗
import TensorKitSectors as TKS
using TensorKitSectors: Sector

# ===================================  Base interface  =====================================
function Base.isless(c1::C, c2::C) where {C<:Sector}
  return isless(sector_label(c1), sector_label(c2))
end

Base.length(s::Sector) = quantum_dimension(s)

# =================================  Sectors interface  ====================================
trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
  return gradedrange([trivial(sector_type(axis_type)) => 1])  # always returns nondual
end
function trivial(type::Type)
  return error("`trivial` not defined for type $(type).")
end

istrivial(c::Sector) = (c == trivial(c))

function sector_label(c::Sector)
  return error("method `sector_label` not defined for type $(typeof(c))")
end

quantum_dimension(g::AbstractUnitRange) = length(g)
quantum_dimension(s::Sector) = quantum_dimension(SymmetryStyle(s), s)

function quantum_dimension(::NotAbelianStyle, c::Sector)
  return error("method `quantum_dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianStyle, ::Sector) = 1

# convert to range
to_gradedrange(c::Sector) = gradedrange([c => 1])
to_gradedrange(sr::SectorUnitRange) = mortar_axis([sr])
to_gradedrange(g::AbstractGradedUnitRange) = g

function nsymbol(s1::Sector, s2::Sector, s3::Sector)
  full_space = to_gradedrange(s1 ⊗ s2)
  i = findfirst(==(s3), sectors(full_space))
  isnothing(i) && return 0
  return sector_multiplicities(full_space)[i]
end

# ===============================  Fusion rule interface  ==================================
function fusion_rule(c1::Sector, c2::Sector)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::NotAbelianStyle, c1::C, c2::C) where {C<:Sector}
  sector_degen_pairs = label_fusion_rule(C, sector_label(c1), sector_label(c2))
  return gradedrange(sector_degen_pairs)
end

# abelian case: return Sector
function fusion_rule(::AbelianStyle, c1::C, c2::C) where {C<:Sector}
  return only(sectors(fusion_rule(NotAbelianStyle(), c1, c2)))
end

function label_fusion_rule(sector_type::Type{<:Sector}, l1, l2)
  return [abelian_label_fusion_rule(sector_type, l1, l2) => 1]
end

# =============================  TensorProducts interface  =====--==========================
TensorProducts.tensor_product(s::Sector) = s

function TensorProducts.tensor_product(c1::Sector, c2::Sector)
  return fusion_rule(c1, c2)
end

# ================================  GradedUnitRanges interface  ==================================
sector_type(S::Type{<:Sector}) = S

function findfirstblock(g::AbstractGradedUnitRange, s::Sector)
  return findfirstblock_sector(g::AbstractGradedUnitRange, s)
end
