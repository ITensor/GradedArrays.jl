# This file defines the abstract type AbstractSector
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of AbstractSector

using BlockArrays: blocklengths
using ..GradedUnitRanges: GradedUnitRanges, blocklabels, gradedrange, sector_multiplicity
using TensorProducts: TensorProducts, ⊗, tensor_product

abstract type AbstractSector end

# ===================================  Base interface  =====================================
function Base.isless(c1::C, c2::C) where {C<:AbstractSector}
  return isless(sector_label(c1), sector_label(c2))
end

Base.length(s::AbstractSector) = quantum_dimension(s)

# =================================  Sectors interface  ====================================
trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
  return gradedrange([trivial(sector_type(axis_type)) => 1])  # always returns nondual
end
function trivial(type::Type)
  return error("`trivial` not defined for type $(type).")
end

istrivial(c::AbstractSector) = (c == trivial(c))

function sector_label(c::AbstractSector)
  return error("method `sector_label` not defined for type $(typeof(c))")
end

quantum_dimension(g::AbstractUnitRange) = length(g)
quantum_dimension(s::AbstractSector) = quantum_dimension(SymmetryStyle(s), s)

function quantum_dimension(::NotAbelianStyle, c::AbstractSector)
  return error("method `quantum_dimension` not defined for type $(typeof(c))")
end

quantum_dimension(::AbelianStyle, ::AbstractSector) = 1

# convert to range
to_gradedrange(c::AbstractSector) = gradedrange([c => 1])
to_gradedrange(g::AbstractUnitRange) = g

function nsymbol(s1::AbstractSector, s2::AbstractSector, s3::AbstractSector)
  full_space = to_gradedrange(s1 ⊗ s2)
  i = findfirst(==(s3), blocklabels(full_space))
  isnothing(i) && return 0
  return sector_multiplicity(full_space)[i]
end

# ===============================  Fusion rule interface  ==================================
function fusion_rule(c1::AbstractSector, c2::AbstractSector)
  return fusion_rule(combine_styles(SymmetryStyle(c1), SymmetryStyle(c2)), c1, c2)
end

function fusion_rule(::NotAbelianStyle, c1::C, c2::C) where {C<:AbstractSector}
  sector_degen_pairs = label_fusion_rule(C, sector_label(c1), sector_label(c2))
  return gradedrange(sector_degen_pairs)
end

# abelian case: return Sector
function fusion_rule(::AbelianStyle, c1::C, c2::C) where {C<:AbstractSector}
  return only(blocklabels(fusion_rule(NotAbelianStyle(), c1, c2)))
end

function label_fusion_rule(sector_type::Type{<:AbstractSector}, l1, l2)
  return [abelian_label_fusion_rule(sector_type, l1, l2) => 1]
end

# =============================  TensorProducts interface  =====--==========================
TensorProducts.tensor_product(s::AbstractSector) = s

function TensorProducts.tensor_product(c1::AbstractSector, c2::AbstractSector)
  return fusion_rule(c1, c2)
end

# allow to fuse a Sector with a GradedUnitRange
function TensorProducts.tensor_product(c::AbstractSector, g::AbstractUnitRange)
  return tensor_product(to_gradedrange(c), g)
end

function TensorProducts.tensor_product(g::AbstractUnitRange, c::AbstractSector)
  return tensor_product(g, to_gradedrange(c))
end

# ================================  GradedUnitRanges interface  ==================================
GradedUnitRanges.sector_type(S::Type{<:AbstractSector}) = S
