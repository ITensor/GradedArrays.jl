# This file defines SymmetryStyle, a trait to distinguish abelian groups, non-abelian groups
# and non-group fusion categories.

using ..GradedUnitRanges: sector_type

abstract type SymmetryStyle end

struct AbelianStyle <: SymmetryStyle end
struct NotAbelianStyle <: SymmetryStyle end

SymmetryStyle(x) = SymmetryStyle(typeof(x))
SymmetryStyle(T::Type) = error("method `SymmetryStyle` not defined for type $(T)")
SymmetryStyle(G::Type{<:AbstractUnitRange}) = SymmetryStyle(sector_type(G))

combine_styles(::AbelianStyle, ::AbelianStyle) = AbelianStyle()
combine_styles(::SymmetryStyle, ::SymmetryStyle) = NotAbelianStyle()
