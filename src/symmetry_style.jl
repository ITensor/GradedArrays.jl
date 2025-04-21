# This file defines SymmetryStyle, a trait to distinguish abelian groups, non-abelian groups
# and non-group fusion categories.

abstract type SymmetryStyle end

struct AbelianStyle <: SymmetryStyle end
struct NotAbelianStyle <: SymmetryStyle end

SymmetryStyle(x) = SymmetryStyle(typeof(x))

# default SymmetryStyle to AbelianStyle
# allows for abelian-like slicing style for GradedUnitRange: assume length(::label) = 1
# and preserve labels in any slicing operation
SymmetryStyle(T::Type) = AbelianStyle()
SymmetryStyle(G::Type{<:AbstractUnitRange}) = SymmetryStyle(sector_type(G))

combine_styles(::AbelianStyle, ::AbelianStyle) = AbelianStyle()
combine_styles(::SymmetryStyle, ::SymmetryStyle) = NotAbelianStyle()
