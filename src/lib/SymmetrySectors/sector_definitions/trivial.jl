#
# Trivial sector
# acts as a trivial sector for any AbstractSector
#

using ...GradedUnitRanges: GradedUnitRanges

# Trivial is special as it does not have a label
struct TrivialSector <: AbstractSector end

SymmetryStyle(::Type{TrivialSector}) = AbelianStyle()

trivial(::Type{TrivialSector}) = TrivialSector()

GradedUnitRanges.dual(::TrivialSector) = TrivialSector()

# TrivialSector acts as trivial on any AbstractSector
function fusion_rule(::NotAbelianStyle, ::TrivialSector, c::AbstractSector)
  return to_gradedrange(c)
end
function fusion_rule(::NotAbelianStyle, c::AbstractSector, ::TrivialSector)
  return to_gradedrange(c)
end
# Fix ambiguity error.
function fusion_rule(::NotAbelianStyle, c::TrivialSector, ::TrivialSector)
  return to_gradedrange(TrivialSector())
end

# abelian case: return Sector
fusion_rule(::AbelianStyle, c::AbstractSector, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::AbstractSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, ::TrivialSector) = TrivialSector()

# any trivial sector equals TrivialSector
Base.:(==)(c::AbstractSector, ::TrivialSector) = istrivial(c)
Base.:(==)(::TrivialSector, c::AbstractSector) = istrivial(c)
Base.:(==)(::TrivialSector, ::TrivialSector) = true

# sorts as trivial for any Sector
Base.isless(c::AbstractSector, ::TrivialSector) = c < trivial(c)
Base.isless(::TrivialSector, c::AbstractSector) = trivial(c) < c
Base.isless(::TrivialSector, ::TrivialSector) = false  # bypass default that calls label
