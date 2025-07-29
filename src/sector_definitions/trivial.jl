#
# Trivial sector
# acts as a trivial sector for any Sector
#

# Trivial is special as it does not have a label
struct TrivialSector <: Sector end

TKS.FusionStyle(::Type{TrivialSector}) = TKS.UniqueFusion()
TKS.BraidingStyle(::Type{TrivialSector}) = TKS.Bosonic()

Base.one(::Type{TrivialSector}) = TrivialSector()

dual(::TrivialSector) = TrivialSector()

# TrivialSector acts as trivial on any Sector
function fusion_rule(::NotAbelianStyle, ::TrivialSector, c::Sector)
  return to_gradedrange(c)
end
function fusion_rule(::NotAbelianStyle, c::Sector, ::TrivialSector)
  return to_gradedrange(c)
end

# abelian case: return Sector
fusion_rule(::AbelianStyle, c::Sector, ::TrivialSector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, c::Sector) = c
fusion_rule(::AbelianStyle, ::TrivialSector, ::TrivialSector) = TrivialSector()

# any trivial sector equals TrivialSector
Base.:(==)(c::Sector, ::TrivialSector) = istrivial(c)
Base.:(==)(::TrivialSector, c::Sector) = istrivial(c)
Base.:(==)(::TrivialSector, ::TrivialSector) = true

# sorts as trivial for any Sector
Base.isless(c::Sector, ::TrivialSector) = c < trivial(c)
Base.isless(::TrivialSector, c::Sector) = trivial(c) < c
Base.isless(::TrivialSector, ::TrivialSector) = false  # bypass default that calls label
