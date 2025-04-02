module GradedUnitRangesSymmetrySectorsExt

using GradedUnitRanges: GradedUnitRanges
using SymmetrySectors: SectorProduct

GradedUnitRanges.to_sector(nt::NamedTuple) = SectorProduct(nt)

end
