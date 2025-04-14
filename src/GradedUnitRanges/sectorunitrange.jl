
using BlockArrays: BlockArrays

# This implementation contains the "full range"
# it does not check that such a range is consistent with the sector quantum_dimension
# when sliced directly, the label is dropped
# when sliced between multiplicities with sr[(:,1:1)], it returns another SectorUnitRange
# TBD impose some compatibility constraints between range and quantum_dimension?
struct SectorUnitRange{T,Sector,Range<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
  nondual_sector::Sector
  full_range::Range
  isdual::Bool

  function SectorUnitRange(s, r, b)
    return new{eltype(r),typeof(s),typeof(r)}(s, r, b)
  end
end

const SectorOneTo{T,Sector,Range} = SectorUnitRange{T,Sector,Base.OneTo{T}}

#
# Constructors
#

to_sector(x) = x

# sectorunitrange(SU2(1), 2:5)
function sectorunitrange(s, r::AbstractUnitRange, b::Bool=false)
  SectorUnitRange(to_sector(s), r, b)
end

# sectorunitrange(SU2(1), 1)
function sectorunitrange(s, m::Integer, b::Bool=false)
  return sectorunitrange(s, Base.oneto(m * length(s)), b)
end

# sectorunitrange(SU2(1) => 1)
function sectorunitrange(p::Pair, b::Bool=false)
  return sectorunitrange(first(p), last(p), b)
end

#
# accessors
#
nondual_sector(sr::SectorUnitRange) = sr.nondual_sector
full_range(sr::SectorUnitRange) = sr.full_range
isdual(sr::SectorUnitRange) = sr.isdual

#
# Base interface
#
Base.first(sr::SectorUnitRange) = first(full_range(sr))

Base.iterate(sr::SectorUnitRange) = iterate(full_range(sr))
Base.iterate(sr::SectorUnitRange, i::Integer) = iterate(full_range(sr), i)

Base.length(sr::SectorUnitRange) = length(full_range(sr))

Base.last(sr::SectorUnitRange) = last(full_range(sr))

# slicing
Base.getindex(sr::SectorUnitRange, i::Integer) = full_range(sr)[i]
function Base.getindex(sr::SectorUnitRange, r::AbstractUnitRange{T}) where {T<:Integer}
  return full_range(sr)[r]
end

# TODO replace (:,x) indexing with kronecker(:, x)
Base.getindex(sr::SectorUnitRange, t::Tuple{Colon,<:Integer}) = sr[(:, last(t):last(t))]
function Base.getindex(sr::SectorUnitRange, t::Tuple{Colon,<:AbstractUnitRange})
  r = last(t)
  new_range =
    ((first(r) - 1) * length(nondual_sector(sr)) + 1):(last(r) * length(nondual_sector(sr)))
  return sectorunitrange(nondual_sector(sr), full_range(sr)[new_range], isdual(sr))
end

function Base.show(io::IO, sr::SectorUnitRange)
  print(io, nameof(typeof(sr)), " ")
  if isdual(sr)
    print(io, "dual(", nondual_sector(sr), ")")
  else
    print(io, nondual_sector(sr))
  end
  return print(io, " => ", full_range(sr))
end

#
# BlockArrays interface
#

#
# GradedUnitRanges interface
#
function blocklabels(sr::SectorUnitRange)
  return isdual(sr) ? [dual(nondual_sector(sr))] : [nondual_sector(sr)]
end

# TBD error for non-integer?
sector_mulitplicities(sr::SectorUnitRange) = length(sr) ÷ length(nondual_sector(sr))

function dual(sr::SectorUnitRange)
  return sectorunitrange(nondual_sector(sr), full_range(sr), !isdual(sr))
end

function flip(sr::SectorUnitRange)
  return sectorunitrange(dual(nondual_sector(sr)), full_range(sr), !isdual(sr))
end

function map_blocklabels(f, sr::SectorUnitRange)
  return sectorunitrange(f(nondual_sector(sr)), full_range(sr), isdual(sr))
end

sector_type(::Type{<:SectorUnitRange{T,Sector}}) where {T,Sector} = Sector
