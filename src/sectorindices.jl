"""
    SectorIndices{I<:TKS.Sector}

Represents one sector's index space — a sector label paired with a multiplicity count
and a dual flag. This is the building block for `GradedIndices`.

Stores the raw label, multiplicity, and dual flag as primitives. The `sector` accessor
returns a `SectorRange` on the fly.
"""
struct SectorIndices{I <: TKS.Sector}
    label::I
    multiplicity::Int
    isdual::Bool
end
function SectorIndices(label::TKS.Sector, multiplicity::Integer)
    return SectorIndices(label, Int(multiplicity), false)
end
function SectorIndices(label::TKS.Sector)
    return SectorIndices(label, 1)
end

# Primitive accessors
label(si::SectorIndices) = si.label
sector_multiplicity(si::SectorIndices) = si.multiplicity
isdual(si::SectorIndices) = si.isdual

# Derived accessors
sector(si::SectorIndices) = SectorRange(label(si), isdual(si))

# Duck-typed interface matching GradedIndices
labels(si::SectorIndices) = [label(si)]
sectors(si::SectorIndices) = [SectorRange(label(si), isdual(si))]
sector_multiplicities(si::SectorIndices) = [sector_multiplicity(si)]
BlockArrays.blocklength(si::SectorIndices) = 1
Base.length(si::SectorIndices) = TKS.dim(label(si)) * sector_multiplicity(si)

# sector_type
sector_type(::Type{SectorIndices{I}}) where {I} = SectorRange{I}

# dual and flip
dual(si::SectorIndices) = SectorIndices(label(si), sector_multiplicity(si), !isdual(si))
function flip(si::SectorIndices)
    return SectorIndices(dual(label(si)), sector_multiplicity(si), !isdual(si))
end

# Equality and hashing
function Base.:(==)(a::SectorIndices, b::SectorIndices)
    return label(a) == label(b) &&
        sector_multiplicity(a) == sector_multiplicity(b) &&
        isdual(a) == isdual(b)
end
function Base.hash(si::SectorIndices, h::UInt)
    return hash(label(si), hash(sector_multiplicity(si), hash(isdual(si), h)))
end

# ========================  sectorrange constructors  ========================

"""
    sectorrange(sector, dim)
    sectorrange(sector, range)

Construct a [`SectorIndices`](@ref) for the given sector and multiplicity.
"""
sectorrange(s::SectorRange, dim::Integer) = SectorIndices(label(s), Int(dim), isdual(s))
sectorrange(s::SectorRange, range::AbstractUnitRange) = sectorrange(s, length(range))
function sectorrange(
        sector::NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}},
        args...
    )
    return sectorrange(to_sector(sector), args...)
end

# Show
function Base.show(io::IO, si::SectorIndices)
    print(io, "SectorIndices(")
    show(io, label(si))
    print(io, ", ", sector_multiplicity(si))
    print(io, ")")
    isdual(si) && print(io, "'")
    return nothing
end
