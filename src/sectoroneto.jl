"""
    SectorOneTo{S<:SectorRange}

Represents one sector's index space — a `SectorRange` (sector label + dual flag) paired
with a multiplicity count. This is the building block for `GradedOneTo`.

Stores a `SectorRange` and a multiplicity. The `isdual` accessor is
derived from the stored `SectorRange`.
"""
struct SectorOneTo{S <: SectorRange} <: AbstractUnitRange{Int}
    sector::S
    multiplicity::Int
end

# Convenience: SectorRange with default multiplicity
SectorOneTo(s::SectorRange) = SectorOneTo(s, 1)

# Primitive accessors
sector(si::SectorOneTo) = si.sector
sector_multiplicity(si::SectorOneTo) = si.multiplicity

# Derived accessors
isdual(si::SectorOneTo) = isdual(sector(si))

# Kronecker factor decomposition:
# SectorOneTo = tensor_product(SectorRange (sector axis), OneTo (data axis))
data(si::SectorOneTo) = Base.OneTo(sector_multiplicity(si))
sectoraxes(si::SectorOneTo) = (sector(si),)
dataaxes(si::SectorOneTo) = (data(si),)

# Generic single-axis accessors (like axes1 = first ∘ axes)
sectoraxes1(a) = first(sectoraxes(a))
dataaxes1(a) = first(dataaxes(a))

# Type-level data axis type (for promote_op in similar)
dataaxistype(::Type{<:SectorOneTo}) = Base.OneTo{Int}

# Duck-typed interface matching GradedOneTo
sectors(si::SectorOneTo) = [sector(si)]
sector_multiplicities(si::SectorOneTo) = [sector_multiplicity(si)]
BlockArrays.blocklength(si::SectorOneTo) = 1
Base.first(::SectorOneTo) = 1
Base.last(si::SectorOneTo) = length(si)
Base.length(si::SectorOneTo) = quantum_dimension(sector(si)) * sector_multiplicity(si)

# sector_type, SymmetryStyle
sector_type(::Type{SectorOneTo{S}}) where {S} = S
SymmetryStyle(::Type{<:SectorOneTo{S}}) where {S} = SymmetryStyle(S)

# dual, flip, flip_dual
dual(si::SectorOneTo) = SectorOneTo(dual(sector(si)), sector_multiplicity(si))
flip(si::SectorOneTo) = SectorOneTo(flip(sector(si)), sector_multiplicity(si))
flip_dual(si::SectorOneTo) = isdual(si) ? flip(si) : si

# Equality and hashing
function Base.isequal(a::SectorOneTo, b::SectorOneTo)
    return isequal(sector(a), sector(b)) &&
        isequal(sector_multiplicity(a), sector_multiplicity(b))
end
Base.:(==)(a::SectorOneTo, b::SectorOneTo) = isequal(a, b)
function Base.hash(si::SectorOneTo, h::UInt)
    return hash(sector(si), hash(sector_multiplicity(si), h))
end

# ========================  sectorrange constructors  ========================

"""
    sectorrange(sector, dim)
    sectorrange(sector, range)

Construct a [`SectorOneTo`](@ref) for the given sector and multiplicity.
"""
sectorrange(s::SectorRange, dim::Integer) = SectorOneTo(s, Int(dim))
sectorrange(s::SectorRange, range::AbstractUnitRange) = sectorrange(s, length(range))
function sectorrange(
        sector::NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}},
        args...
    )
    return sectorrange(to_sector(sector), args...)
end

to_gradedrange(si::SectorOneTo) = gradedrange([sector(si) => sector_multiplicity(si)])

# ========================  BlockSparseArrays interface  ========================

BlockSparseArrays.eachblockaxis(si::SectorOneTo) = [si]

# ========================  tensor_product  ========================

function tensor_product(si::SectorOneTo)
    return isdual(si) ? flip(si) : si
end

function tensor_product(sr1::SectorOneTo, sr2::SectorOneTo)
    return tensor_product(
        combine_styles(SymmetryStyle(sr1), SymmetryStyle(sr2)), sr1, sr2
    )
end

function tensor_product(::AbelianStyle, sr1::SectorOneTo, sr2::SectorOneTo)
    s = tensor_product(sector(flip_dual(sr1)), sector(flip_dual(sr2)))
    return sectorrange(s, sector_multiplicity(sr1) * sector_multiplicity(sr2))
end

function tensor_product(::NotAbelianStyle, sr1::SectorOneTo, sr2::SectorOneTo)
    g = tensor_product(sector(flip_dual(sr1)), sector(flip_dual(sr2)))
    d₁ = sector_multiplicity(sr1)
    d₂ = sector_multiplicity(sr2)
    return gradedrange(
        [
            c => (d₁ * d₂ * d) for (c, d) in zip(sectors(g), sector_multiplicities(g))
        ]
    )
end

# ========================  Show  ========================

function Base.show(io::IO, si::SectorOneTo)
    print(io, "SectorOneTo(")
    show(io, label(sector(si)))
    print(io, ", ", sector_multiplicity(si))
    print(io, ")")
    isdual(si) && print(io, "'")
    return nothing
end
