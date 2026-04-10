"""
    SectorOneTo{S<:SectorRange}

Represents one sector's index space — a `SectorRange` (sector label + dual flag) paired
with a data length (multiplicity). This is the building block for `GradedOneTo`.

Stores a `SectorRange` and a data length. The `isdual` accessor is
derived from the stored `SectorRange`.
"""
struct SectorOneTo{S <: SectorRange} <: AbstractUnitRange{Int}
    sector::S
    datalength::Int
end

# Convenience: SectorRange with default data length
SectorOneTo(s::SectorRange) = SectorOneTo(s, 1)

# Primitive accessors
sector(r::SectorOneTo) = r.sector
datalength(r::SectorOneTo) = r.datalength

# Derived accessors
isdual(r::SectorOneTo) = isdual(sector(r))
sectorlength(r::SectorOneTo) = length(sector(r))

# Kronecker factor decomposition:
# SectorOneTo = tensor_product(SectorRange (sector axis), OneTo (data axis))
data(r::SectorOneTo) = Base.OneTo(datalength(r))
sectoraxes(r::SectorOneTo) = (sector(r),)
dataaxes(r::SectorOneTo) = (data(r),)

# Generic single-axis accessors (like axes1 = first ∘ axes)
sectoraxes1(a) = first(sectoraxes(a))
dataaxes1(a) = first(dataaxes(a))

# Type-level data axis type (for promote_op in similar)
dataaxistype(::Type{<:SectorOneTo}) = Base.OneTo{Int}

# Duck-typed interface matching GradedOneTo
sectors(r::SectorOneTo) = [sector(r)]
datalengths(r::SectorOneTo) = [datalength(r)]
BlockArrays.blocklength(::SectorOneTo) = 1
Base.first(::SectorOneTo) = 1
Base.last(r::SectorOneTo) = length(r)
Base.length(r::SectorOneTo) = sectorlength(r) * datalength(r)

# sector_type, SymmetryStyle
sector_type(::Type{SectorOneTo{S}}) where {S} = S
SymmetryStyle(::Type{<:SectorOneTo{S}}) where {S} = SymmetryStyle(S)

# dual, flip, flip_dual
dual(r::SectorOneTo) = SectorOneTo(dual(sector(r)), datalength(r))
flip(r::SectorOneTo) = SectorOneTo(flip(sector(r)), datalength(r))
flip_dual(r::SectorOneTo) = isdual(r) ? flip(r) : r

# Equality and hashing
function Base.isequal(a::SectorOneTo, b::SectorOneTo)
    return isequal(sector(a), sector(b)) &&
        isequal(datalength(a), datalength(b))
end
Base.:(==)(a::SectorOneTo, b::SectorOneTo) = isequal(a, b)
function Base.hash(r::SectorOneTo, h::UInt)
    return hash(sector(r), hash(datalength(r), h))
end

# ========================  sectorrange constructors  ========================

"""
    sectorrange(sector, dim)
    sectorrange(sector, range)

Construct a [`SectorOneTo`](@ref) for the given sector and data length.
"""
sectorrange(s::SectorRange, dim::Integer) = SectorOneTo(s, Int(dim))
sectorrange(s::SectorRange, range::AbstractUnitRange) = sectorrange(s, length(range))
function sectorrange(
        sector::NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}},
        args...
    )
    return sectorrange(to_sector(sector), args...)
end

to_gradedrange(r::SectorOneTo) = gradedrange([sector(r) => datalength(r)])

# ========================  BlockSparseArrays interface  ========================

BlockSparseArrays.eachblockaxis(r::SectorOneTo) = [r]

# ========================  tensor_product  ========================

function tensor_product(r::SectorOneTo)
    return isdual(r) ? flip(r) : r
end

function tensor_product(r1::SectorOneTo, r2::SectorOneTo)
    return tensor_product(
        combine_styles(SymmetryStyle(r1), SymmetryStyle(r2)), r1, r2
    )
end

function tensor_product(::AbelianStyle, r1::SectorOneTo, r2::SectorOneTo)
    s = tensor_product(sector(flip_dual(r1)), sector(flip_dual(r2)))
    return sectorrange(s, datalength(r1) * datalength(r2))
end

function tensor_product(::NotAbelianStyle, r1::SectorOneTo, r2::SectorOneTo)
    g = tensor_product(sector(flip_dual(r1)), sector(flip_dual(r2)))
    d₁ = datalength(r1)
    d₂ = datalength(r2)
    return gradedrange(
        [
            c => (d₁ * d₂ * d) for (c, d) in zip(sectors(g), datalengths(g))
        ]
    )
end

# ========================  Show  ========================

function Base.show(io::IO, r::SectorOneTo)
    print(io, "SectorOneTo(")
    show(io, label(sector(r)))
    print(io, ", ", datalength(r))
    print(io, ")")
    isdual(r) && print(io, "'")
    return nothing
end
