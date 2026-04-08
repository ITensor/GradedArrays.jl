"""
    SectorUnitRange{I<:TKS.Sector}

Represents one sector's index space — a sector label paired with a multiplicity count
and a dual flag. This is the building block for `GradedUnitRange`.

Stores the raw label, multiplicity, and dual flag as primitives. The `sector` accessor
returns a `SectorRange` on the fly.
"""
struct SectorUnitRange{I <: TKS.Sector} <: AbstractUnitRange{Int}
    label::I
    multiplicity::Int
    isdual::Bool
end
function SectorUnitRange(label::TKS.Sector, multiplicity::Integer)
    return SectorUnitRange(label, Int(multiplicity), false)
end
function SectorUnitRange(label::TKS.Sector)
    return SectorUnitRange(label, 1)
end

# Primitive accessors
label(si::SectorUnitRange) = si.label
sector_multiplicity(si::SectorUnitRange) = si.multiplicity
isdual(si::SectorUnitRange) = si.isdual

# Derived accessors
sector(si::SectorUnitRange) = SectorRange(label(si), isdual(si))

# Duck-typed interface matching GradedUnitRange
labels(si::SectorUnitRange) = [label(si)]
sectors(si::SectorUnitRange) = [SectorRange(label(si), isdual(si))]
sector_multiplicities(si::SectorUnitRange) = [sector_multiplicity(si)]
BlockArrays.blocklength(si::SectorUnitRange) = 1
Base.length(si::SectorUnitRange) = TKS.dim(label(si)) * sector_multiplicity(si)

# sector_type, SymmetryStyle
sector_type(::Type{SectorUnitRange{I}}) where {I} = SectorRange{I}
SymmetryStyle(::Type{<:SectorUnitRange{I}}) where {I} = SymmetryStyle(SectorRange{I})

# dual, flip, flip_dual
dual(si::SectorUnitRange) = SectorUnitRange(label(si), sector_multiplicity(si), !isdual(si))
function flip(si::SectorUnitRange)
    return SectorUnitRange(dual(label(si)), sector_multiplicity(si), !isdual(si))
end
flip_dual(si::SectorUnitRange) = isdual(si) ? flip(si) : si

# Equality and hashing
function Base.isequal(a::SectorUnitRange, b::SectorUnitRange)
    return isequal(label(a), label(b)) &&
        isequal(sector_multiplicity(a), sector_multiplicity(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::SectorUnitRange, b::SectorUnitRange) = isequal(a, b)
function Base.hash(si::SectorUnitRange, h::UInt)
    return hash(label(si), hash(sector_multiplicity(si), hash(isdual(si), h)))
end

# ========================  sectorrange constructors  ========================

"""
    sectorrange(sector, dim)
    sectorrange(sector, range)

Construct a [`SectorUnitRange`](@ref) for the given sector and multiplicity.
"""
sectorrange(s::SectorRange, dim::Integer) = SectorUnitRange(label(s), Int(dim), isdual(s))
sectorrange(s::SectorRange, range::AbstractUnitRange) = sectorrange(s, length(range))
function sectorrange(
        sector::NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}},
        args...
    )
    return sectorrange(to_sector(sector), args...)
end

to_gradedrange(si::SectorUnitRange) = gradedrange([sector(si) => sector_multiplicity(si)])

# ========================  BlockSparseArrays interface  ========================

BlockSparseArrays.eachblockaxis(si::SectorUnitRange) = [si]

# ========================  tensor_product  ========================

function tensor_product(si::SectorUnitRange)
    return isdual(si) ? flip(si) : si
end

function tensor_product(sr1::SectorUnitRange, sr2::SectorUnitRange)
    return tensor_product(
        combine_styles(SymmetryStyle(sr1), SymmetryStyle(sr2)), sr1, sr2
    )
end

function tensor_product(::AbelianStyle, sr1::SectorUnitRange, sr2::SectorUnitRange)
    s = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    return sectorrange(s, sector_multiplicity(sr1) * sector_multiplicity(sr2))
end

function tensor_product(::NotAbelianStyle, sr1::SectorUnitRange, sr2::SectorUnitRange)
    g = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    d₁ = sector_multiplicity(sr1)
    d₂ = sector_multiplicity(sr2)
    return gradedrange(
        [
            c => (d₁ * d₂ * d) for (c, d) in zip(sectors(g), sector_multiplicities(g))
        ]
    )
end

# ========================  Show  ========================

function Base.show(io::IO, si::SectorUnitRange)
    print(io, "SectorUnitRange(")
    show(io, label(si))
    print(io, ", ", sector_multiplicity(si))
    print(io, ")")
    isdual(si) && print(io, "'")
    return nothing
end
