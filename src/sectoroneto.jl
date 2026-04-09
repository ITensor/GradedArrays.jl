"""
    SectorOneTo{I<:TKS.Sector}

Represents one sector's index space — a sector label paired with a multiplicity count
and a dual flag. This is the building block for `GradedOneTo`.

Stores the raw label, multiplicity, and dual flag as primitives. The `sector` accessor
returns a `SectorRange` on the fly.
"""
struct SectorOneTo{I <: TKS.Sector} <: AbstractUnitRange{Int}
    label::I
    multiplicity::Int
    isdual::Bool
end
function SectorOneTo(label::TKS.Sector, multiplicity::Integer)
    return SectorOneTo(label, Int(multiplicity), false)
end
function SectorOneTo(label::TKS.Sector)
    return SectorOneTo(label, 1)
end

# Primitive accessors
label(si::SectorOneTo) = si.label
sector_multiplicity(si::SectorOneTo) = si.multiplicity
isdual(si::SectorOneTo) = si.isdual

# Derived accessors
sector(si::SectorOneTo) = SectorRange(label(si), isdual(si))

# Duck-typed interface matching GradedOneTo
labels(si::SectorOneTo) = [label(si)]
sectors(si::SectorOneTo) = [SectorRange(label(si), isdual(si))]
sector_multiplicities(si::SectorOneTo) = [sector_multiplicity(si)]
BlockArrays.blocklength(si::SectorOneTo) = 1
Base.length(si::SectorOneTo) = quantum_dimension(sector(si)) * sector_multiplicity(si)

# sector_type, SymmetryStyle
sector_type(::Type{SectorOneTo{I}}) where {I} = SectorRange{I}
SymmetryStyle(::Type{<:SectorOneTo{I}}) where {I} = SymmetryStyle(SectorRange{I})

# dual, flip, flip_dual
dual(si::SectorOneTo) = SectorOneTo(label(si), sector_multiplicity(si), !isdual(si))
function flip(si::SectorOneTo)
    return SectorOneTo(dual(label(si)), sector_multiplicity(si), !isdual(si))
end
flip_dual(si::SectorOneTo) = isdual(si) ? flip(si) : si

# Equality and hashing
function Base.isequal(a::SectorOneTo, b::SectorOneTo)
    return isequal(label(a), label(b)) &&
        isequal(sector_multiplicity(a), sector_multiplicity(b)) &&
        isequal(isdual(a), isdual(b))
end
Base.:(==)(a::SectorOneTo, b::SectorOneTo) = isequal(a, b)
function Base.hash(si::SectorOneTo, h::UInt)
    return hash(label(si), hash(sector_multiplicity(si), hash(isdual(si), h)))
end

# ========================  sectorrange constructors  ========================

"""
    sectorrange(sector, dim)
    sectorrange(sector, range)

Construct a [`SectorOneTo`](@ref) for the given sector and multiplicity.
"""
sectorrange(s::SectorRange, dim::Integer) = SectorOneTo(label(s), Int(dim), isdual(s))
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
    s = sector(flip_dual(sr1)) ⊗ sector(flip_dual(sr2))
    return sectorrange(s, sector_multiplicity(sr1) * sector_multiplicity(sr2))
end

function tensor_product(::NotAbelianStyle, sr1::SectorOneTo, sr2::SectorOneTo)
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

function Base.show(io::IO, si::SectorOneTo)
    print(io, "SectorOneTo(")
    show(io, label(si))
    print(io, ", ", sector_multiplicity(si))
    print(io, ")")
    isdual(si) && print(io, "'")
    return nothing
end
