"""
    SectorOneTo{S<:Sector}

One sector's index space: a sector, a data length (multiplicity), and an arrow. This is the
building block for `GradedOneTo`.

The arrow is stored here rather than on the sector, matching `GradedOneTo`. An
[`OrientedSector`](@ref) is built on demand, by `structure`, for the places that want the
sector and its arrow as one value.
"""
struct SectorOneTo{S <: Sector} <: AbstractUnitRange{Int}
    sector::S
    datalength::Int
    isdual::Bool
end

SectorOneTo(s::Sector, datalength::Int = 1) = SectorOneTo(s, datalength, false)
function SectorOneTo(s::Sector, r::Base.OneTo, isdual::Bool = false)
    return SectorOneTo(s, last(r), isdual)
end
# An `OrientedSector` already carries the arrow, so it splits into the two stored fields.
function SectorOneTo(s::OrientedSector, datalength::Int = 1)
    return SectorOneTo(sector(s), datalength, isdual(s))
end
SectorOneTo(s::OrientedSector, r::Base.OneTo) = SectorOneTo(s, last(r))

# Primitive accessors
sector(r::SectorOneTo) = r.sector
TensorAlgebra.isdual(r::SectorOneTo) = r.isdual
datalength(r::SectorOneTo) = r.datalength

# Derived accessors
sectorlength(r::SectorOneTo) = length(sector(r))

# Kronecker factor decomposition:
# SectorOneTo = tensor_product(OrientedSector (sector axis), OneTo (data axis))
data(r::SectorOneTo) = Base.OneTo(datalength(r))
structure(r::SectorOneTo) = OrientedSector(sector(r), isdual(r))
dataaxes(r::SectorOneTo) = (data(r),)

# Type-level data axis type (for promote_op in similar)
dataaxistype(::Type{<:SectorOneTo}) = Base.OneTo{Int}

# Duck-typed interface matching GradedOneTo: `sectors` reports the bare sector, which no
# longer carries an arrow, while `eachstructureaxis` pairs it with this range's arrow.
sectors(r::SectorOneTo) = [sector(r)]
datalengths(r::SectorOneTo) = [datalength(r)]
BlockArrays.blocklength(::SectorOneTo) = 1
Base.first(::SectorOneTo) = 1
Base.last(r::SectorOneTo) = length(r)
Base.length(r::SectorOneTo) = sectorlength(r) * datalength(r)

# sectortype, FusionStyle
sectortype(::Type{SectorOneTo{S}}) where {S} = S
TKS.FusionStyle(r::SectorOneTo) = TKS.FusionStyle(typeof(r))
TKS.FusionStyle(::Type{<:SectorOneTo{S}}) where {S} = TKS.FusionStyle(S)

# dual, flip, flip_dual
TensorAlgebra.dual(r::SectorOneTo) = SectorOneTo(sector(r), datalength(r), !isdual(r))
flip(r::SectorOneTo) =
    SectorOneTo(dual_sector(sector(r)), datalength(r), !isdual(r))
flip_dual(r::SectorOneTo) = isdual(r) ? flip(r) : r

# Equality and hashing
function Base.isequal(a::SectorOneTo, b::SectorOneTo)
    return isequal(sector(a), sector(b)) &&
        isequal(isdual(a), isdual(b)) &&
        isequal(datalength(a), datalength(b))
end
Base.:(==)(a::SectorOneTo, b::SectorOneTo) = isequal(a, b)
function Base.hash(r::SectorOneTo, h::UInt)
    return hash(sector(r), hash(isdual(r), hash(datalength(r), h)))
end

function to_gradedrange(r::SectorOneTo)
    return GradedOneTo([sector(r)], [datalength(r)], isdual(r))
end

# ========================  BlockSparseArrays interface  ========================

eachblockaxis(r::SectorOneTo) = [r]
eachdataaxis(r::SectorOneTo) = [data(r)]
eachstructureaxis(r::SectorOneTo) = [structure(r)]

# ========================  tensor_product  ========================

function tensor_product(r::SectorOneTo)
    return isdual(r) ? flip(r) : r
end

function tensor_product(r1::SectorOneTo, r2::SectorOneTo)
    return tensor_product(
        TKS.FusionStyle(r1) & TKS.FusionStyle(r2), r1, r2
    )
end

function tensor_product(::TKS.UniqueFusion, r1::SectorOneTo, r2::SectorOneTo)
    s = tensor_product(sector(flip_dual(r1)), sector(flip_dual(r2)))
    return SectorOneTo(s, datalength(r1) * datalength(r2))
end

function tensor_product(::TKS.MultipleFusion, r1::SectorOneTo, r2::SectorOneTo)
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

# Factor the `dual` to the outside — `dual(SectorOneTo(sector, n))` — rather
# than decorating the inner sector, mirroring the `GradedOneTo` convention.
function Base.show(io::IO, r::SectorOneTo)
    isdual(r) && print(io, "dual(")
    print(io, "SectorOneTo(")
    show(io, sector(r))
    print(io, ", ", datalength(r))
    print(io, ")")
    isdual(r) && print(io, ")")
    return nothing
end

Base.conj(r::SectorOneTo) = dual(r)
