# The index range of one block of a graded axis: a sector, a data length (multiplicity), and an
# arrow.
struct SectorOneTo{S <: Sector} <: AbstractUnitRange{Int}
    sector::S
    datalength::Int
    isdual::Bool
end

SectorOneTo(s::Sector, datalength::Int = 1) = SectorOneTo(s, datalength, false)
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

# Kronecker factors: the sector axis (`structure`) and the data axis (`data`).
data(r::SectorOneTo) = Base.OneTo(datalength(r))
structure(r::SectorOneTo) = OrientedSector(sector(r), isdual(r))
dataaxes(r::SectorOneTo) = (data(r),)

dataaxistype(::Type{<:SectorOneTo}) = Base.OneTo{Int}

# Matches the `GradedOneTo` interface.
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
    return gradedrange([c => (d₁ * d₂ * d) for (c, d) in zip(sectors(g), datalengths(g))])
end

# ========================  Show  ========================

# Print `dual(SectorOneTo(sector, n))`, as for `GradedOneTo`.
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
