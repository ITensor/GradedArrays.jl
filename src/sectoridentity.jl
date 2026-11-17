# The structural factor of a `FusedSectorMatrix`, which by Schur's lemma is the identity matrix
# of the irrep. The codomain axis is non-dual and the domain axis is dual.
struct SectorIdentity{T, S <: Sector} <: AbstractSectorDelta{T, S, 2}
    sector::S
end
SectorIdentity{T}(s::S) where {T, S <: Sector} = SectorIdentity{T, S}(s)

sector(a::SectorIdentity) = a.sector

TensorAlgebra.ndims_codomain(::SectorIdentity) = 1
TensorAlgebra.ndims_domain(::SectorIdentity) = 1
TensorAlgebra.has_bipartition(::SectorIdentity) = true

Base.@propagate_inbounds function Base.getindex(
        A::SectorIdentity{T}, i::Int, j::Int
    ) where {T}
    @boundscheck checkbounds(A, i, j)
    return ifelse(i == j, one(T), zero(T))
end

function biaxes(A::SectorIdentity)
    return bispace((OrientedSector(sector(A)),), (OrientedSector(sector(A)),))
end
Base.axes(A::SectorIdentity) = Tuple(biaxes(A))

# The quantum dimension.
function LinearAlgebra.dot(a::SectorIdentity, b::SectorIdentity)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return length(sector(a))
end

# `length(sector)` unit entries. Also correct for `p == Inf`.
function LinearAlgebra.norm(a::SectorIdentity{T}, p::Real = 2) where {T}
    return convert(real(float(T)), length(sector(a))^(1 / p))
end

function LinearAlgebra.tr(a::SectorIdentity)
    return diaglength(a)
end

storedlength(a::SectorIdentity) = diaglength(a)

# A transpose would make the first axis dual, which the fused storage types disallow.
function Base.permutedims(a::SectorIdentity, perm)
    perm == ntuple(identity, ndims(a)) || throw_flips_first_axis(permutedims, a)
    return a
end

Base.conj(a::SectorIdentity) = throw_flips_first_axis(conj, a)
