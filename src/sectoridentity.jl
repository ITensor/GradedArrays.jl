"""
    SectorIdentity{T,S<:Sector} <: AbstractSectorDelta{T,S,2}

Fused 2D structural factor for a single coupled sector. By Schur's lemma, the
structural part of each block in the fused (matricized) basis is the identity
matrix for the irrep. Carries no free data — completely determined by the sector.
The codomain axis is non-dual, the domain axis is dual.
"""
struct SectorIdentity{T, S <: Sector} <: AbstractSectorDelta{T, S, 2}
    sector::S
end
SectorIdentity{T}(s::S) where {T, S <: Sector} = SectorIdentity{T, S}(s)
# The arrows live on the two axes, not the sector, so an oriented key is rejected by `to_sector`.
SectorIdentity{T}(s::OrientedSector) where {T} = SectorIdentity{T}(to_sector(s))

# The fused structural factor is always a coupled-sector matrix: one codomain, one domain leg.
TensorAlgebra.ndims_codomain(::SectorIdentity) = 1
TensorAlgebra.ndims_domain(::SectorIdentity) = 1

Base.@propagate_inbounds function Base.getindex(
        A::SectorIdentity{T}, i::Int, j::Int
    ) where {T}
    @boundscheck checkbounds(A, i, j)
    return ifelse(i == j, one(T), zero(T))
end

function biaxes(A::SectorIdentity)
    return bispace((OrientedSector(A.sector),), (OrientedSector(A.sector),))
end
Base.axes(A::SectorIdentity) = Tuple(biaxes(A))

# Structural inner product: the identity contracts to its dimension, the quantum dimension.
function LinearAlgebra.dot(a::SectorIdentity, b::SectorIdentity)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return length(a.sector)
end

# `p`-norm: the identity has `length(sector)` unit entries (its diagonal), so `norm^p` counts them.
# The single formula also covers `p == Inf` (`count^0 == 1`, the max entry).
function LinearAlgebra.norm(a::SectorIdentity{T}, p::Real = 2) where {T}
    return convert(real(float(T)), length(a.sector)^(1 / p))
end

# The identity structural factor is a matrix, so its trace is defined (unlike the general structural
# deltas): the sector's quantum dimension, the length of the diagonal.
function LinearAlgebra.tr(a::SectorIdentity)
    return diaglength(a)
end

# The stored (nonzero) entries of the identity are its diagonal.
storedlength(a::SectorIdentity) = diaglength(a)

# Only the identity permutation is supported: a transposing `permutedims` would flip the first axis
# to dual, which the fused storage types disallow.
function Base.permutedims(a::SectorIdentity, perm)
    perm == ntuple(identity, ndims(a)) || throw_flips_first_axis(permutedims, a)
    return a
end

Base.conj(a::SectorIdentity) = throw_flips_first_axis(conj, a)
