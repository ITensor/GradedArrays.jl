# The structural factor of a `FusedSectorVector`: the all-ones vector whose length is the quantum
# dimension, which is the diagonal of a `SectorIdentity`. The axis is non-dual.
struct SectorOnesVector{T, S <: Sector} <: AbstractSectorDelta{T, S, 1}
    sector::S
end
SectorOnesVector{T}(s::S) where {T, S <: Sector} = SectorOnesVector{T, S}(s)

sector(a::SectorOnesVector) = a.sector

TensorAlgebra.ndims_codomain(::SectorOnesVector) = 1
TensorAlgebra.ndims_domain(::SectorOnesVector) = 0
TensorAlgebra.has_bipartition(::SectorOnesVector) = true

Base.@propagate_inbounds function Base.getindex(A::SectorOnesVector{T}, i::Int) where {T}
    @boundscheck checkbounds(A, i)
    return one(T)
end

biaxes(A::SectorOnesVector) = bispace((OrientedSector(sector(A)),), ())
Base.axes(A::SectorOnesVector) = Tuple(biaxes(A))

# The quantum dimension.
function LinearAlgebra.dot(a::SectorOnesVector, b::SectorOnesVector)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return length(sector(a))
end

# `length(sector)` unit entries. Also correct for `p == Inf`.
function LinearAlgebra.norm(a::SectorOnesVector{T}, p::Real = 2) where {T}
    return convert(real(float(T)), length(sector(a))^(1 / p))
end

Base.permutedims(a::SectorOnesVector, perm) = a

Base.conj(a::SectorOnesVector) = throw_flips_first_axis(conj, a)
