# The structural factor of a `UniqueSectorArray`: one sector and one arrow per axis, split into
# `NC` codomain and `ND` domain legs. Every element is `one(T)`.
struct UniqueSectorDelta{T, S <: Sector, N, NC, ND} <: AbstractSectorDelta{T, S, N}
    sectors_codomain::NTuple{NC, S}
    isduals_codomain::NTuple{NC, Bool}
    sectors_domain::NTuple{ND, S}
    isduals_domain::NTuple{ND, Bool}
    function UniqueSectorDelta{T, S, N, NC, ND}(
            sectors_codomain::NTuple{NC, S}, isduals_codomain::NTuple{NC, Bool},
            sectors_domain::NTuple{ND, S}, isduals_domain::NTuple{ND, Bool}
        ) where {T, S <: Sector, N, NC, ND}
        NC + ND == N ||
            throw(ArgumentError("codomain ($NC) + domain ($ND) legs must equal N ($N)"))
        return new{T, S, N, NC, ND}(
            sectors_codomain, isduals_codomain, sectors_domain, isduals_domain
        )
    end
end

# `T` cannot be inferred from the sectors, so it must be given.
function UniqueSectorDelta{T}(
        sectors_codomain::NTuple{NC, S}, isduals_codomain::NTuple{NC, Bool},
        sectors_domain::NTuple{ND, S}, isduals_domain::NTuple{ND, Bool}
    ) where {T, S <: Sector, NC, ND}
    return UniqueSectorDelta{T, S, NC + ND, NC, ND}(
        sectors_codomain, isduals_codomain, sectors_domain, isduals_domain
    )
end

function UniqueSectorDelta{T, S, N}(
        sectors_codomain::NTuple{NC, S}, isduals_codomain::NTuple{NC, Bool},
        sectors_domain::NTuple{ND, S}, isduals_domain::NTuple{ND, Bool}
    ) where {T, S <: Sector, N, NC, ND}
    return UniqueSectorDelta{T, S, N, NC, ND}(
        sectors_codomain, isduals_codomain, sectors_domain, isduals_domain
    )
end

function UniqueSectorDelta{T}(
        sectors_codomain::Tuple{Vararg{AbstractOrientedSector}},
        sectors_domain::Tuple{Vararg{AbstractOrientedSector}}
    ) where {T}
    return UniqueSectorDelta{T}(
        splitarrows(sectors_codomain)..., splitarrows(sectors_domain)...
    )
end

# All legs in the codomain. Rank 0 needs `S` given explicitly, as below.
function UniqueSectorDelta{T}(
        sectors::Tuple{AbstractOrientedSector, Vararg{AbstractOrientedSector}}
    ) where {T}
    return UniqueSectorDelta{T}(sectors, ())
end
function UniqueSectorDelta{T, S, N}(sectors::NTuple{N, S}) where {T, S <: Sector, N}
    return UniqueSectorDelta{T, S, N}(sectors, ntuple(Returns(false), Val(N)), (), ())
end

TensorAlgebra.ndims_codomain(d::UniqueSectorDelta) = length(d.sectors_codomain)
TensorAlgebra.ndims_domain(d::UniqueSectorDelta) = length(d.sectors_domain)
TensorAlgebra.has_bipartition(::UniqueSectorDelta) = true

# ========================  AbstractArray interface  ========================

Base.@propagate_inbounds function Base.getindex(
        A::UniqueSectorDelta{T, <:Any, N},
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return one(T)
end

# The domain sectors are stored un-dualized, as for `GradedArray`.
function biaxes(A::UniqueSectorDelta)
    return bispace(
        map(OrientedSector, A.sectors_codomain, A.isduals_codomain),
        map(OrientedSector, A.sectors_domain, A.isduals_domain)
    )
end
Base.axes(A::UniqueSectorDelta) = Tuple(biaxes(A))

function LinearAlgebra.dot(a::UniqueSectorDelta, b::UniqueSectorDelta)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return 1
end

function LinearAlgebra.norm(a::UniqueSectorDelta{T}, p::Real = 2) where {T}
    return oneunit(real(float(T)))
end
