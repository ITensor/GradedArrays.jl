"""
    UniqueSectorDelta{T,S<:SectorRange,N,NC,ND} <: AbstractSectorDelta{T,S,N}

Unfused N-D structural tensor for abelian symmetries. Stores one `SectorRange` per axis,
split into `NC` codomain legs and `ND` domain legs (`NC + ND == N`); the all-codomain case
(`NC == N`) is the block an `AbelianGradedArray` yields. For abelian symmetries, every element
equals `one(T)` (the Kronecker delta selection rule).
"""
struct UniqueSectorDelta{T, S <: SectorRange, N, NC, ND} <: AbstractSectorDelta{T, S, N}
    sectors_codomain::NTuple{NC, S}
    sectors_domain::NTuple{ND, S}
    function UniqueSectorDelta{T, S, N, NC, ND}(
            sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
        ) where {T, S <: SectorRange, N, NC, ND}
        NC + ND == N ||
            throw(ArgumentError("codomain ($NC) + domain ($ND) legs must equal N ($N)"))
        return new{T, S, N, NC, ND}(sectors_codomain, sectors_domain)
    end
end

# `T` explicit, everything else inferred from the two tuples. Unlike the data-carrying array (whose
# `T` comes from the data), the delta's `T` is independent of the sectors, so it must be given.
function UniqueSectorDelta{T}(
        sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
    ) where {T, S <: SectorRange, NC, ND}
    return UniqueSectorDelta{T, S, NC + ND, NC, ND}(sectors_codomain, sectors_domain)
end

# `N` explicit, `NC`/`ND` inferred from the two tuples, so the all-codomain conveniences below spell
# only the split-free prefix.
function UniqueSectorDelta{T, S, N}(
        sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
    ) where {T, S <: SectorRange, N, NC, ND}
    return UniqueSectorDelta{T, S, N, NC, ND}(sectors_codomain, sectors_domain)
end

# Convenience: all-codomain delta, inferring N and S from a flat sector tuple. Requires at
# least one sector: the sector type of a rank-0 delta cannot be inferred from an empty tuple,
# so a rank-0 delta is built through the fully-parameterized constructor with an explicit `S`.
function UniqueSectorDelta{T}(sectors::Tuple{SectorRange, Vararg{SectorRange}}) where {T}
    return UniqueSectorDelta{T}(sectors, ())
end
# All-codomain delta from a flat tuple with `S`/`N` given (covers the rank-0 case, whose empty
# tuple carries no `S`).
function UniqueSectorDelta{T, S, N}(sectors::NTuple{N, S}) where {T, S <: SectorRange, N}
    return UniqueSectorDelta{T, S, N}(sectors, ())
end

# Codomain/domain leg counts, used by the bend-phase bookkeeping in `bipermutedimsopadd!`.
ndims_codomain(d::UniqueSectorDelta) = length(d.sectors_codomain)
ndims_domain(d::UniqueSectorDelta) = length(d.sectors_domain)

# ========================  AbstractArray interface  ========================

Base.@propagate_inbounds function Base.getindex(
        A::UniqueSectorDelta{T, <:Any, N},
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return one(T)
end

# The domain sectors are stored codomain-facing (un-dualed), matching how `FusionArray` stores its
# `axes_domain`; `axes` returns a `BiTuple` whose domain half is dualized, so a domain leg reads as a
# dual external axis and the codomain/domain split rides along.
Base.axes(A::UniqueSectorDelta) = BiTuple(A.sectors_codomain, map(conj, A.sectors_domain))

# Structural inner product: an abelian delta has a single allowed (unique-fusion) unit entry.
function LinearAlgebra.dot(a::UniqueSectorDelta, b::UniqueSectorDelta)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return 1
end

# `p`-norm: a single unit entry, so the norm is `1` for every `p` (including `Inf`), like its `dot`.
function LinearAlgebra.norm(a::UniqueSectorDelta{T}, p::Real = 2) where {T}
    return oneunit(real(float(T)))
end

# ========================  Accessors  ========================

sectoraxes(x, d::Int) = sectoraxes(x)[d]
