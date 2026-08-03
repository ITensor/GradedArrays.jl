"""
    AbelianSectorDelta{T,S<:SectorRange,N,NC,ND} <: AbstractSectorDelta{T, S, N}

Unfused N-D structural tensor for abelian symmetries. Stores one `SectorRange` per axis,
split into `NC` codomain legs and `ND` domain legs (`NC + ND == N`); the all-codomain case
(`NC == N`) is the block an `AbelianGradedArray` yields. For abelian symmetries, every element
equals `one(T)` (the Kronecker delta selection rule).
"""
struct AbelianSectorDelta{T, S <: SectorRange, N, NC, ND} <: AbstractSectorDelta{T, S, N}
    sectors_codomain::NTuple{NC, S}
    sectors_domain::NTuple{ND, S}
    function AbelianSectorDelta{T, S, N, NC, ND}(
            sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
        ) where {T, S <: SectorRange, N, NC, ND}
        NC + ND == N ||
            throw(ArgumentError("codomain ($NC) + domain ($ND) legs must equal N ($N)"))
        return new{T, S, N, NC, ND}(sectors_codomain, sectors_domain)
    end
end

# Convenience: all-codomain delta, inferring N and S from a flat sector tuple. Requires at
# least one sector: the sector type of a rank-0 delta cannot be inferred from an empty tuple,
# so a rank-0 delta is built through the fully-parameterized constructor with an explicit `S`.
function AbelianSectorDelta{T}(sectors::Tuple{SectorRange, Vararg{SectorRange}}) where {T}
    N = length(sectors)
    return AbelianSectorDelta{T, eltype(sectors), N, N, 0}(sectors, ())
end
# All-codomain delta from a flat tuple with `S`/`N` given (covers the rank-0 case, whose empty
# tuple carries no `S`).
function AbelianSectorDelta{T, S, N}(sectors::NTuple{N, S}) where {T, S <: SectorRange, N}
    return AbelianSectorDelta{T, S, N, N, 0}(sectors, ())
end

# ========================  AbstractArray interface  ========================

Base.@propagate_inbounds function Base.getindex(
        A::AbelianSectorDelta{T, <:Any, N},
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return one(T)
end

# The stored codomain/domain split is not yet reflected in `axes`, which stays the flat
# combined leg tuple so `sectoraxes`/`axes` of the sector array are unchanged (the BiTuple
# axes representation is a separate follow-up).
Base.axes(A::AbelianSectorDelta) = (A.sectors_codomain..., A.sectors_domain...)

# Structural inner product: an abelian delta has a single allowed (unique-fusion) unit entry.
function LinearAlgebra.dot(a::AbelianSectorDelta, b::AbelianSectorDelta)
    axes(a) == axes(b) || throw(DimensionMismatch("sector mismatch in dot"))
    return 1
end

# `p`-norm: a single unit entry, so the norm is `1` for every `p` (including `Inf`), like its `dot`.
function LinearAlgebra.norm(a::AbelianSectorDelta{T}, p::Real = 2) where {T}
    return oneunit(real(float(T)))
end

# ========================  Accessors  ========================

sectoraxes(x, d::Int) = sectoraxes(x)[d]

# ========================  permutedims  ========================

# Permuting can mix codomain and domain legs, so the result collapses to an all-codomain delta.
function Base.permutedims(x::AbelianSectorDelta, perm)
    new_sectors = ntuple(n -> axes(x)[perm[n]], Val(ndims(x)))
    return AbelianSectorDelta{eltype(x)}(new_sectors)
end

# ========================  adjoint / broadcasting  ========================

function Base.copy(A::Adjoint{T, <:AbelianSectorDelta{T, <:Any, 2}}) where {T}
    return AbelianSectorDelta{T}(reverse(dual.(axes(adjoint(A)))))
end
function LinearAlgebra.adjoint!(
        A::AbelianSectorDelta{T, <:Any, 2}, B::AbelianSectorDelta{T, <:Any, 2}
    ) where {T}
    reverse(dual.(axes(B))) == axes(A) || throw(DimensionMismatch())
    return A
end

# ========================  multiplication  ========================

function Base.:(*)(
        a::AbelianSectorDelta{T₁, <:Any, 2},
        b::AbelianSectorDelta{T₂, <:Any, 2}
    ) where {T₁, T₂}
    axes(a, 2) == dual(axes(b, 1)) ||
        throw(DimensionMismatch("$(axes(a, 2)) != dual($(axes(b, 1))))"))
    T = Base.promote_type(T₁, T₂)
    return AbelianSectorDelta{T}((axes(a, 1), axes(b, 2)))
end
