"""
    AbelianSectorDelta{T,S<:SectorRange,N} <: AbstractSectorDelta{T, S, N}

Unfused N-D structural tensor for abelian symmetries. Stores one `SectorRange` per axis.
For abelian symmetries, every element equals `one(T)` (the Kronecker delta selection rule).
"""
struct AbelianSectorDelta{T, S <: SectorRange, N} <: AbstractSectorDelta{T, S, N}
    sectors::NTuple{N, S}
end
# Convenience: infer N and S from the sectors. Requires at least one sector: the sector
# type of a rank-0 delta cannot be inferred from an empty tuple, so a rank-0 delta is built
# through the fully-parameterized constructor with an explicit `S`.
function AbelianSectorDelta{T}(sectors::Tuple{SectorRange, Vararg{SectorRange}}) where {T}
    return AbelianSectorDelta{T, eltype(sectors), length(sectors)}(sectors)
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

Base.axes(A::AbelianSectorDelta) = A.sectors

# ========================  Accessors  ========================

isdual(x, d::Int) = isdual(axes(x, d))
sectoraxes(x, d::Int) = sectoraxes(x)[d]

# ========================  permutedims  ========================

function Base.permutedims(x::AbelianSectorDelta, perm)
    new_sectors = ntuple(n -> x.sectors[perm[n]], Val(ndims(x)))
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
