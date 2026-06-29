"""
    AbstractSectorArray{T,S,N} <: AbstractArray{T,N}

Abstract supertype for data tensors labeled by sector information.
Concrete subtypes:

  - [`AbelianSectorArray`](@ref): unfused N-D abelian data tensor (one sector per axis)
  - [`SectorMatrix`](@ref): fused 2D data matrix (one coupled sector label)
"""
abstract type AbstractSectorArray{T, S, N} <: AbstractArray{T, N} end

sectortype(::Type{<:AbstractSectorArray{T, S}}) where {T, S} = S

"""
    data(sa::AbstractSectorArray)

Return the raw data array underlying the sector array.
"""
data(sa::AbstractSectorArray) = sa.data

# Reconstruct a sector array from its structural sector factor and raw data
# (the inverse of the `sector` / `data` split). Each concrete subtype defines a method.
function sector_kron end

Base.size(sa::AbstractSectorArray) = map(length, axes(sa))

Base.@propagate_inbounds function Base.getindex(
        A::AbstractSectorArray{T, <:Any, N},
        I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    return @inbounds data(A)[I...]
end
Base.@propagate_inbounds function Base.setindex!(
        A::AbstractSectorArray{T, <:Any, N},
        v,
        I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds data(A)[I...] = v
    return A
end

# Copy between sector arrays, including across the unfused/fused representations (e.g. writing
# a block into a graded array). The axes carry the sector labels, so the equality check
# rejects a sector mismatch; the raw data is then copied directly.
function Base.copy!(dest::AbstractSectorArray, src::AbstractSectorArray)
    axes(dest) == axes(src) || throw(
        DimensionMismatch("sector axes mismatch in copy!: $(axes(dest)) vs $(axes(src))")
    )
    copyto!(data(dest), data(src))
    return dest
end

# ========================  Shared utilities  ========================

function require_unique_fusion(A)
    return TKS.FusionStyle(sectortype(A)) === TKS.UniqueFusion() ||
        error("not implemented for non-abelian tensors")
end

# ========================  scale! / zero!  ========================

function TensorAlgebra.scale!(a::AbstractSectorArray, β::Number)
    scale!(data(a), β)
    return a
end

function TensorAlgebra.zero!(a::AbstractSectorArray)
    zero!(data(a))
    return a
end

# ========================  display  ========================

function Base.print_array(io::IO, sa::AbstractSectorArray)
    Base.print_array(io, sector(sa))
    println(io, "\n ⊗")
    Base.print_array(io, data(sa))
    return nothing
end

function Base.show(io::IO, sa::AbstractSectorArray)
    show(io, sector(sa))
    print(io, " ⊗ ")
    show(io, data(sa))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sa::AbstractSectorArray)
    summary(io, sa)
    println(io, ":")
    Base.print_array(io, sa)
    return nothing
end
