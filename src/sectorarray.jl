# Axes
# ----
"""
    const SectorUnitRange{I <: SectorRange, RB <: AbstractUnitRange{Int}, R <: AbstractUnitRange{Int}} =
        CartesianProductUnitRange{Int, I, RB, R}

Type alias for the cartesian product of a sector range of type `I`, and a unit range of type `RB`, which yields a total range of type `R`.
"""
const SectorUnitRange{I <: SectorRange, RB <: AbstractUnitRange{Int}, R <: AbstractUnitRange{Int}} =
    CartesianProductUnitRange{Int, I, RB, R}

const SectorOneTo{I <: SectorRange, R <: AbstractUnitRange{Int}} = SectorUnitRange{I, Base.OneTo{Int}, R}

function SectorUnitRange(sector::SectorRange, range::AbstractUnitRange, isdual::Bool = false)
    return cartesianrange(sector, range)
end

sectorrange(sector::SectorRange, range::AbstractUnitRange, isdual::Bool = false) = (isdual ? dual(sector) : sector) × range
sectorrange(sector::SectorRange, dim::Integer, isdual::Bool = false) = sectorrange(sector, 1:dim, isdual)
sectorrange(sector_dim::Pair{<:SectorRange}, isdual::Bool = false) = sectorrange(sector_dim..., isdual)

sectorrange(sector::NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}}, range::AbstractUnitRange, isdual::Bool = false) =
    sectorrange(to_sector(sector), range, isdual)
sectorrange(sector::NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}}, dim::Integer, isdual::Bool = false) = sectorrange(sector, 1:dim, isdual)
sectorrange(sector_dim::Pair{<:NamedTuple{<:Any, <:Tuple{SectorRange, Vararg{SectorRange}}}}, isdual::Bool = false) = sectorrange(sector_dim..., isdual)

×(a::SectorRange, g::AbstractUnitRange) = cartesianrange(a, g)
×(g::AbstractUnitRange, a::SectorRange) = cartesianrange(a, g)

"""
    const SectorOneTo{I <: SectorRange} =
        SectorUnitRange{I, Base.OneTo{Int}, Base.OneTo{Int}}
"""
const SectorOneTo{I <: SectorRange} = SectorUnitRange{I, Base.OneTo{Int}, Base.OneTo{Int}}

sector_type(::Type{<:SectorUnitRange{I}}) where {I} = I
sector(r::SectorUnitRange) = kroneckerfactors(r, 1)
sector_multiplicity(r::SectorUnitRange) = length(kroneckerfactors(r, 2))
sectors(r::SectorUnitRange) = [sector(r)]
sector_multiplicities(r::SectorUnitRange) = [sector_multiplicity(r)]

# should this be trivial?
sector(x) = nothing
sectors(x) = nothing

dual(x::SectorUnitRange) = cartesianrange(dual(kroneckerfactors(x, 1)), kroneckerfactors(x, 2), unproduct(x))
flip(x::SectorUnitRange) = cartesianrange(flip(kroneckerfactors(x, 1)), kroneckerfactors(x, 2), unproduct(x))
isdual(x::SectorUnitRange) = isdual(kroneckerfactors(x, 1))

flux(sr::SectorUnitRange) = sector(sr)

# allow getindex for abelian symmetries
function Base.getindex(x::SectorUnitRange, y::AbstractUnitRange{Int})
    return cartesianrange(kroneckerfactors(x, 1), kroneckerfactors(x, 2)[y], unproduct(x)[y])
end

ungrade(x::SectorUnitRange) = KroneckerArrays.unproduct(x)

function Base.show(io::IO, g::SectorOneTo)
    return print(io, "sectorrange(", sector(g), " => ", unproduct(g), ")")
end

# Array
# -----
"""
    SectorDelta{T}(sectors::NTuple{N, I}) <: AbstractArray{T, N}

An immutable representation of the structural tensor associated to the representation space of a number of sectors.
For abelian symmetries, this boils down to a scalar which can always be normalized to 1.
"""
struct SectorDelta{T, N, I <: SectorRange} <: AbstractArray{T, N}
    sectors::NTuple{N, I}
end
SectorDelta{T}(sectors::NTuple{N, I}) where {T, N, I} = SectorDelta{T, N, I}(sectors)

require_unique_fusion(A) = TKS.FusionStyle(sector_type(A)) === TKS.UniqueFusion() || error("not implemented for non-abelian tensors")

Base.@propagate_inbounds function Base.getindex(A::SectorDelta{T, N}, I::Vararg{Int, N}) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return one(T)
end

Base.axes(A::SectorDelta) = A.sectors
Base.size(A::SectorDelta) = length.(axes(A))

Base.similar(::SectorDelta, ::Type{T}, sectors::Tuple{I, Vararg{I}}) where {T, I <: SectorRange} =
    SectorDelta{T}(sectors)
Base.similar(::Type{<:AbstractArray{T}}, sectors::Tuple{I, Vararg{I}}) where {T, I <: SectorRange} =
    SectorDelta{T}(sectors)

sectors(x::SectorDelta) = x.sectors
sector_type(::Type{SectorDelta{T, N, I}}) where {T, N, I} = I

function Base.permutedims(x::SectorDelta, perm)
    return SectorDelta{eltype(x)}(Base.Fix1(getindex, sectors(x)).(perm))
end
KroneckerArrays.FunctionImplementations.permuteddims(x::SectorDelta, perm) = permutedims(x, perm)

# Defined as this makes broadcasting work better
Base.copy(A::SectorDelta) = A
function Base.copy!(C::SectorDelta, A::SectorDelta)
    axes(C) == axes(A) || throw(DimensionMismatch())
    return C
end
function Base.copyto!(C::SectorDelta, A::SectorDelta)
    axes(C) == axes(A) || throw(DimensionMismatch())
    return C
end
Base.copy(A::Adjoint{T, <:SectorDelta{T, 2}}) where {T} = SectorDelta{T}(reverse(dual.(sectors(adjoint(A)))))
function LinearAlgebra.adjoint!(A::SectorDelta{T, 2, I}, B::SectorDelta{T, 2, I}) where {T, I}
    reverse(dual.(sectors(B))) == sectors(A) || throw(DimensionMismatch())
    return A
end

function Base.:(*)(a::SectorDelta{T₁, 2, I}, b::SectorDelta{T₂, 2, I}) where {T₁, T₂, I}
    axes(a, 2) == dual(axes(b, 1)) || throw(DimensionMismatch("$(axes(a, 2)) != dual($(axes(b, 1))))"))
    T = Base.promote_type(T₁, T₂)
    return SectorDelta{T}((axes(a, 1), axes(b, 2)))
end

# want to add something to opt out of the broadcasting kronecker thingies
# so need something to dispatch on...
# struct SectorStyle{I, N} <: Broadcast.AbstractArrayStyle{N} end
# SectorStyle{I, N}(::Val{M}) where {I, N, M} = SectorStyle{I, M}()
#
# Base.BroadcastStyle(::Type{T}) where {T <: SectorDelta} = SectorStyle{sector_type(T), ndims(T)}
#

"""
    SectorArray(sectors, data) <: AbstractKroneckerArray

A representation of a general symmetric array as the combination of a structural part (`sectors`) and a data part (`data`).
This can be thought of as a direct implementation of the Wigner-Eckart theorem.
"""
struct SectorArray{T, N, I <: SectorRange, A <: AbstractArray{T, N}} <: AbstractKroneckerArray{T, N}
    sectors::NTuple{N, I}
    data::A

    # constructing from undef
    function SectorArray{T, N, I, A}(::UndefInitializer, axs::Tuple{Vararg{SectorUnitRange{I}, N}}) where {T, N, I, A}
        sectors = kroneckerfactors.(axs, 1)
        data = similar(A, kroneckerfactors.(axs, 2))
        return new{T, N, I, A}(sectors, data)
    end

    # constructing from data
    function SectorArray{T, N, I, A}(sectors::NTuple{N, I}, data::A) where {T, N, I, A}
        return new{T, N, I, A}(sectors, data)
    end
end

function SectorArray{T}(::UndefInitializer, axs::Tuple{Ax, Vararg{Ax}}) where {T, Ax <: SectorUnitRange}
    N = length(axs)
    I = sector_type(Ax)
    return SectorArray{T, N, I, Array{T, N}}(undef, axs)
end
function SectorArray(sectors::NTuple{N, I}, data::AbstractArray{T, N}) where {T, I, N}
    return SectorArray{T, N, I, typeof(data)}(sectors, data)
end

const SectorMatrix{T, I, A <: AbstractMatrix{T}} = SectorArray{T, 2, I, A}

# Accessors
# ---------
KroneckerArrays.kroneckerfactors(A::SectorArray) = (SectorDelta{eltype(A)}(sectors(A)), A.data)
KroneckerArrays.kroneckerfactortypes(::Type{SectorArray{T, N, I, A}}) where {T, N, I, A} = (SectorDelta{T, N, I}, A)

sectors(A::SectorArray) = A.sectors

sector_type(::Type{<:SectorArray{T, N, I}}) where {T, N, I} = I
data_type(::Type{SectorArray{T, N, I, A}}) where {T, N, I, A} = A

# AbstractArray interface
# -----------------------
Base.@propagate_inbounds function Base.getindex(A::SectorArray{T, N}, I::Vararg{Int, N}) where {T, N}
    TKS.FusionStyle(sector_type(A)) === TKS.UniqueFusion() ||
        error("not implemented for non-abelian tensors")
    @boundscheck checkbounds(A, I...)
    return @inbounds A.data[I...]
end
Base.@propagate_inbounds function Base.setindex!(A::SectorArray{T, N}, v, I::Vararg{Int, N}) where {T, N}
    TKS.FusionStyle(sector_type(A)) === TKS.UniqueFusion() ||
        error("not implemented for non-abelian tensors")
    @boundscheck checkbounds(A, I...)
    @inbounds A.data[I...] = v
    return A
end

function Base.similar(A::AbstractArray, elt::Type, axs::Tuple{SectorUnitRange, Vararg{SectorUnitRange}})
    return SectorArray(kroneckerfactors.(axs, 1), similar(A, elt, kroneckerfactors.(axs, 2)))
end
# disambiguate
function Base.similar(A::AbstractKroneckerArray, elt::Type, axs::Tuple{SectorUnitRange, Vararg{SectorUnitRange}})
    return SectorArray(kroneckerfactors.(axs, 1), similar(A, elt, kroneckerfactors.(axs, 2)))
end
function Base.similar(::Type{A}, axs::Tuple{SectorUnitRange, Vararg{SectorUnitRange}}) where {A <: SectorArray}
    return SectorArray(kroneckerfactors.(axs, 1), similar(data_type(A), kroneckerfactors.(axs, 2)))
end

Base.copy(A::SectorArray) = SectorArray(sectors(A), copy(A.data))
function Base.copy!(C::SectorArray, A::SectorArray)
    axes(C) == axes(A) || throw(DimensionMismatch()) # TODO: sector error?
    copy!(arg2(C), arg2(A))
    return C
end

function Base.convert(::Type{SectorArray{T₁, N, I, A}}, x::SectorArray{T₂, N, I, B})::SectorArray{T₁, N, I, A} where {T₁, T₂, N, I, A, B}
    A === B && return x
    return SectorArray(sectors(x), convert(A, x.data))
end

# Avoid infinite recursion while eagerly adjointing the sectors
function Base.adjoint(a::SectorArray)
    sectors_adjoint = adjoint(kroneckerfactors(a, 1))
    return SectorArray(axes(sectors_adjoint), adjoint(kroneckerfactors(a, 2)))
end

# Other
# -----
KroneckerArrays.:(⊗)(A::SectorDelta{T, N}, data::AbstractArray{T, N}) where {T, N} = SectorArray(A.sectors, data)
function KroneckerArrays.:(⊗)(A::SectorDelta{T₁, N}, data::AbstractArray{T₂, N}) where {T₁, T₂, N}
    T = Base.promote_type(*, T₁, T₂)
    return SectorArray(A.sectors, collect(T, data))
end

# TODO: can we avoid this?
function Base.materialize!(dst::SectorArray, src::KroneckerArrays.KroneckerBroadcasted)
    Base.materialize!(kroneckerfactors(dst, 1), kroneckerfactors(src, 1))
    Base.materialize!(kroneckerfactors(dst, 2), kroneckerfactors(src, 2))
    return dst
end
