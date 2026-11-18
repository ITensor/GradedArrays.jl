# A block of a graded array with abelian symmetry: a data array plus one sector and one arrow per
# axis, split into `NC` codomain and `ND` domain legs.
struct UniqueSectorArray{T, S <: Sector, N, NC, ND, A <: AbstractArray{T, N}} <:
    AbstractSectorArray{T, S, N}
    data::A
    sectors_codomain::NTuple{NC, S}
    isduals_codomain::NTuple{NC, Bool}
    sectors_domain::NTuple{ND, S}
    isduals_domain::NTuple{ND, Bool}
    function UniqueSectorArray{T, S, N, NC, ND, A}(
            data::A,
            sectors_codomain::NTuple{NC, S}, isduals_codomain::NTuple{NC, Bool},
            sectors_domain::NTuple{ND, S}, isduals_domain::NTuple{ND, Bool}
        ) where {T, S <: Sector, N, NC, ND, A <: AbstractArray{T, N}}
        NC + ND == N ||
            throw(ArgumentError("codomain ($NC) + domain ($ND) legs must equal N ($N)"))
        return new{T, S, N, NC, ND, A}(
            data, sectors_codomain, isduals_codomain, sectors_domain, isduals_domain
        )
    end
end

# Constructors

function UniqueSectorArray(
        data::AbstractArray{T, N},
        sectors_codomain::NTuple{NC, S}, isduals_codomain::NTuple{NC, Bool},
        sectors_domain::NTuple{ND, S}, isduals_domain::NTuple{ND, Bool}
    ) where {T, S <: Sector, N, NC, ND}
    return UniqueSectorArray{T, S, N, NC, ND, typeof(data)}(
        data, sectors_codomain, isduals_codomain, sectors_domain, isduals_domain
    )
end

# Covers rank 0, where the empty tuples carry no `S`.
function UniqueSectorArray{T, S, N}(
        data::AbstractArray{T, N},
        sectors_codomain::NTuple{NC, S}, isduals_codomain::NTuple{NC, Bool},
        sectors_domain::NTuple{ND, S}, isduals_domain::NTuple{ND, Bool}
    ) where {T, S <: Sector, N, NC, ND}
    return UniqueSectorArray{T, S, N, NC, ND, typeof(data)}(
        data, sectors_codomain, isduals_codomain, sectors_domain, isduals_domain
    )
end

function UniqueSectorArray(
        data::AbstractArray,
        sectors_codomain::Tuple{Vararg{AbstractOrientedSector}},
        sectors_domain::Tuple{Vararg{AbstractOrientedSector}}
    )
    return UniqueSectorArray(
        data, splitarrows(sectors_codomain)..., splitarrows(sectors_domain)...
    )
end

function UniqueSectorArray(
        data::AbstractArray,
        sectors::Tuple{Vararg{AbstractOrientedSector}}
    )
    return UniqueSectorArray(data, sectors, ())
end

# The delta's eltype may differ from the data's, for example for `real`/`imag`.
function UniqueSectorArray(
        data::AbstractArray{T}, delta::UniqueSectorDelta{<:Any, S, N, NC, ND}
    ) where {T, S, N, NC, ND}
    return UniqueSectorArray{T, S, N, NC, ND, typeof(data)}(
        data, delta.sectors_codomain, delta.isduals_codomain,
        delta.sectors_domain, delta.isduals_domain
    )
end

function UniqueSectorArray{T}(
        ::UndefInitializer, axs::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    N = length(axs)
    S = sectortype(eltype(axs))
    return UniqueSectorArray{T, S, N, N, 0, Array{T, N}}(
        similar(Array{T, N}, data.(axs)), sector.(axs), isdual.(axs), (), ()
    )
end

const UniqueSectorVector{T, S <: Sector, NC, ND, A <: AbstractVector{T}} =
    UniqueSectorArray{T, S, 1, NC, ND, A}
const UniqueSectorMatrix{T, S <: Sector, NC, ND, A <: AbstractMatrix{T}} =
    UniqueSectorArray{T, S, 2, NC, ND, A}

# Accessors

function structure(sa::UniqueSectorArray{T, S, N, NC, ND, A}) where {T, S, N, NC, ND, A}
    return UniqueSectorDelta{T, S, N, NC, ND}(
        sa.sectors_codomain, sa.isduals_codomain, sa.sectors_domain, sa.isduals_domain
    )
end

datatype(::Type{<:UniqueSectorArray{T, S, N, NC, ND, A}}) where {T, S, N, NC, ND, A} = A

function Base.copy(a::UniqueSectorArray)
    return UniqueSectorArray(copy(data(a)), structure(a))
end

# A sub-view of the reduced data, keeping the sectors.
function Base.view(
        a::UniqueSectorArray{<:Any, <:Any, N}, I::Vararg{AbstractUnitRange, N}
    ) where {N}
    return sector_kron(structure(a), view(data(a), I...))
end

function Base.similar(
        ::UniqueSectorArray,
        ::Type{T},
        axes::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    return UniqueSectorArray{T}(undef, axes)
end

function Base.convert(
        ::Type{UniqueSectorArray{T₁, S, N, NC, ND, A}},
        x::UniqueSectorArray{T₂, S, N, NC, ND, B}
    )::UniqueSectorArray{T₁, S, N, NC, ND, A} where {T₁, T₂, S, N, NC, ND, A, B}
    A === B && return x
    return UniqueSectorArray(convert(A, data(x)), structure(x))
end

# ========================  permutedims  ========================

# The result has all legs in the codomain.
function Base.permutedims(x::UniqueSectorArray, perm)
    new_axes = ntuple(n -> axes(x, perm[n]), Val(ndims(x)))
    return permutedims!(similar(x, new_axes), x, perm)
end
function Base.permutedims!(y::UniqueSectorArray, x::UniqueSectorArray, perm)
    TensorAlgebra.permutedimsconjadd!(y, false, x, perm, true, false)
    return y
end

# ========================  conj  ========================

# Keeps the codomain/domain split, unlike `conj.(a)`. The permute-add applies the fermionic leg
# reversal sign, which a bare data `conj` would drop.
function Base.conj(a::UniqueSectorArray{T, S, N, NC, ND}) where {T, S, N, NC, ND}
    dest = UniqueSectorArray{T, S, N}(
        similar(data(a)),
        a.sectors_codomain, map(!, a.isduals_codomain),
        a.sectors_domain, map(!, a.isduals_domain)
    )
    TensorAlgebra.bipermutedimsconjadd!(
        dest, true, a, ntuple(identity, Val(NC)), ntuple(i -> NC + i, Val(ND)), true, false
    )
    return dest
end

# ========================  mul!  ========================

# TODO: Define this as part of:
# `check_input(::typeof(mul!), ::UniqueSectorMatrix, ::UniqueSectorMatrix, ::UniqueSectorMatrix)`
function check_mul_axes(
        c::UniqueSectorMatrix,
        a::UniqueSectorMatrix,
        b::UniqueSectorMatrix
    )
    structureaxes(a, 2) == dual(structureaxes(b, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    structureaxes(c, 1) == structureaxes(a, 1) || throw(DimensionMismatch())
    structureaxes(c, 2) == structureaxes(b, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        c::UniqueSectorMatrix, a::UniqueSectorMatrix, b::UniqueSectorMatrix, α::Number,
        β::Number
    )
    check_mul_axes(c, a, b)
    mul!(data(c), data(a), data(b), α, β)
    return c
end

# ========================  twist!  ========================

function TA.twist!(a::UniqueSectorArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    phase = mapreduce(i -> twist(sector(structureaxes(a, i))), *, dims; init = 1)
    isone(phase) || (data(a) .*= phase)
    return a
end

# ========================  Other  ========================

function sector_kron(
        s::UniqueSectorDelta{<:Any, <:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return UniqueSectorArray(data, s)
end
