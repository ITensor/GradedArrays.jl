"""
    AbelianSectorArray{T,S,N,NC,ND,A} <: AbstractSectorArray{T,S,N}

Unfused N-D data tensor for abelian symmetries. Stores a dense data array plus one `SectorRange`
per axis with a codomain/domain split (`NC` codomain legs, `ND` domain legs, `NC + ND == N`).
Implements the Wigner-Eckart decomposition: the full tensor is the Kronecker product of the
structural [`AbelianSectorDelta`](@ref) (`sector`) with the data array (reduced matrix elements).
The all-codomain case (`NC == N`) is the block an `AbelianGradedArray` yields.
"""
struct AbelianSectorArray{T, S <: SectorRange, N, NC, ND, A <: AbstractArray{T, N}} <:
    AbstractSectorArray{T, S, N}
    data::A
    sectors_codomain::NTuple{NC, S}
    sectors_domain::NTuple{ND, S}
    function AbelianSectorArray{T, S, N, NC, ND, A}(
            data::A, sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
        ) where {T, S <: SectorRange, N, NC, ND, A <: AbstractArray{T, N}}
        NC + ND == N ||
            throw(ArgumentError("codomain ($NC) + domain ($ND) legs must equal N ($N)"))
        return new{T, S, N, NC, ND, A}(data, sectors_codomain, sectors_domain)
    end
end

# Constructors
# `AbelianSectorArray` is a block type (the `fa[Block]` view, and the `AbelianGradedArray` block),
# not meant to be built directly; these are the forms used internally.

# Primary: data plus the two sector tuples, inferring the parameters. `N == NC + ND`.
function AbelianSectorArray(
        data::AbstractArray{T, N},
        sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
    ) where {T, S <: SectorRange, N, NC, ND}
    return AbelianSectorArray{T, S, N, NC, ND, typeof(data)}(
        data, sectors_codomain, sectors_domain
    )
end

# `S` explicit, `A`/`NC`/`ND` inferred: lets a construction that already knows `T`/`S`/`N` skip the
# derivable parameters, and covers the rank-0 case (empty tuples carry no `S`) that the parameterless
# primary cannot infer. Used by `conj`.
function AbelianSectorArray{T, S, N}(
        data::AbstractArray{T, N},
        sectors_codomain::NTuple{NC, S}, sectors_domain::NTuple{ND, S}
    ) where {T, S <: SectorRange, N, NC, ND}
    return AbelianSectorArray{T, S, N, NC, ND, typeof(data)}(
        data, sectors_codomain, sectors_domain
    )
end

# All-codomain shorthand from a flat sector tuple: the block an `AbelianGradedArray` yields, and
# what `fa[Block]` returns when there is no domain leg. Flat always means all-codomain.
function AbelianSectorArray(
        data::AbstractArray{T, N}, sectors::NTuple{N, S}
    ) where {T, S <: SectorRange, N}
    return AbelianSectorArray(data, sectors, ())
end

# Inverse of the `sector`/`data` split, preserving the split; used by `sector_kron` and the per-op
# forwards. The sector tuples are eltype-independent, so this also covers a delta whose eltype
# differs from the data (e.g. `real`/`imag`).
function AbelianSectorArray(
        data::AbstractArray{T}, delta::AbelianSectorDelta{<:Any, S, N, NC, ND}
    ) where {T, S, N, NC, ND}
    return AbelianSectorArray{T, S, N, NC, ND, typeof(data)}(
        data, delta.sectors_codomain, delta.sectors_domain
    )
end

# `undef` for `similar`: all-codomain, from a flat tuple of SectorOneTo axes (`S` from the axes,
# so at least one axis is required).
function AbelianSectorArray{T}(
        ::UndefInitializer, axs::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    N = length(axs)
    S = sectortype(eltype(axs))
    return AbelianSectorArray{T, S, N, N, 0, Array{T, N}}(
        similar(Array{T, N}, data.(axs)), sector.(axs), ()
    )
end

const AbelianSectorVector{T, S <: SectorRange, NC, ND, A <: AbstractVector{T}} =
    AbelianSectorArray{T, S, 1, NC, ND, A}
const AbelianSectorMatrix{T, S <: SectorRange, NC, ND, A <: AbstractMatrix{T}} =
    AbelianSectorArray{T, S, 2, NC, ND, A}

# Accessors

# Kronecker factor decomposition: AbelianSectorArray = sector ⊗ data. `sector` wraps the stored
# codomain/domain sector tuples in a delta, so `sector_kron(sector(a), data(a)) === a`.
function sector(sa::AbelianSectorArray{T, S, N, NC, ND, A}) where {T, S, N, NC, ND, A}
    return AbelianSectorDelta{T, S, N, NC, ND}(sa.sectors_codomain, sa.sectors_domain)
end

datatype(::Type{<:AbelianSectorArray{T, S, N, NC, ND, A}}) where {T, S, N, NC, ND, A} = A

function Base.copy(a::AbelianSectorArray)
    return AbelianSectorArray(copy(data(a)), a.sectors_codomain, a.sectors_domain)
end

# similar for AbelianSectorArray with SectorOneTo axes.
# Delegates to similar on the data array for the data dimensions.
function Base.similar(
        ::AbelianSectorArray,
        ::Type{T},
        axes::Tuple{SectorOneTo, Vararg{SectorOneTo}}
    ) where {T}
    return AbelianSectorArray{T}(undef, axes)
end

function Base.convert(
        ::Type{AbelianSectorArray{T₁, S, N, NC, ND, A}},
        x::AbelianSectorArray{T₂, S, N, NC, ND, B}
    )::AbelianSectorArray{T₁, S, N, NC, ND, A} where {T₁, T₂, S, N, NC, ND, A, B}
    A === B && return x
    return AbelianSectorArray(convert(A, data(x)), sector(x))
end

# ========================  permutedims  ========================

# Permuting can mix codomain and domain legs, so the result is all-codomain. The permuted axes carry
# both the new sector labels and the destination data shape; the fermion sign from the permutation is
# applied to the reduced data by `permutedimsopadd!`, not to the structural factor (which is `one(T)`
# at its single allowed entry and cannot carry a sign).
function Base.permutedims(x::AbelianSectorArray, perm)
    new_axes = ntuple(n -> axes(x, perm[n]), Val(ndims(x)))
    return permutedims!(similar(x, new_axes), x, perm)
end
function Base.permutedims!(y::AbelianSectorArray, x::AbelianSectorArray, perm)
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ========================  conj  ========================

# Conjugate while keeping the codomain/domain split, mirroring `conj(::FusionArray)` (the generic
# `conj.(a)` broadcast collapses to all-codomain). A same-split destination with every stored sector
# dualized is filled by a single `op = conj` permute-add over the identity biperm; the fermionic
# leg-reversal sign rides `bipermutedimsopadd!` (folded in by `fermion_permutation_phase`), which a
# bare data `conj` would drop.
function Base.conj(a::AbelianSectorArray{T, S, N, NC, ND}) where {T, S, N, NC, ND}
    dest = AbelianSectorArray{T, S, N}(
        similar(data(a)), map(dual, a.sectors_codomain), map(dual, a.sectors_domain)
    )
    TensorAlgebra.bipermutedimsopadd!(
        dest, conj, a, ntuple(identity, Val(NC)), ntuple(i -> NC + i, Val(ND)), true, false
    )
    return dest
end

# ========================  mul!  ========================

# TODO: Define this as part of:
# `check_input(::typeof(mul!), ::AbelianSectorMatrix, ::AbelianSectorMatrix, ::AbelianSectorMatrix)`
function check_mul_axes(
        c::AbelianSectorMatrix,
        a::AbelianSectorMatrix,
        b::AbelianSectorMatrix
    )
    sectoraxes(a, 2) == dual(sectoraxes(b, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    sectoraxes(c, 1) == sectoraxes(a, 1) || throw(DimensionMismatch())
    sectoraxes(c, 2) == sectoraxes(b, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        c::AbelianSectorMatrix, a::AbelianSectorMatrix, b::AbelianSectorMatrix, α::Number,
        β::Number
    )
    check_mul_axes(c, a, b)
    mul!(data(c), data(a), data(b), α, β)
    return c
end

# ========================  twist!  ========================

function twist!(a::AbelianSectorArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    phase = mapreduce(i -> twist(sectoraxes(a, i)), *, dims; init = 1)
    isone(phase) || (data(a) .*= phase)
    return a
end

# ========================  Other  ========================

function sector_kron(
        s::AbelianSectorDelta{<:Any, <:Any, N},
        data::AbstractArray{<:Any, N}
    ) where {N}
    return AbelianSectorArray(data, s)
end
