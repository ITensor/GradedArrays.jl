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

# The axes decompose into the structural sector factor and the reduced data, derived once here from
# the `sector`/`data` primitives every concrete subtype provides, so no type re-derives them and
# they cannot drift apart. `size` follows from `axes`.
sectoraxes(sa::AbstractSectorArray) = axes(sector(sa))
dataaxes(sa::AbstractSectorArray) = axes(data(sa))
Base.axes(sa::AbstractSectorArray) = map(SectorOneTo, sectoraxes(sa), dataaxes(sa))

Base.size(sa::AbstractSectorArray) = map(length, axes(sa))

# Scalar indexing reads/writes the reduced data directly, which only coincides with the full array
# for unique (abelian) fusion, where the structural factor is trivial and `size == size(data)`. For
# non-abelian fusion the full extent exceeds the reduced data, so scalar indexing would run past it;
# require unique fusion rather than silently reading/writing out of bounds.
Base.@propagate_inbounds function Base.getindex(
        A::AbstractSectorArray{T, <:Any, N},
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return @inbounds data(A)[I...]
end
Base.@propagate_inbounds function Base.setindex!(
        A::AbstractSectorArray{T, <:Any, N},
        v,
        I::Vararg{Int, N}
    ) where {T, N}
    require_unique_fusion(A)
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

# `fill!` sets the symmetry-allowed (reduced) data, leaving the structural sector factor unchanged;
# it is a shorthand for setting the allowed values, like `rand!`/`randn!`, not a fill of the dense
# array (which the structural factor would scale for a non-abelian sector). A zero fill routes
# through `zero!`, which storage backends can implement more efficiently than a general fill.
function Base.fill!(a::AbstractSectorArray, v)
    iszero(v) && return zero!(a)
    fill!(data(a), v)
    return a
end

# ========================  trace  ========================

# A 2D sector block is the tensor product of its structural factor `sector(a)` and its reduced
# data `data(a)`, so the trace factorizes: the sector's quantum dimension times the trace of
# the reduced data.
function LinearAlgebra.tr(a::AbstractSectorArray{<:Any, <:Any, 2})
    return LinearAlgebra.tr(sector(a)) * LinearAlgebra.tr(data(a))
end

# The inner product factorizes the same way: the structural factors contract to their inner product
# (the squared Frobenius norm of the delta, a quantum-dimension weight) times the reduced-data inner
# product. Matches `dot(Array(a), Array(b))` (the `kron`/`dot` mixed-product identity) and the
# quantum-dimension weighting a fused graded array applies to each block.
function LinearAlgebra.dot(a::AbstractSectorArray, b::AbstractSectorArray)
    return LinearAlgebra.dot(sector(a), sector(b)) * LinearAlgebra.dot(data(a), data(b))
end

# The `p`-norm factorizes through the Kronecker structure the same way: the product of the
# structural-factor norm and the reduced-data norm (the `norm(A ⊗ B, p) = norm(A, p) * norm(B, p)`
# identity that `KroneckerArrays` uses). Correct for every `p`, `Inf` included, with no special case.
function LinearAlgebra.norm(a::AbstractSectorArray, p::Real = 2)
    return LinearAlgebra.norm(sector(a), p) * LinearAlgebra.norm(data(a), p)
end

# ========================  densification  ========================

# Materialize the structural factor `sector(a) ⊗ data(a)` densely, the same per-block densification
# a fused graded array uses (`I ⊗ reduced` for the identity/ones structural factor, repeating each
# reduced value over the irrep's quantum dimension). The generic `AbstractArray` fallback can't be
# used: `size` is the full (structural × reduced) extent while `data` is only the reduced block, so
# copying elementwise scalar-indexes past the reduced data into garbage.
Base.Array(a::AbstractSectorArray) = kron_nd(Array(sector(a)), Array(data(a)))

# ========================  random fills  ========================

# Fill the reduced data block, bypassing the generic `AbstractArray` fallbacks that go through
# (disallowed) scalar indexing. The 3-arg `rand!(rng, a, sp)` form is what Random's `rand!` entry
# points ultimately call.
function Random.rand!(rng::AbstractRNG, a::AbstractSectorArray, sp::Random.Sampler)
    Random.rand!(rng, data(a), sp)
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbstractSectorArray)
    Random.randn!(rng, data(a))
    return a
end

# ========================  conj  ========================

# Conjugate the data, flip the duality of every axis, and (for fermions) apply the fermionic phase
# from reversing the leg order. Routed through the lazy conjugating broadcast so there is a single
# implementation: `conj.` lowers to a `ConjArray` (dualizing the axes) and materializes via
# `bipermutedimsopadd!` with `op = conj`, which carries the reversal phase that a bare data
# conjugation drops. This also overrides Base's `conj(::AbstractArray{<:Real}) = A` short-circuit,
# so a real-eltype sector array still dualizes its axes.
Base.conj(a::AbstractSectorArray) = conj.(a)

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
