"""
    AbstractFusedArray{T,S,N} <: AbstractArray{T,N}

Supertype of the fused (coupled-sector-block) graded arrays, [`FusedGradedMatrix`](@ref) and
[`FusedGradedVector`](@ref). Holds the code shared between the two.
"""
abstract type AbstractFusedArray{T, S, N} <: AbstractArray{T, N} end
const AbstractFusedMatrix{T, S} = AbstractFusedArray{T, S, 2}
const AbstractFusedVector{T, S} = AbstractFusedArray{T, S, 1}

using BlockArrays: mortar
using FillArrays: Zeros

# The block storage type is the datatype of the blocks, so a concrete fused array only needs to
# define `blocktype`.
datatype(::Type{T}) where {T <: AbstractFusedArray} = datatype(blocktype(T))
datatype(a::AbstractFusedArray) = datatype(typeof(a))
sectortype(::Type{<:AbstractFusedArray{T, S}}) where {T, S} = S

function isblockdiagonal(A::AbstractFusedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

# Diagonal when block-diagonal and every stored (diagonal) block is itself diagonal. Checking blocks
# avoids densifying a block-sparse matrix through the generic dense-slicing fallback.
function LinearAlgebra.isdiag(A::AbstractFusedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        (row == col && LinearAlgebra.isdiag(view(A, bI))) || return false
    end
    return true
end

# ---------------------------------------------------------------------------
#  fill! / zero! / scale! — block-wise over the stored blocks
#
#  Defined once via the `eachblockstoredindex`/`view` interface every
#  `AbstractFusedArray` implements, so both fused subtypes are covered.
#  These only touch stored (symmetry-allowed) blocks, so a nonzero `fill!`
#  value leaves the forbidden positions at zero.
# ---------------------------------------------------------------------------

function TensorAlgebra.scale!(a::AbstractFusedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

# The `LinearAlgebra` spelling of blockwise scaling (the generic fallback
# scalar-indexes).
LinearAlgebra.rmul!(a::AbstractFusedArray, β::Number) = TensorAlgebra.scale!(a, β)
LinearAlgebra.lmul!(β::Number, a::AbstractFusedArray) = TensorAlgebra.scale!(a, β)

function TensorAlgebra.zero!(a::AbstractFusedArray)
    for bI in eachblockstoredindex(a)
        zero!(view(a, bI))
    end
    return a
end

function Base.fill!(a::AbstractFusedArray, v)
    for bI in eachblockstoredindex(a)
        fill!(view(a, bI), v)
    end
    return a
end

# ---------------------------------------------------------------------------
#  Display — render through a BlockArrays block array. BlockArrays draws the
#  block grid; unstored blocks become `Zeros`, which print as `⋅`.
# ---------------------------------------------------------------------------

# Compact type name for the summary line. The sector parameter is dotted (it is spelled out in
# full in the `Dim` lines below, so repeating it in the header only adds noise); the element,
# order, and storage parameters are kept. `make_typealias` recovers the `Vector`/`Matrix` alias
# names and leaves the order `N` explicit for higher-rank arrays.
function summary_typename(type::Type{<:AbstractFusedArray})
    alias = Base.make_typealias(type)
    base, params = if isnothing(alias)
        string(nameof(type)), collect(type.parameters)
    else
        globalref, alias_params = alias
        string(globalref.name), collect(alias_params)
    end
    isempty(params) && return base
    strs = map(p -> (p isa Type && p <: SectorRange) ? "…" : string(p), params)
    return string(base, "{", join(strs, ", "), "}")
end

function _to_blockarray(a::AbstractFusedArray{T, <:Any, N}) where {T, N}
    blens = map(blocklengths, axes(a))
    blockmat = Array{AbstractArray{T, N}, N}(undef, map(length, blens)...)
    # Unstored blocks render as `Zeros` (printed as `⋅`); stored blocks carry their data.
    for I in CartesianIndices(blockmat)
        b = Tuple(I)
        blockmat[I] = Zeros{T}(ntuple(d -> blens[d][b[d]], N)...)
    end
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        blockmat[CartesianIndex(Int.(Tuple(bI)))] =
            kron_nd(Array(sector(blk)), collect(data(blk)))
    end
    return mortar(blockmat)
end

# A rank-0 graded array is a single trivial-sector scalar block. There is no block structure to
# `mortar` (it does not support a 0-dimensional block array), so materialize the one (possibly
# unstored) block directly.
function _to_blockarray(a::AbstractFusedArray{T, <:Any, 0}) where {T}
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        return kron_nd(Array(sector(blk)), collect(data(blk)))
    end
    return fill(zero(T))
end

function Base.print_array(io::IO, a::AbstractFusedArray)
    return Base.print_array(io, _to_blockarray(a))
end

# Materialize into a dense `Array` (the generic fallback copies elementwise, which scalar-indexes).
# `_to_blockarray` reintroduces each block's structural factor (`I ⊗ reduced`), the identity for
# abelian sectors but a repeat over the irrep's quantum dimension for non-abelian ones.
Base.Array(a::AbstractFusedArray) = Array(_to_blockarray(a))

# Block-diagonal inner product: sum the inner products of the stored blocks, each a sector array
# whose own `dot` carries the quantum-dimension weight of its coupled sector (unit weight for
# abelian sectors). Matching axes mean matching allocated blocks (every allowed block is stored),
# so iterating one operand's stored indices lines up one-to-one with the other's.
function LinearAlgebra.dot(a::AbstractFusedArray, b::AbstractFusedArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    init = zero(LinearAlgebra.dot(zero(eltype(a)), zero(eltype(b))))
    return sum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.dot(view(a, I), view(b, I))
    end
end

# Block-diagonal `p`-norm: the stored blocks have disjoint support, so the `p`-th powers add (a
# `max` at `p == Inf`), and each block is a sector array carrying its own quantum-dimension weight.
# This is the `BlockSparseArrays` block reduction, with the `Inf` case handled correctly (unlike the
# `p`-sum formula, which collapses to `1` there).
function LinearAlgebra.norm(a::AbstractFusedArray, p::Real = 2)
    p > 0 || throw(ArgumentError("norm with non-positive p ($p) is not defined"))
    init = zero(float(real(eltype(a))))
    p == Inf && return maximum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.norm(view(a, I), p)
    end
    s = sum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.norm(view(a, I), p)^p
    end
    return s^inv(p)
end

# `LinearAlgebra.normalize` infers its result eltype via `typeof(first(a)/nrm)`, which scalar-indexes
# opaque block storage; route through the graded `/` instead.
function LinearAlgebra.normalize(a::AbstractFusedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# Conjugate through broadcasting, which conjugates each block and dualizes the sectors and axes
# (and folds in the fermionic leg-reversal sign). Overrides `Base`'s real-eltype short-circuit,
# which would keep the axes non-dual.
Base.conj(a::AbstractFusedArray) = conj.(a)

# `real`/`imag` act on the reduced data of each stored block, leaving the (real) structural sector
# factor untouched (`f(I ⊗ A) = I ⊗ f(A)`). Unlike `conj` they are not semilinear, so they cannot go
# through the linear-broadcast fold; each block delegates to the block-level `AbstractSectorArray`
# method.
function Base.real(a::AbstractFusedArray)
    eltype(a) <: Real && return a
    r = similar(a, real(eltype(a)))
    for I in eachblockstoredindex(a)
        copy!(view(r, I), real(view(a, I)))
    end
    return r
end
function Base.imag(a::AbstractFusedArray)
    r = similar(a, real(eltype(a)))
    for I in eachblockstoredindex(a)
        copy!(view(r, I), imag(view(a, I)))
    end
    return r
end

# Block-aware random fills: dispatch to each stored block's `rand!`/`randn!`, bypassing the generic
# `AbstractArray` fallbacks that go through (disallowed) scalar indexing. The 3-arg
# `rand!(rng, a, sp)` form is what Random's `rand!` entry points ultimately call.
function Random.rand!(rng::AbstractRNG, a::AbstractFusedArray, sp::Random.Sampler)
    for I in eachblockstoredindex(a)
        Random.rand!(rng, view(a, I), sp)
    end
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbstractFusedArray)
    for I in eachblockstoredindex(a)
        Random.randn!(rng, view(a, I))
    end
    return a
end

# ============================  matrix operations (guarded)  ============================
# Matrix / linear-algebra operations live on the matrix storage type `FusedGradedMatrix`. On a fused
# vector they would otherwise fall through to the generic `AbstractArray` methods, which scalar-index
# into a dense, non-graded result. Error instead. `transpose` is guarded on the whole
# `AbstractFusedArray` because, unlike `adjoint`, it is unavailable on the matrix too.
Base.transpose(A::AbstractFusedArray) = _matrix_op_error(transpose, A)
Base.adjoint(A::AbstractFusedVector) = _matrix_op_error(adjoint, A)
Base.:*(A::AbstractFusedVector, B::AbstractFusedVector) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::AbstractFusedVector) = _matrix_op_error(LinearAlgebra.tr, A)
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::AbstractFusedVector) = _matrix_op_error($f, A)
end

# ============================  similar_map  ============================
# A split-axes `similar_map` off a fused prototype reproduces a `FusionArray` (the external-axis
# array), the same target the graded constructors allocate. Three anchored entries: codomain-led,
# empty-codomain domain-led, and the fully empty rank-0 case (whose sector type is read from the
# prototype, since the empty axes carry none).
function TensorAlgebra.similar_map(
        ::AbstractFusedArray, ::Type{T},
        axes_codomain::Tuple{GradedOneTo, Vararg{GradedOneTo}},
        axes_domain::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        ::AbstractFusedArray, ::Type{T},
        axes_codomain::Tuple{}, axes_domain::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        prototype::AbstractFusedArray, ::Type{T},
        axes_codomain::Tuple{}, axes_domain::Tuple{}
    ) where {T}
    return FusionArray{T, sectortype(prototype)}(undef, axes_codomain, axes_domain)
end
