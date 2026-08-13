"""
    AbstractFusedGradedArray{T,S,N} <: AbstractArray{T,N}

Supertype of the fused (coupled-sector-block) graded arrays, [`FusedGradedMatrix`](@ref) and
[`FusedGradedVector`](@ref). Holds the code shared between the two.
"""
abstract type AbstractFusedGradedArray{T, S, N} <: AbstractArray{T, N} end
const AbstractFusedGradedMatrix{T, S} = AbstractFusedGradedArray{T, S, 2}
const AbstractFusedGradedVector{T, S} = AbstractFusedGradedArray{T, S, 1}

using BlockArrays: mortar
using FillArrays: Zeros

# The block storage type is the datatype of the blocks, so a concrete fused array only needs to
# define `blocktype`.
datatype(::Type{T}) where {T <: AbstractFusedGradedArray} = datatype(blocktype(T))
datatype(a::AbstractFusedGradedArray) = datatype(typeof(a))
sectortype(::Type{<:AbstractFusedGradedArray{T, S}}) where {T, S} = S

# Storage accessors (internal, not a public interface yet): the sector → block-data map, and the
# codomain/domain sector → reduced-data-length maps (the reduced/degeneracy dimension per sector, not
# the sector's quantum dimension). Each storage variant implements these, so the shared matrix algebra
# reads through them instead of raw fields, and a lazy adjoint or diagonal factor needs no faked fields.
function sectordata end
function sectordatalengths_codomain end
function sectordatalengths_domain end

# Codomain / domain axis groups, recovered from `biaxes` (the split `bispace` builds). Uniform across
# the fused family, matching `FusionArray`'s external-axis accessors of the same name.
axes_codomain(a::AbstractFusedGradedArray) = codomain(biaxes(a))
axes_domain(a::AbstractFusedGradedArray) = domain(biaxes(a))

function isblockdiagonal(A::AbstractFusedGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

# Diagonal when block-diagonal and every stored (diagonal) block is itself diagonal. Checking blocks
# avoids densifying a block-sparse matrix through the generic dense-slicing fallback.
function LinearAlgebra.isdiag(A::AbstractFusedGradedMatrix)
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
#  `AbstractFusedGradedArray` implements, so both fused subtypes are covered.
#  These only touch stored (symmetry-allowed) blocks, so a nonzero `fill!`
#  value leaves the forbidden positions at zero.
# ---------------------------------------------------------------------------

function TensorAlgebra.scale!(a::AbstractFusedGradedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

# The `LinearAlgebra` spelling of blockwise scaling (the generic fallback
# scalar-indexes).
LinearAlgebra.rmul!(a::AbstractFusedGradedArray, β::Number) = TensorAlgebra.scale!(a, β)
LinearAlgebra.lmul!(β::Number, a::AbstractFusedGradedArray) = TensorAlgebra.scale!(a, β)

function TensorAlgebra.zero!(a::AbstractFusedGradedArray)
    for bI in eachblockstoredindex(a)
        zero!(view(a, bI))
    end
    return a
end

function Base.fill!(a::AbstractFusedGradedArray, v)
    for bI in eachblockstoredindex(a)
        fill!(view(a, bI), v)
    end
    return a
end

# Linear-combination arithmetic, as `.`-broadcasts over the stored blocks. `FusionArray` defines its
# own split-preserving versions in `fusionarray.jl`.
Base.:+(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray) = a .+ b
Base.:-(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray) = a .- b
Base.:*(a::AbstractFusedGradedArray, x::Number) = a .* x
Base.:*(x::Number, a::AbstractFusedGradedArray) = x .* a
Base.:/(a::AbstractFusedGradedArray, x::Number) = a ./ x

# ---------------------------------------------------------------------------
#  Display — render through a BlockArrays block array. BlockArrays draws the
#  block grid; unstored blocks become `Zeros`, which print as `⋅`.
# ---------------------------------------------------------------------------

# Compact type name for the summary line. The buffer-backed fused arrays carry `{T,S,D,V}` — element,
# sector, block-view, and storage-buffer types. Only the element `T` and the storage buffer `V` are
# informative in the header (the sector is spelled out in the `Dim` lines below, and the block-view `D`
# is a derived reshape/view of the buffer), so keep those two and elide the middle to `…`.
function summary_typename(type::Type{<:AbstractFusedGradedArray})
    alias = Base.make_typealias(type)
    base, params = if isnothing(alias)
        string(nameof(type)), collect(type.parameters)
    else
        globalref, alias_params = alias
        string(globalref.name), collect(alias_params)
    end
    isempty(params) && return base
    strs = if length(params) <= 2
        map(string, params)
    else
        [string(first(params)), "…", string(last(params))]
    end
    return string(base, "{", join(strs, ", "), "}")
end

function _to_blockarray(a::AbstractFusedGradedArray{T, <:Any, N}) where {T, N}
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
            kron_nd(collect(data(blk)), Array(sector(blk)))
    end
    return mortar(blockmat)
end

# A rank-0 graded array is a single trivial-sector scalar block. There is no block structure to
# `mortar` (it does not support a 0-dimensional block array), so materialize the one (possibly
# unstored) block directly.
function _to_blockarray(a::AbstractFusedGradedArray{T, <:Any, 0}) where {T}
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        return kron_nd(collect(data(blk)), Array(sector(blk)))
    end
    return fill(zero(T))
end

function Base.print_array(io::IO, a::AbstractFusedGradedArray)
    return Base.print_array(io, _to_blockarray(a))
end

# Materialize into a dense `Array` (the generic fallback copies elementwise, which scalar-indexes).
# `_to_blockarray` reintroduces each block's structural factor (`reduced ⊗ I`), the identity for
# abelian sectors but a repeat over the irrep's quantum dimension for non-abelian ones.
Base.Array(a::AbstractFusedGradedArray) = Array(_to_blockarray(a))

# Block-diagonal inner product: sum the inner products of the stored blocks, each a sector array
# whose own `dot` carries the quantum-dimension weight of its coupled sector (unit weight for
# abelian sectors). Matching axes mean matching allocated blocks (every allowed block is stored),
# so iterating one operand's stored indices lines up one-to-one with the other's.
function LinearAlgebra.dot(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray)
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
function LinearAlgebra.norm(a::AbstractFusedGradedArray, p::Real = 2)
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
function LinearAlgebra.normalize(a::AbstractFusedGradedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# `conj` would dualize the sectors and axes, flipping the first axis to dual, which the fused storage
# types disallow. Conjugate the `FusionArray` (or matricize) instead.
Base.conj(a::AbstractFusedGradedArray) = throw_flips_first_axis(conj, a)

# `real`/`imag` act on the reduced data of each stored block, leaving the (real) structural sector
# factor untouched (`f(I ⊗ A) = I ⊗ f(A)`). Unlike `conj` they are not semilinear, so they cannot go
# through the linear-broadcast fold; each block delegates to the block-level `AbstractSectorArray`
# method.
function Base.real(a::AbstractFusedGradedArray)
    eltype(a) <: Real && return a
    r = similar(a, real(eltype(a)))
    for I in eachblockstoredindex(a)
        copy!(view(r, I), real(view(a, I)))
    end
    return r
end
function Base.imag(a::AbstractFusedGradedArray)
    r = similar(a, real(eltype(a)))
    for I in eachblockstoredindex(a)
        copy!(view(r, I), imag(view(a, I)))
    end
    return r
end

# Block-aware random fills: dispatch to each stored block's `rand!`/`randn!`, bypassing the generic
# `AbstractArray` fallbacks that go through (disallowed) scalar indexing. The 3-arg
# `rand!(rng, a, sp)` form is what Random's `rand!` entry points ultimately call.
function Random.rand!(rng::AbstractRNG, a::AbstractFusedGradedArray, sp::Random.Sampler)
    for I in eachblockstoredindex(a)
        Random.rand!(rng, view(a, I), sp)
    end
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbstractFusedGradedArray)
    for I in eachblockstoredindex(a)
        Random.randn!(rng, view(a, I))
    end
    return a
end

# ============================  matrix operations (guarded)  ============================
# Matrix / linear-algebra operations live on the matrix storage type `FusedGradedMatrix`. On a fused
# vector they would otherwise fall through to the generic `AbstractArray` methods, which scalar-index
# into a dense, non-graded result. Error instead. `transpose` is guarded on the whole
# `AbstractFusedGradedArray` because, unlike `adjoint`, it is unavailable on the matrix too.
Base.transpose(A::AbstractFusedGradedArray) = _matrix_op_error(transpose, A)
Base.adjoint(A::AbstractFusedGradedVector) = _matrix_op_error(adjoint, A)
Base.:*(A::AbstractFusedGradedVector, B::AbstractFusedGradedVector) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::AbstractFusedGradedVector) = _matrix_op_error(LinearAlgebra.tr, A)
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::AbstractFusedGradedVector) = _matrix_op_error($f, A)
end

# ============================  similar_map  ============================
# A split-axes `similar_map` off a fused prototype reproduces a `FusionArray` (the external-axis
# array), the same target the graded constructors allocate. Three anchored entries: codomain-led,
# empty-codomain domain-led, and the fully empty rank-0 case (whose sector type is read from the
# prototype, since the empty axes carry none).
function TensorAlgebra.similar_map(
        ::AbstractFusedGradedArray, ::Type{T},
        axes_codomain::Tuple{GradedOneTo, Vararg{GradedOneTo}},
        axes_domain::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        ::AbstractFusedGradedArray, ::Type{T},
        axes_codomain::Tuple{}, axes_domain::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    ) where {T}
    return FusionArray{T}(undef, axes_codomain, axes_domain)
end
function TensorAlgebra.similar_map(
        prototype::AbstractFusedGradedArray, ::Type{T},
        axes_codomain::Tuple{}, axes_domain::Tuple{}
    ) where {T}
    return FusionArray{T, sectortype(prototype)}(undef, axes_codomain, axes_domain)
end

# ============================  matrix algebra  ============================
# Block-wise matrix operations on the abstract fused graded matrix. They read the storage through the
# `sectordata` / `sectordatalengths_codomain` / `sectordatalengths_domain` accessors (which every
# storage variant provides), so a lazy adjoint or a diagonal factor works as an operand without
# materializing. The mutated operand may be any fused graded matrix; an incompatible one (e.g. a
# diagonal destination for a non-diagonal product) fails at the block level.

function TensorAlgebra.check_input(
        ::typeof(*),
        A::AbstractFusedGradedMatrix,
        B::AbstractFusedGradedMatrix
    )
    axes(A, 2) == dual(axes(B, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    return nothing
end

function TensorAlgebra.check_input(
        ::typeof(mul!),
        C::AbstractFusedGradedMatrix, A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix
    )
    check_input(*, A, B)
    axes(C, 1) == axes(A, 1) || throw(DimensionMismatch())
    axes(C, 2) == axes(B, 2) || throw(DimensionMismatch())
    return nothing
end

function LinearAlgebra.mul!(
        C::AbstractFusedGradedMatrix, A::AbstractFusedGradedMatrix,
        B::AbstractFusedGradedMatrix,
        α::Number, β::Number
    )
    check_input(mul!, C, A, B)
    dA, dB = sectordata(A), sectordata(B)
    for (s, c) in pairs(sectordata(C))
        if haskey(dA, s) && haskey(dB, s)
            mul!(c, dA[s], dB[s], α, β)
        else
            iszero(β) ? fill!(c, β) : scale!(c, β)
        end
    end
    return C
end

function allocate_output(
        ::typeof(*),
        A::AbstractFusedGradedMatrix,
        B::AbstractFusedGradedMatrix
    )
    cod = sectordatalengths_codomain(A)
    dom = sectordatalengths_domain(B)
    Tout = Base.promote_op(*, eltype(A), eltype(B))
    return FusedGradedMatrix{Tout}(undef, cod, dom)
end

function Base.:(*)(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    check_input(*, A, B)
    C = allocate_output(*, A, B)
    return mul!(C, A, B)
end

# MatrixAlgebraKit's SVD-based `left_orth!` / `right_orth!` fold the singular values into the
# orthogonal factor in place with `lmul!(S, C)` / `rmul!(C, S)`, where `S` is the (diagonal)
# singular-value matrix. The scalar-argument `lmul!` / `rmul!` above do not cover this two-matrix
# form, so define it block-wise: each stored sector block delegates to the `LinearAlgebra` method for
# that block pair, an in-place row / column scaling for the diagonal `S` blocks the factorizations
# feed in. The `check_input(mul!, ...)` call validates the contracted axes and that the product fits
# the mutated operand (the operand plays the role of the `mul!` destination `C`: `B` for `lmul!`, `A`
# for `rmul!`), so the block sectors line up by construction.
function LinearAlgebra.lmul!(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    check_input(mul!, B, A, B)
    dA = sectordata(A)
    for (s, b) in pairs(sectordata(B))
        LinearAlgebra.lmul!(dA[s], b)
    end
    return B
end
function LinearAlgebra.rmul!(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    check_input(mul!, A, A, B)
    dB = sectordata(B)
    for (s, a) in pairs(sectordata(A))
        LinearAlgebra.rmul!(a, dB[s])
    end
    return A
end

# Compare coupled-sector blocks directly instead of falling back to element-wise iteration (which
# would index forbidden blocks). Equal axes guarantee the same stored (allowed) sectors. Broad enough
# to compare a lazy adjoint or diagonal factor (whose blocks are lazy views).
function Base.:(==)(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    axes(A) == axes(B) || return false
    dA, dB = sectordata(A), sectordata(B)
    return all(dA[c] == dB[c] for c in keys(dA))
end
