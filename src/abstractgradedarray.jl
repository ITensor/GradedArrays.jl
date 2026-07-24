"""
    AbstractGradedArray{T,S,N} <: AbstractArray{T,N}

Abstract supertype for graded (symmetry-structured) arrays whose axes carry sector labels.
Concrete subtypes include [`AbelianGradedArray`](@ref) and [`FusedGradedMatrix`](@ref).
"""
abstract type AbstractGradedArray{T, S, N} <: AbstractArray{T, N} end
const AbstractGradedMatrix{T, S} = AbstractGradedArray{T, S, 2}

function isblockdiagonal(A::AbstractGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

# Trace: sum the traces of the diagonal blocks (same block position on both axes). Each block
# view carries its sector, so its `tr` includes that sector's quantum dimension. This is the
# plain matricized trace, with no fermionic twist sign.
function LinearAlgebra.tr(A::AbstractGradedMatrix)
    return sum(eachblockstoredindex(A); init = zero(eltype(A))) do bI
        row, col = Tuple(bI)
        return row == col ? LinearAlgebra.tr(view(A, bI)) : zero(eltype(A))
    end
end

# Whether a block is stored (allocated), following the `SparseArraysBase.isstored(a, ::Block)`
# interface `BlockSparseArrays` uses: delegate to the block container's element `isstored`.
function SparseArraysBase.isstored(
        a::AbstractGradedArray{<:Any, <:Any, N}, I::Block{N}
    ) where {N}
    return SparseArraysBase.isstored(blocks(a), Int.(Tuple(I))...)
end

using BlockArrays: block, blockindex, findblockindex
# Well-defined only for unique (abelian) fusion, where the trivial structural factor lets a
# coordinate pick out a single element.
function Base.getindex(a::AbstractGradedArray, I1::Int, I_rest::Vararg{Int})
    require_unique_fusion(a)
    I = (I1, I_rest...)
    @boundscheck checkbounds(a, I...)
    bis = map(findblockindex, axes(a), I)
    b = Block(map(bi -> Int(block(bi)), bis))
    SparseArraysBase.isstored(a, b) || return zero(eltype(a))
    return view(a, b)[map(blockindex, bis)...]
end
function Base.setindex!(a::AbstractGradedArray, v, I1::Int, I_rest::Vararg{Int})
    require_unique_fusion(a)
    I = (I1, I_rest...)
    @boundscheck checkbounds(a, I...)
    bis = map(findblockindex, axes(a), I)
    b = Block(map(bi -> Int(block(bi)), bis))
    SparseArraysBase.isstored(a, b) ||
        error("cannot set element at $(I): it lies in a symmetry-forbidden block.")
    view(a, b)[map(blockindex, bis)...] = v
    return a
end

# There is no generic block-aware `adjoint`: the lazy `Adjoint` wrapper falls back to
# LinearAlgebra's scalar-indexing path, which silently produces a dense, non-graded result. Error
# instead. `FusedGradedMatrix` defines its own working `adjoint` (more specific, so it still wins).
function Base.adjoint(::AbstractGradedArray)
    return error("`adjoint` is not supported for this graded array")
end

# ---------------------------------------------------------------------------
#  Block indexing interface
#
#  Concrete subtypes must implement:
#    view(a::ConcreteType, ::Block{N})  → sector-wrapped view (e.g. SectorMatrix)
#
#  Everything else is derived here.
# ---------------------------------------------------------------------------

function Base.view(a::AbstractGradedArray{T, <:Any, N}, I::Vararg{Block{1}, N}) where {T, N}
    return view(a, Block(Int.(I)))
end

function Base.getindex(a::AbstractGradedArray{T, <:Any, N}, I::Block{N}) where {T, N}
    return copy(view(a, I))
end
function Base.getindex(
        a::AbstractGradedArray{T, <:Any, N},
        I::Vararg{Block{1}, N}
    ) where {T, N}
    return a[Block(Int.(I))]
end
# Disambiguate the N=1 case: route through the `Block{N}` method to avoid recursion.
Base.getindex(a::AbstractGradedArray{T, <:Any, 1}, I::Block{1}) where {T} = copy(view(a, I))

function Base.setindex!(
        a::AbstractGradedArray{<:Any, <:Any, N},
        value,
        I::Block{N}
    ) where {N}
    return setindex!(a, value, Tuple(I)...)
end
function Base.setindex!(
        a::AbstractGradedArray{<:Any, <:Any, N}, value, I::Vararg{Block{1}, N}
    ) where {N}
    copy!(view(a, I...), value)
    return a
end
function Base.setindex!(a::AbstractGradedArray{<:Any, <:Any, 1}, value, I::Block{1})
    copy!(view(a, I), value)
    return a
end

# ---------------------------------------------------------------------------
#  Data indexing — raw block data without sector wrappers
#
#  Built on top of Block view: view(a, Data(I)) = data(view(a, Block(I)))
# ---------------------------------------------------------------------------

function Base.view(a::AbstractGradedArray{T, <:Any, N}, I::Data{N}) where {T, N}
    return data(view(a, Block(I)))
end

function Base.getindex(a::AbstractGradedArray{T, <:Any, N}, I::Data{N}) where {T, N}
    return copy(view(a, I))
end

function Base.setindex!(
        a::AbstractGradedArray{<:Any, <:Any, N}, value::AbstractArray{<:Any, N}, I::Data{N}
    ) where {N}
    view(a, I) .= value
    return a
end

# ---------------------------------------------------------------------------
#  Accessors
# ---------------------------------------------------------------------------

# The block storage type is the datatype of the blocks, so a concrete graded array only
# needs to define `blocktype`.
datatype(::Type{T}) where {T <: AbstractGradedArray} = datatype(blocktype(T))
datatype(a::AbstractGradedArray) = datatype(typeof(a))
sectortype(::Type{<:AbstractGradedArray{T, S}}) where {T, S} = S

# ---------------------------------------------------------------------------
#  fill! / zero! / scale! — block-wise over the stored blocks
#
#  Defined once via the `eachblockstoredindex`/`view` interface every
#  `AbstractGradedArray` implements, so every concrete subtype is covered.
#  These only touch stored (symmetry-allowed) blocks, so a nonzero `fill!`
#  value leaves the forbidden positions at zero.
# ---------------------------------------------------------------------------

function TensorAlgebra.scale!(a::AbstractGradedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

# The `LinearAlgebra` spelling of blockwise scaling (the generic fallback
# scalar-indexes).
LinearAlgebra.rmul!(a::AbstractGradedArray, β::Number) = TensorAlgebra.scale!(a, β)
LinearAlgebra.lmul!(β::Number, a::AbstractGradedArray) = TensorAlgebra.scale!(a, β)

function TensorAlgebra.zero!(a::AbstractGradedArray)
    for bI in eachblockstoredindex(a)
        zero!(view(a, bI))
    end
    return a
end

function Base.fill!(a::AbstractGradedArray, v)
    for bI in eachblockstoredindex(a)
        fill!(view(a, bI), v)
    end
    return a
end

# ---------------------------------------------------------------------------
#  Display — render through a BlockArrays block array. BlockArrays draws the
#  block grid; unstored blocks become `Zeros`, which print as `⋅`.
# ---------------------------------------------------------------------------

using BlockArrays: mortar
using FillArrays: Zeros

# Compact type name for the summary line. The sector parameter is dotted (it is spelled out in
# full in the `Dim` lines below, so repeating it in the header only adds noise); the element,
# order, and storage parameters are kept. `make_typealias` recovers the `Vector`/`Matrix` alias
# names and leaves the order `N` explicit for higher-rank arrays.
function summary_typename(type::Type{<:AbstractGradedArray})
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

function _to_blockarray(a::AbstractGradedArray{T, <:Any, N}) where {T, N}
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
function _to_blockarray(a::AbstractGradedArray{T, <:Any, 0}) where {T}
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        return kron_nd(Array(sector(blk)), collect(data(blk)))
    end
    return fill(zero(T))
end

function Base.print_array(io::IO, a::AbstractGradedArray)
    return Base.print_array(io, _to_blockarray(a))
end

# Materialize into a dense `Array` (the generic fallback copies elementwise, which scalar-indexes).
# `_to_blockarray` reintroduces each block's structural factor (`I ⊗ reduced`), the identity for
# abelian sectors but a repeat over the irrep's quantum dimension for non-abelian ones.
Base.Array(a::AbstractGradedArray) = Array(_to_blockarray(a))

# Block-diagonal inner product: sum the inner products of the stored blocks, each a sector array
# whose own `dot` carries the quantum-dimension weight of its coupled sector (unit weight for
# abelian sectors). Matching axes mean matching allocated blocks (every allowed block is stored),
# so iterating one operand's stored indices lines up one-to-one with the other's.
function LinearAlgebra.dot(a::AbstractGradedArray, b::AbstractGradedArray)
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
function LinearAlgebra.norm(a::AbstractGradedArray, p::Real = 2)
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

# Conjugate through broadcasting, which conjugates each block and dualizes the sectors and axes
# (and folds in the fermionic leg-reversal sign). Overrides `Base`'s real-eltype short-circuit,
# which would keep the axes non-dual.
Base.conj(a::AbstractGradedArray) = conj.(a)

# Block-aware random fills: dispatch to each stored block's `rand!`/`randn!`, bypassing the generic
# `AbstractArray` fallbacks that go through (disallowed) scalar indexing. The 3-arg
# `rand!(rng, a, sp)` form is what Random's `rand!` entry points ultimately call.
function Random.rand!(rng::AbstractRNG, a::AbstractGradedArray, sp::Random.Sampler)
    for I in eachblockstoredindex(a)
        Random.rand!(rng, view(a, I), sp)
    end
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbstractGradedArray)
    for I in eachblockstoredindex(a)
        Random.randn!(rng, view(a, I))
    end
    return a
end
