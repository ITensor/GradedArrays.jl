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

# Scalar indexing is not supported for graded arrays.
function Base.getindex(::AbstractGradedArray, ::Vararg{Int})
    return error(
        "Scalar indexing is not supported for AbstractGradedArray. Use block indexing."
    )
end
function Base.setindex!(::AbstractGradedArray, _, ::Vararg{Int})
    return error(
        "Scalar indexing is not supported for AbstractGradedArray. Use block indexing."
    )
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

function Base.print_array(io::IO, a::AbstractGradedArray)
    return Base.print_array(io, _to_blockarray(a))
end
