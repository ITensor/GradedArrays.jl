using Base.Broadcast: Broadcast as BC
using FillArrays: Zeros, fillsimilar
using TensorAlgebra: TensorAlgebra, *ₗ, +ₗ, -ₗ, /ₗ, conjed

struct SectorStyle{I, N} <: BC.AbstractArrayStyle{N} end
SectorStyle{I, N}(::Val{M}) where {I, N, M} = SectorStyle{I, M}()

function BC.BroadcastStyle(::Type{T}) where {T <: SectorDelta}
    return SectorStyle{sector_type(T), ndims(T)}()
end
function BC.BroadcastStyle(::Type{T}) where {T <: SectorArray}
    return SectorStyle{sector_type(T), ndims(T)}()
end
BC.BroadcastStyle(style::SectorStyle{I, N}, ::BC.DefaultArrayStyle{0}) where {I, N} = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::SectorStyle{I, N}) where {I, N} = style
function BC.BroadcastStyle(
        style1::SectorStyle{I, N},
        style2::SectorStyle{I, N}
    ) where {I, N}
    return style1
end

function set_data(a::SectorArray, data::AbstractArray)
    axes(data) == axes(a.data) ||
        throw(ArgumentError("linear broadcasting must preserve SectorArray axes"))
    return SectorArray(sectors(a), data)
end
ofsector(a::SectorArray, data) = set_data(a, data)

function Base.Broadcast.materialize(a::SectorArray)
    return ofsector(a, Base.Broadcast.materialize(a.data))
end

function TensorAlgebra.:+ₗ(a::SectorArray, b::SectorArray)
    _check_add_axes(a, b)
    return ofsector(a, a.data +ₗ b.data)
end

function TensorAlgebra.:*ₗ(α::Number, a::SectorArray)
    return ofsector(a, α *ₗ a.data)
end
TensorAlgebra.:*ₗ(a::SectorArray, α::Number) = α *ₗ a
function TensorAlgebra.conjed(a::SectorArray)
    return ofsector(a, TensorAlgebra.conjed(a.data))
end

function BC.broadcasted(style::SectorStyle, f, args...)
    return TensorAlgebra.broadcasted_linear(style, f, args...)
end

struct GradedStyle{I, N, B <: BC.AbstractArrayStyle{N}} <: BC.AbstractArrayStyle{N}
    blockstyle::B
end
function GradedStyle{I, N}(blockstyle::BC.AbstractArrayStyle{N}) where {I, N}
    return GradedStyle{I, N, typeof(blockstyle)}(blockstyle)
end
function GradedStyle{I, N, B}() where {I, N, B <: BC.AbstractArrayStyle{N}}
    return GradedStyle{I, N}(B())
end
GradedStyle{I, N}(::Val{M}) where {I, N, M} = GradedStyle{I, M}(BC.DefaultArrayStyle{M}())

blockstyle(style::GradedStyle) = style.blockstyle

function BC.BroadcastStyle(arraytype::Type{<:GradedArray{<:Any, N, I}}) where {N, I}
    return GradedStyle{I, N}(BC.BroadcastStyle(blocktype(arraytype)))
end
BC.BroadcastStyle(style::GradedStyle, ::BC.DefaultArrayStyle{0}) = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::GradedStyle) = style
function BC.BroadcastStyle(::GradedStyle{I, N}, ::BC.DefaultArrayStyle{N}) where {I, N}
    return BC.DefaultArrayStyle{N}()
end
function BC.BroadcastStyle(::BC.DefaultArrayStyle{N}, ::GradedStyle{I, N}) where {I, N}
    return BC.DefaultArrayStyle{N}()
end
function BC.BroadcastStyle(
        style1::GradedStyle{I, N},
        style2::GradedStyle{I, N}
    ) where {I, N}
    style = BC.result_style(blockstyle(style1), blockstyle(style2))
    return GradedStyle{I, N}(style)
end

# TODO: Revisit `similar(::Broadcasted{<:GradedStyle}, ...)`.
# The current implementation still allocates based on a selected broadcast argument.
# Follow-up work should make this style-driven, deriving the block type from the
# graded/block style and allocating a default `BlockSparseArray` directly.
function Base.similar(bc::BC.Broadcasted{<:GradedStyle}, elt::Type, ax)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractArray, bc′.args)]
    return graded_similar(arg, elt, ax)
end

function _check_add_axes(a::AbstractArray, b::AbstractArray)
    axes(a) == axes(b) ||
        throw(
        ArgumentError("linear broadcasting requires matching axes")
    )
    return nothing
end

function lazyblock(a::GradedArray{<:Any, N}, I::Vararg{Block{1}, N}) where {N}
    if isstored(a, I...)
        return blocks(a)[Int.(I)...]
    else
        block_ax = map((ax, i) -> eachblockaxis(ax)[Int(i)], axes(a), I)
        return fillsimilar(Zeros{eltype(a)}(block_ax), block_ax)
    end
end
lazyblock(a::GradedArray, I::Block) = lazyblock(a, Tuple(I)...)

TensorAlgebra.@scaledarray_type ScaledGradedArray
TensorAlgebra.@scaledarray ScaledGradedArray
TensorAlgebra.@conjarray_type ConjGradedArray
TensorAlgebra.@conjarray ConjGradedArray
TensorAlgebra.@addarray_type AddGradedArray
TensorAlgebra.@addarray AddGradedArray

const LazyGradedArray = Union{
    GradedArray, ScaledGradedArray, ConjGradedArray, AddGradedArray,
}

function TensorAlgebra.BroadcastStyle_scaled(arrayt::Type{<:ScaledGradedArray})
    return BC.BroadcastStyle(TensorAlgebra.unscaled_type(arrayt))
end
function TensorAlgebra.BroadcastStyle_conj(arrayt::Type{<:ConjGradedArray})
    return BC.BroadcastStyle(TensorAlgebra.conjed_type(arrayt))
end
function TensorAlgebra.BroadcastStyle_add(arrayt::Type{<:AddGradedArray})
    args_type = TensorAlgebra.addends_type(arrayt)
    return Base.promote_op(BC.combine_styles, fieldtypes(args_type)...)()
end

function lazyblock(a::ScaledGradedArray, I::Block)
    return TensorAlgebra.coeff(a) *ₗ lazyblock(TensorAlgebra.unscaled(a), I)
end
function lazyblock(a::ConjGradedArray, I::Block)
    return conjed(lazyblock(conjed(a), I))
end
function lazyblock(a::AddGradedArray, I::Block)
    return +ₗ(map(Base.Fix2(lazyblock, I), TensorAlgebra.addends(a))...)
end

# TODO: Use `eachblockstoredindex` directly for lazy graded wrappers and delete the
# `graded_eachblockstoredindex` helper once that refactor is split into its own PR.
graded_eachblockstoredindex(a::GradedArray) = collect(eachblockstoredindex(a))
function graded_eachblockstoredindex(a::ScaledGradedArray)
    return graded_eachblockstoredindex(TensorAlgebra.unscaled(a))
end
graded_eachblockstoredindex(a::ConjGradedArray) = graded_eachblockstoredindex(conjed(a))
function graded_eachblockstoredindex(a::AddGradedArray)
    return unique!(vcat(map(graded_eachblockstoredindex, TensorAlgebra.addends(a))...))
end

# TODO: Rename `graded_similar` to `similar_graded` or fold it into `similar`
# entirely once the follow-up allocator cleanup is ready.
function graded_similar(
        a::GradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return similar(a, elt, ax)
end
function graded_similar(
        a::ScaledGradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return graded_similar(TensorAlgebra.unscaled(a), elt, ax)
end
function graded_similar(
        a::ConjGradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return graded_similar(conjed(a), elt, ax)
end
function graded_similar(
        a::AddGradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    style = BC.combine_styles(TensorAlgebra.addends(a)...)
    bc = BC.Broadcasted(style, +, TensorAlgebra.addends(a))
    return similar(bc, elt, ax)
end

function copy_lazygraded(a::LazyGradedArray)
    c = graded_similar(a, eltype(a), axes(a))
    for I in graded_eachblockstoredindex(a)
        c[I] = lazyblock(a, I)
    end
    return c
end

function TensorAlgebra.:+ₗ(a::LazyGradedArray, b::LazyGradedArray)
    _check_add_axes(a, b)
    return AddGradedArray(a, b)
end
TensorAlgebra.:*ₗ(α::Number, a::GradedArray) = ScaledGradedArray(α, a)
TensorAlgebra.conjed(a::GradedArray) = ConjGradedArray(a)

Base.copy(a::ScaledGradedArray) = copy_lazygraded(a)
Base.copy(a::ConjGradedArray) = copy_lazygraded(a)
Base.copy(a::AddGradedArray) = copy_lazygraded(a)
Base.Array(a::ScaledGradedArray) = Array(copy(a))
Base.Array(a::ConjGradedArray) = Array(copy(a))
Base.Array(a::AddGradedArray) = Array(copy(a))

function BC.broadcasted(style::GradedStyle, f, args...)
    return TensorAlgebra.broadcasted_linear(style, f, args...)
end
