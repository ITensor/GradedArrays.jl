using Base.Broadcast: Broadcast as BC
using TensorAlgebra: TensorAlgebra

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

function Base.similar(bc::BC.Broadcasted{<:SectorStyle}, elt::Type, ax)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa SectorArray, bc′.args)]
    return ofsector(arg, similar(arg.data, elt))
end

function Base.copyto!(dest::SectorArray, bc::BC.Broadcasted{<:SectorStyle})
    lb = TensorAlgebra.tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("SectorArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
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

# TODO: Rename `graded_similar` to `similar_graded` or fold it into `similar`
# entirely once the follow-up allocator cleanup is ready.
function graded_similar(
        a::GradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return similar(a, elt, ax)
end

function Base.copyto!(dest::GradedArray, bc::BC.Broadcasted{<:GradedStyle})
    lb = TensorAlgebra.tryflattenlinear(bc)
    isnothing(lb) &&
        throw(ArgumentError("GradedArray broadcasting requires linear operations"))
    return copyto!(dest, lb)
end
