# ---------------------------------------------------------------------------
#  blocks
# ---------------------------------------------------------------------------

# A sparse array over the stored blocks of a `FusedGradedMatrix`. Accessing a symmetry-forbidden
# block errors.
struct FusedGradedMatrixBlocks{T, S, D, A <: FusedGradedMatrix{T, S}} <:
    AbstractMatrix{FusedSectorMatrix{T, S, D}}
    parent::A
end
function BlockArrays.blocks(m::FusedGradedMatrix)
    return FusedGradedMatrixBlocks{eltype(m), sectortype(m), datatype(m), typeof(m)}(m)
end

Base.size(b::FusedGradedMatrixBlocks) = blocklength.(axes(b.parent))
Base.getindex(b::FusedGradedMatrixBlocks, I::Int...) = getindex_sparse(b, I...)
function Base.setindex!(b::FusedGradedMatrixBlocks, value, I::Int...)
    return setindex!_sparse(b, value, I...)
end

# Materialized `Vector`s in the same block order, so `concatenate_sparse!` can zip the two.
function eachstoredindex(::IndexCartesian, b::FusedGradedMatrixBlocks)
    return [CartesianIndex(Int.(Tuple(bI))) for bI in eachblockstoredindex(b.parent)]
end
function storedvalues(b::FusedGradedMatrixBlocks)
    return [view(b.parent, bI) for bI in eachblockstoredindex(b.parent)]
end

function isstored(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    @boundscheck checkbounds(b, i, j)
    cod, dom = axis_codomain(b.parent), axis_domain(b.parent)
    s_cod = sectors(cod)[i]
    s_dom = sectors(dom)[j]
    return s_cod == s_dom && haskey(sectordatalengths(dom), s_cod)
end

function getstoredindex(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    return view(b.parent, Block(i), Block(j))
end
function setstoredindex!(b::FusedGradedMatrixBlocks, value, i::Int, j::Int)
    copy_sector!(view(b.parent, Block(i), Block(j)), value)
    return b
end
# An unstored block is symmetry-forbidden, not an omitted zero.
function getunstoredindex(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    return error("Block ($(i), $(j)) is not stored.")
end
function setunstoredindex!(
        b::FusedGradedMatrixBlocks,
        value,
        i::Int,
        j::Int
    )
    return error("Block ($(i), $(j)) is not stored.")
end

# A `FusedGradedVector` stores one block per sector, so there are no forbidden entries.
struct FusedGradedVectorBlocks{T, S, D, A <: FusedGradedVector{T, S}} <:
    AbstractVector{FusedSectorVector{T, S, D}}
    parent::A
end
function BlockArrays.blocks(v::FusedGradedVector)
    return FusedGradedVectorBlocks{eltype(v), sectortype(v), datatype(v), typeof(v)}(v)
end

Base.size(b::FusedGradedVectorBlocks) = (blocklength(only(axes(b.parent))),)
Base.getindex(b::FusedGradedVectorBlocks, i::Int) = view(b.parent, Block(i))
