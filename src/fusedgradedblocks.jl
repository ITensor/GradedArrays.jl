# ---------------------------------------------------------------------------
#  blocks — lazy views over a fused graded array's stored blocks
# ---------------------------------------------------------------------------

# A `FusedGradedMatrix` is block-diagonal in sector space, so most of its block grid is empty:
# its `blocks` view is a sparse array over the allocated (symmetry-allowed) blocks, mirroring
# `AbelianBlocks`. A stored entry is `view(parent, Block(I)...)` (shares data); an unstored entry
# is a symmetry-forbidden block and errors.
struct FusedGradedMatrixBlocks{T, S, D, A <: FusedGradedMatrix{T, S, D}} <:
    AbstractSparseMatrix{SectorMatrix{T, S, D}}
    parent::A
end
BlockArrays.blocks(m::FusedGradedMatrix) = FusedGradedMatrixBlocks(m)

Base.size(b::FusedGradedMatrixBlocks) = Tuple(blocklength.(axes(b.parent)))

# Return `Vector`s (not lazy generators): the `SubArray` wrapper path in SparseArraysBase
# `filter`s over these, and `filter` is not defined for `Base.Generator`.
# TODO: make these lazy once the SparseArraysBase `filter` path handles generators.
function SparseArraysBase.eachstoredindex(::IndexCartesian, b::FusedGradedMatrixBlocks)
    return [CartesianIndex(Int.(Tuple(bI))) for bI in eachblockstoredindex(b.parent)]
end
function SparseArraysBase.storedvalues(b::FusedGradedMatrixBlocks)
    return [view(b.parent, bI) for bI in eachblockstoredindex(b.parent)]
end

# Block `(i, j)` is stored only when its codomain and domain sectors coincide and that sector has
# an allocated block.
function SparseArraysBase.isstored(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    m = b.parent
    (i in 1:length(m.codomain) && j in 1:length(m.domain)) || return false
    s_cod = gettokenvalue(keys(m.codomain), i)
    s_dom = gettokenvalue(keys(m.domain), j)
    return s_cod == s_dom && haskey(m.blocks, s_cod)
end

# A stored entry is the block view, sharing data with the parent.
function SparseArraysBase.getstoredindex(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    return view(b.parent, Block(i), Block(j))
end
function SparseArraysBase.setstoredindex!(b::FusedGradedMatrixBlocks, value, i::Int, j::Int)
    copyto!(view(b.parent, Block(i), Block(j)), value)
    return b
end
# An unstored index is a symmetry-forbidden block, not a lazily-omitted zero, so reading or
# writing one is a structural error.
function SparseArraysBase.getunstoredindex(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    return error("Block ($(i), $(j)) is not stored.")
end
function SparseArraysBase.setunstoredindex!(
        b::FusedGradedMatrixBlocks,
        value,
        i::Int,
        j::Int
    )
    return error("Block ($(i), $(j)) is not stored.")
end

# A `FusedGradedVector` allocates one block per axis sector, so its blocks are dense: the view is
# a plain `AbstractVector` of block views (sharing data), with no forbidden entries.
struct FusedGradedVectorBlocks{T, S, D, A <: FusedGradedVector{T, S, D}} <:
    AbstractVector{SectorVector{T, S, D}}
    parent::A
end
BlockArrays.blocks(v::FusedGradedVector) = FusedGradedVectorBlocks(v)

Base.size(b::FusedGradedVectorBlocks) = (blocklength(only(axes(b.parent))),)
Base.getindex(b::FusedGradedVectorBlocks, i::Int) = view(b.parent, Block(i))
