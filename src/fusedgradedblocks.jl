# ---------------------------------------------------------------------------
#  blocks — lazy views over a fused graded array's stored blocks
# ---------------------------------------------------------------------------

# A `FusedGradedMatrix` is block-diagonal in sector space, so most of its block grid is empty:
# its `blocks` view is a sparse array over the allocated (symmetry-allowed) blocks, mirroring
# `AbelianBlocks`. A stored entry is `view(parent, Block(I)...)` (shares data); an unstored entry
# is a symmetry-forbidden block and errors.
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

# Return materialized `Vector`s (not lazy generators), paired in the same block order, so
# `storedvalues` satisfies the `AbstractVector` guard and `concatenate_sparse!` can zip the two.
function eachstoredindex(::IndexCartesian, b::FusedGradedMatrixBlocks)
    return [CartesianIndex(Int.(Tuple(bI))) for bI in eachblockstoredindex(b.parent)]
end
function storedvalues(b::FusedGradedMatrixBlocks)
    return [view(b.parent, bI) for bI in eachblockstoredindex(b.parent)]
end

# Block `(i, j)` is stored only when its codomain and domain sectors coincide and that sector has
# an allocated block.
function isstored(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    cod, dom = axis_codomain(b.parent), axis_domain(b.parent)
    (i in 1:blocklength(cod) && j in 1:blocklength(dom)) || return false
    s_cod = sectors(cod)[i]
    s_dom = sectors(dom)[j]
    # Stored iff codomain and domain share the sector: `s_cod` is a codomain sector by construction,
    # so a stored block needs it in the domain too.
    return s_cod == s_dom && haskey(sectordatalengths(dom), s_cod)
end

# A stored entry is the block view, sharing data with the parent.
function getstoredindex(b::FusedGradedMatrixBlocks, i::Int, j::Int)
    return view(b.parent, Block(i), Block(j))
end
function setstoredindex!(b::FusedGradedMatrixBlocks, value, i::Int, j::Int)
    copy_sector!(view(b.parent, Block(i), Block(j)), value)
    return b
end
# An unstored index is a symmetry-forbidden block, not a lazily-omitted zero, so reading or
# writing one is a structural error.
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

# A `FusedGradedVector` allocates one block per axis sector, so its blocks are dense: the view is
# a plain `AbstractVector` of block views (sharing data), with no forbidden entries.
struct FusedGradedVectorBlocks{T, S, D, A <: FusedGradedVector{T, S}} <:
    AbstractVector{FusedSectorVector{T, S, D}}
    parent::A
end
function BlockArrays.blocks(v::FusedGradedVector)
    return FusedGradedVectorBlocks{eltype(v), sectortype(v), datatype(v), typeof(v)}(v)
end

Base.size(b::FusedGradedVectorBlocks) = (blocklength(only(axes(b.parent))),)
Base.getindex(b::FusedGradedVectorBlocks, i::Int) = view(b.parent, Block(i))
