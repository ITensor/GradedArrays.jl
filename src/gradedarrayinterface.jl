# =============================================================================
#  Shared block-indexing infrastructure.
#
#  `GradedArray` and the fused arrays (`AbstractFusedGradedArray`) both implement the block-indexing
#  primitive `view(a, ::Block)`, and the rest of the block/scalar indexing surface is derived from
#  it identically for both. That shared derivation is the reason to define it once here, by an
#  `@eval` loop over the two types, rather than the definitions merely coinciding. The `GradedArray`
#  side is only well-defined for unique (abelian) fusion, guarded by `require_unique_fusion`. This
#  file is included after `gradedarray.jl` so both types exist.
# =============================================================================

using BlockArrays: Block, BlockIndexRange, block, blockindex, blocks, findblockindex

for AT in (:GradedArray, :AbstractFusedGradedArray)
    @eval begin
        # Whether a block is stored (allocated), following the `isstored(a, ::Block)`
        # interface `BlockSparseArrays` uses: delegate to the block container's element `isstored`.
        function isstored(a::$AT{<:Any, <:Any, N}, I::Block{N}) where {N}
            return isstored(blocks(a), Int.(Tuple(I))...)
        end

        # Scalar indexing is well-defined only for unique (abelian) fusion, where the trivial
        # structural factor lets a coordinate pick out a single element.
        function Base.getindex(a::$AT, I1::Int, I_rest::Vararg{Int})
            assert_scalar_indexing()
            require_unique_fusion(a)
            I = (I1, I_rest...)
            @boundscheck checkbounds(a, I...)
            bis = map(findblockindex, axes(a), I)
            b = Block(map(bi -> Int(block(bi)), bis))
            isstored(a, b) || return zero(eltype(a))
            # Scalar access reaches its element through the block view; allow that internal block
            # indexing even when block indexing is otherwise disabled.
            return with_block_indexing() do
                return view(a, b)[map(blockindex, bis)...]
            end
        end
        function Base.setindex!(a::$AT, v, I1::Int, I_rest::Vararg{Int})
            assert_scalar_indexing()
            require_unique_fusion(a)
            I = (I1, I_rest...)
            @boundscheck checkbounds(a, I...)
            bis = map(findblockindex, axes(a), I)
            b = Block(map(bi -> Int(block(bi)), bis))
            isstored(a, b) ||
                error("cannot set element at $(I): it lies in a symmetry-forbidden block.")
            with_block_indexing() do
                return view(a, b)[map(blockindex, bis)...] = v
            end
            return a
        end

        # ---- Block indexing, derived from the `view(a, ::Block)` primitive each type implements ----

        function Base.view(a::$AT{T, <:Any, N}, I::Vararg{Block{1}, N}) where {T, N}
            return view(a, Block(Int.(I)))
        end

        # A `BlockIndexRange` view is the block view sliced by the within-block ranges. Routing
        # through the `Block` view (then the range sub-view) keeps the result a sector array, which
        # Base's generic `BlockSlice` path would otherwise flatten to a plain dense `SubArray`. Both
        # the combined form and the per-axis splatted form land here.
        function Base.view(a::$AT{T, <:Any, N}, I::BlockIndexRange{N}) where {T, N}
            return view(view(a, block(I)), I.indices...)
        end
        # The per-axis splatted form combines into the `BlockIndexRange{N}` above. Requiring at least
        # two arguments keeps a lone per-axis range on the combined method (a single splat would
        # rebuild the same `BlockIndexRange{1}` and recurse) and off the empty `Vararg` view at N=0.
        function Base.view(
                a::$AT, I1::BlockIndexRange{1}, I2::BlockIndexRange{1},
                Irest::BlockIndexRange{1}...
            )
            return view(a, BlockIndexRange((I1, I2, Irest...)))
        end

        function Base.getindex(a::$AT{T, <:Any, N}, I::Block{N}) where {T, N}
            return copy(view(a, I))
        end
        function Base.getindex(a::$AT{T, <:Any, N}, I::Vararg{Block{1}, N}) where {T, N}
            return a[Block(Int.(I))]
        end
        # Disambiguate the N=1 case: route through the `Block{N}` method to avoid recursion.
        Base.getindex(a::$AT{T, <:Any, 1}, I::Block{1}) where {T} = copy(view(a, I))

        # `BlockIndexRange` indexing mirrors the `Block` methods: a copy of the sector-array block
        # view (the two-argument splat rule matches `view` above).
        function Base.getindex(a::$AT{T, <:Any, N}, I::BlockIndexRange{N}) where {T, N}
            return copy(view(a, I))
        end
        function Base.getindex(
                a::$AT, I1::BlockIndexRange{1}, I2::BlockIndexRange{1},
                Irest::BlockIndexRange{1}...
            )
            return copy(view(a, I1, I2, Irest...))
        end

        function Base.setindex!(a::$AT{<:Any, <:Any, N}, value, I::Block{N}) where {N}
            return setindex!(a, value, Tuple(I)...)
        end
        function Base.setindex!(
                a::$AT{<:Any, <:Any, N}, value, I::Vararg{Block{1}, N}
            ) where {N}
            copy_sector!(view(a, I...), value)
            return a
        end
        function Base.setindex!(a::$AT{<:Any, <:Any, 1}, value, I::Block{1})
            copy_sector!(view(a, I), value)
            return a
        end

        # ---- Data indexing: raw block data without sector wrappers ----
        #  view(a, Data(I)) = data(view(a, Block(I)))

        function Base.view(a::$AT{T, <:Any, N}, I::Data{N}) where {T, N}
            return data(view(a, Block(I)))
        end
        function Base.getindex(a::$AT{T, <:Any, N}, I::Data{N}) where {T, N}
            return copy(view(a, I))
        end
        function Base.setindex!(
                a::$AT{<:Any, <:Any, N}, value::AbstractArray{<:Any, N}, I::Data{N}
            ) where {N}
            view(a, I) .= value
            return a
        end
    end
end

# =============================  dense conversions  =============================
# Each family's `Array{T, N}` method is the materialization worker (eltype converts during the
# dense pass), and everything else delegates to it, following the `Base.Array` design.
# `Vector{T}` and `Matrix{T}` are `Array{T, 1}` and `Array{T, 2}`, so the worker covers them
# directly.
for AT in (:GradedArray, :AbstractFusedGradedArray, :AbstractSectorArray)
    @eval begin
        Base.Array(a::$AT) = Array{eltype(a), ndims(a)}(a)
        Base.Array{T}(a::$AT) where {T} = Array{T, ndims(a)}(a)
        # Covers `Vector(a)` and `Matrix(a)`: they are `Array{<:Any, 1}` and `Array{<:Any, 2}`.
        Base.Array{<:Any, N}(a::$AT{<:Any, <:Any, N}) where {N} = Array{eltype(a), N}(a)
    end
end
