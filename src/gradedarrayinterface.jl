# =============================================================================
#  Block and scalar indexing, derived from the `view(a, ::Block)` method each type defines.
# =============================================================================

using BlockArrays: Block, BlockIndexRange, block, blockindex, blocks, findblockindex

for AT in (:GradedArray, :AbstractFusedGradedArray)
    @eval begin
        function isstored(a::$AT{<:Any, <:Any, N}, I::Block{N}) where {N}
            return isstored(blocks(a), Int.(Tuple(I))...)
        end

        function Base.getindex(a::$AT, I1::Int, I_rest::Vararg{Int})
            assert_scalar_indexing()
            require_unique_fusion(a)
            I = (I1, I_rest...)
            @boundscheck checkbounds(a, I...)
            bis = map(findblockindex, axes(a), I)
            b = Block(map(bi -> Int(block(bi)), bis))
            isstored(a, b) || return zero(eltype(a))
            # Internal block indexing, allowed even when block indexing is disabled.
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

        # ---- Block indexing ----

        function Base.view(a::$AT{T, <:Any, N}, I::Vararg{Block{1}, N}) where {T, N}
            return view(a, Block(Int.(I)))
        end

        # Through the `Block` view so the result is a sector array rather than the dense
        # `SubArray` of Base's generic path.
        function Base.view(a::$AT{T, <:Any, N}, I::BlockIndexRange{N}) where {T, N}
            return view(view(a, block(I)), I.indices...)
        end
        # At least two arguments, since a single one would recurse and none would clash with the
        # N=0 method.
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

        # ---- Data indexing ----

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
# Each type defines `Array{T, N}`, and the other forms forward to it.
for AT in (:GradedArray, :AbstractFusedGradedArray, :AbstractSectorArray)
    @eval begin
        Base.Array(a::$AT) = Array{eltype(a), ndims(a)}(a)
        Base.Array{T}(a::$AT) where {T} = Array{T, ndims(a)}(a)
        # Covers `Vector(a)` and `Matrix(a)`.
        Base.Array{<:Any, N}(a::$AT{<:Any, <:Any, N}) where {N} = Array{eltype(a), N}(a)
    end
end
