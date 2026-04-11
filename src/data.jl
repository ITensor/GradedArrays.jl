"""
    Data{N}

Block-data indexing type analogous to `BlockArrays.Block{N}`. Indexing a graded
array with `Data(i, j, ...)` accesses the raw data array for that block, without
sector metadata wrappers.
"""
struct Data{N}
    n::NTuple{N, Int}
end
Data(n::Vararg{Int, N}) where {N} = Data{N}(n)
BlockArrays.Block(I::Data) = Block(I.n)
Data(I::Block) = Data(Int.(Tuple(I)))
Base.Tuple(I::Data) = I.n
