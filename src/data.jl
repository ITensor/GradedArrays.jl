# Like `BlockArrays.Block{N}`, but indexing a graded array with `Data(i, j, ...)` accesses the
# raw data array of that block.
struct Data{N}
    n::NTuple{N, Int}
end
Data(n::Vararg{Int, N}) where {N} = Data{N}(n)
BlockArrays.Block(I::Data) = Block(I.n)
Data(I::Block) = Data(Int.(Tuple(I)))
Base.Tuple(I::Data) = I.n
