# ===========================================================================
#  AbelianArray — dict-of-keys graded array with GradedIndices axes
# ===========================================================================

"""
    AbelianArray{T,N,I<:TKS.Sector,D<:AbstractArray{T,N}} <: AbstractGradedArray{T,N}

A graded array that stores non-zero blocks in a dictionary keyed by block indices.
Each axis is a [`GradedIndices`](@ref) carrying sector labels, multiplicities, and a dual flag.

Blocks are stored as plain dense arrays of type `D` (default `Array{T,N}`).
Accessing a block via `a[Block(i,j)]` returns a [`SectorArray`](@ref) wrapping the data
with the appropriate sector labels and dual flags.
"""
struct AbelianArray{T, N, I <: TKS.Sector, D <: AbstractArray{T, N}} <:
    AbstractGradedArray{T, N}
    axes::NTuple{N, GradedIndices{I}}
    blockdata::Dict{NTuple{N, Int}, D}
end

# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

function AbelianArray{T}(
        ::UndefInitializer, axs::NTuple{N, GradedIndices{I}}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T, N, I, Array{T, N}}(axs, Dict{NTuple{N, Int}, Array{T, N}}())
end

function AbelianArray{T}(
        init::UndefInitializer, axs::Vararg{GradedIndices{I}, N}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T}(init, axs)
end

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

"""
    _block_length(g::GradedIndices, k::Int)

Total length of block `k` in axis `g`: quantum dimension times multiplicity.
"""
function _block_length(g::GradedIndices, k::Int)
    return TKS.dim(labels(g)[k]) * sector_multiplicities(g)[k]
end

# ---------------------------------------------------------------------------
#  AbstractArray interface
# ---------------------------------------------------------------------------

Base.size(a::AbelianArray) = map(length, a.axes)
Base.axes(a::AbelianArray) = a.axes

"""
    _find_block_and_offset(g::GradedIndices, idx::Int)

Given a linear integer index into the total dimension of `g`, return `(block_index, local_offset)`
where `block_index` is the 1-based block number and `local_offset` is the 1-based position
within that block.
"""
function _find_block_and_offset(g::GradedIndices, idx::Int)
    cumulative = 0
    for k in 1:BlockArrays.blocklength(g)
        blen = _block_length(g, k)
        if idx <= cumulative + blen
            return (k, idx - cumulative)
        end
        cumulative += blen
    end
    throw(BoundsError())
end

Base.@propagate_inbounds function Base.getindex(
        a::AbelianArray{T, N}, I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(a, I...)
    bk_off = ntuple(d -> _find_block_and_offset(a.axes[d], I[d]), Val(N))
    bk = ntuple(d -> bk_off[d][1], Val(N))
    off = ntuple(d -> bk_off[d][2], Val(N))
    if haskey(a.blockdata, bk)
        return @inbounds a.blockdata[bk][off...]
    else
        return zero(T)
    end
end

Base.@propagate_inbounds function Base.setindex!(
        a::AbelianArray{T, N}, v, I::Vararg{Int, N}
    ) where {T, N}
    @boundscheck checkbounds(a, I...)
    bk_off = ntuple(d -> _find_block_and_offset(a.axes[d], I[d]), Val(N))
    bk = ntuple(d -> bk_off[d][1], Val(N))
    off = ntuple(d -> bk_off[d][2], Val(N))
    if !haskey(a.blockdata, bk)
        block_dims = ntuple(d -> _block_length(a.axes[d], bk[d]), Val(N))
        a.blockdata[bk] = zeros(T, block_dims)
    end
    @inbounds a.blockdata[bk][off...] = v
    return a
end

# ---------------------------------------------------------------------------
#  Block indexing
# ---------------------------------------------------------------------------

function Base.getindex(a::AbelianArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    block_labels = ntuple(d -> labels(a.axes[d])[bk[d]], Val(N))
    block_isdual = ntuple(d -> isdual(a.axes[d]), Val(N))
    if haskey(a.blockdata, bk)
        return SectorArray(block_labels, block_isdual, a.blockdata[bk])
    else
        block_dims = ntuple(d -> _block_length(a.axes[d], bk[d]), Val(N))
        return SectorArray(block_labels, block_isdual, zeros(T, block_dims))
    end
end

# Single Block{N} argument: splat to N Block{1} arguments
function Base.getindex(a::AbelianArray{T, N}, I::Block{N}) where {T, N}
    return a[Block.(Tuple(I))...]
end

function Base.setindex!(a::AbelianArray{T, N}, value, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    raw = value isa SectorArray ? value.data : value
    a.blockdata[bk] = convert(Array{T, N}, raw)
    return a
end

function Base.setindex!(a::AbelianArray{T, N}, value, I::Block{N}) where {T, N}
    return setindex!(a, value, Block.(Tuple(I))...)
end

# ---------------------------------------------------------------------------
#  eachblockstoredindex
# ---------------------------------------------------------------------------

function BlockSparseArrays.eachblockstoredindex(a::AbelianArray{T, N}) where {T, N}
    return (Block(k) for k in keys(a.blockdata))
end

# ---------------------------------------------------------------------------
#  similar
# ---------------------------------------------------------------------------

function Base.similar(a::AbelianArray{T, N, I}) where {T, N, I}
    return AbelianArray{T}(undef, a.axes)
end

function Base.similar(a::AbelianArray{<:Any, N, I}, ::Type{S}) where {S, N, I}
    return AbelianArray{S}(undef, a.axes)
end

function Base.similar(
        ::AbelianArray{<:Any, <:Any, I}, ::Type{S}, axs::NTuple{M, GradedIndices{I}}
    ) where {S, M, I}
    return AbelianArray{S}(undef, axs)
end

# ---------------------------------------------------------------------------
#  sector_type
# ---------------------------------------------------------------------------

sector_type(::Type{<:AbelianArray{T, N, I}}) where {T, N, I} = SectorRange{I}

# ---------------------------------------------------------------------------
#  show
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", a::AbelianArray{T, N}) where {T, N}
    block_str = join(map(g -> string(BlockArrays.blocklength(g)), a.axes), "×")
    size_str = join(map(string, size(a)), "×")
    nstored = length(a.blockdata)
    print(io, block_str, "-blocked ", size_str, " AbelianArray{", T, "}")
    print(io, " with ", nstored, " stored block", nstored == 1 ? "" : "s")
    return nothing
end

function Base.show(io::IO, a::AbelianArray{T, N}) where {T, N}
    block_str = join(map(g -> string(BlockArrays.blocklength(g)), a.axes), "×")
    size_str = join(map(string, size(a)), "×")
    print(io, block_str, "-blocked ", size_str, " AbelianArray{", T, "}")
    return nothing
end
