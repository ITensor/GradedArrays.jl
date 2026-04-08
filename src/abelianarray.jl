# ===========================================================================
#  AbelianArray — dict-of-keys graded array with GradedUnitRange axes
# ===========================================================================

"""
    AbelianArray{T,N,I<:TKS.Sector,D<:AbstractArray{T,N}} <: AbstractGradedArray{T,N}

A graded array that stores non-zero blocks in a dictionary keyed by block indices.
Each axis is a [`GradedUnitRange`](@ref) carrying sector labels, multiplicities, and a dual flag.

Blocks are stored as plain dense arrays of type `D` (default `Array{T,N}`).
Accessing a block via `a[Block(i,j)]` returns a [`SectorArray`](@ref) wrapping the data
with the appropriate sector labels and dual flags.
"""
struct AbelianArray{T, N, I <: TKS.Sector, D <: AbstractArray{T, N}} <:
    AbstractGradedArray{T, N}
    axes::NTuple{N, GradedUnitRange{I}}
    blockdata::Dict{NTuple{N, Int}, D}
end

# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

function AbelianArray{T}(
        ::UndefInitializer, axs::NTuple{N, GradedUnitRange{I}}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T, N, I, Array{T, N}}(axs, Dict{NTuple{N, Int}, Array{T, N}}())
end

function AbelianArray{T}(
        init::UndefInitializer, axs::Vararg{GradedUnitRange{I}, N}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T}(init, axs)
end

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

"""
    _block_length(g::GradedUnitRange, k::Int)

Total length of block `k` in axis `g`: quantum dimension times multiplicity.
"""
function _block_length(g::GradedUnitRange, k::Int)
    return quantum_dimension(sectors(g)[k]) * sector_multiplicities(g)[k]
end

# ---------------------------------------------------------------------------
#  AbstractArray interface
# ---------------------------------------------------------------------------

Base.size(a::AbelianArray) = map(length, a.axes)
Base.axes(a::AbelianArray) = a.axes

# ---------------------------------------------------------------------------
#  view (primitive): returns SectorArray sharing data with blockdata
# ---------------------------------------------------------------------------

function _wrap_block(a::AbelianArray{T, N}, bk::NTuple{N, Int}, data) where {T, N}
    block_sectors = ntuple(d -> sectors(a.axes[d])[bk[d]], Val(N))
    return SectorArray(block_sectors, data)
end

# view: returns a SectorArray sharing data (errors for unstored blocks)
function Base.view(a::AbelianArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    haskey(a.blockdata, bk) || error("Block $bk is not stored. Use view! to create it.")
    return _wrap_block(a, bk, a.blockdata[bk])
end
function Base.view(a::AbelianArray{T, N}, I::Block{N}) where {T, N}
    return view(a, Block.(Tuple(I))...)
end

# view!: get or create, then view
function BlockSparseArrays.view!(
        a::AbelianArray{T, N}, I::Vararg{Block{1}, N}
    ) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    if !haskey(a.blockdata, bk)
        block_dims = ntuple(d -> _block_length(a.axes[d], bk[d]), Val(N))
        a.blockdata[bk] = zeros(T, block_dims)
    end
    return _wrap_block(a, bk, a.blockdata[bk])
end
function BlockSparseArrays.view!(a::AbelianArray{<:Any, N}, I::Block{N}) where {N}
    return BlockSparseArrays.view!(a, Tuple(I)...)
end

# ---------------------------------------------------------------------------
#  blocks — lazy view delegating to view (following BlockArrays convention)
# ---------------------------------------------------------------------------

"""
    AbelianBlocks{T,N,A<:AbelianArray{T,N}} <: AbstractArray{SectorArray,N}

Lazy view of an `AbelianArray`'s block storage, following the BlockArrays
convention: `getindex` delegates to `view(parent, Block.(I)...)` (shares data),
`setindex!` copies into the existing view.
"""
struct AbelianBlocks{T, N, A <: AbelianArray{T, N}} <: AbstractArray{SectorArray, N}
    parent::A
end

BlockArrays.blocks(a::AbelianArray) = AbelianBlocks(a)
Base.size(b::AbelianBlocks) = Tuple(blocklength.(b.parent.axes))

function Base.getindex(b::AbelianBlocks{T, N}, I::Vararg{Int, N}) where {T, N}
    return view(b.parent, Block.(I)...)
end

function Base.setindex!(
        b::AbelianBlocks{T, N}, value, I::Vararg{Int, N}
    ) where {T, N}
    # Use view! to get-or-create, then copyto! (following BlockSparseArrays pattern)
    dest = view!(b.parent, Block.(I)...)
    copyto!(dest, value)
    return b
end

# ---------------------------------------------------------------------------
#  getindex / setindex! on AbelianArray with Block — convenience wrappers
# ---------------------------------------------------------------------------

# getindex: returns a copy (unstored blocks return zeros)
function Base.getindex(a::AbelianArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    if haskey(a.blockdata, bk)
        return copy(view(a, I...))
    else
        block_dims = ntuple(d -> _block_length(a.axes[d], bk[d]), Val(N))
        return _wrap_block(a, bk, zeros(T, block_dims))
    end
end
function Base.getindex(a::AbelianArray{T, N}, I::Block{N}) where {T, N}
    return a[Block.(Tuple(I))...]
end

# setindex!: replaces block data
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
#  Splitting getindex: each I[d][k] = Block(b)[r] means dest block k comes
#  from source block b at subrange r. Inverse of the merging getindex.
# ---------------------------------------------------------------------------

# Ported from the old GradedArray getindex(::AbstractVector{<:BlockIndexRange{1}}...).
function Base.getindex(
        a::AbelianArray{T, N}, I::Vararg{AbstractVector{<:BlockIndexRange{1}}, N}
    ) where {T, N}
    ax_dest = ntuple(d -> a.axes[d][I[d]], Val(N))
    a_dest = AbelianArray{T}(undef, ax_dest)
    # Map source block b → list of (dest BlockIndexRange, src subrange).
    src_to_dests = ntuple(Val(N)) do d
        key_type = Block{1, Int}
        dest_bir_type = Base.promote_op(getindex, key_type, Base.OneTo{Int})
        val_type = Tuple{dest_bir_type, UnitRange{Int}}
        dict = Dict{key_type, Vector{val_type}}()
        for k in eachindex(I[d])
            bir = I[d][k]
            b = Block(Int(bir.block))
            r = only(bir.indices)
            push!(get!(dict, b, val_type[]), (Block(k)[Base.axes1(r)], r))
        end
        return dict
    end
    for bI_src in eachblockstoredindex(a)
        src_tuple = Tuple(bI_src)
        all(d -> haskey(src_to_dests[d], src_tuple[d]), 1:N) || continue
        dest_refs = ntuple(d -> src_to_dests[d][src_tuple[d]], Val(N))
        for combo in Iterators.product(dest_refs...)
            src_r = ntuple(d -> combo[d][2], Val(N))
            src_data = view(a[bI_src], src_r...)
            iszero(src_data) && continue
            dest_b = Block(ntuple(d -> only(Tuple(combo[d][1].block)), Val(N)))
            a_dest_b = @view!(a_dest[dest_b])
            dest_r = ntuple(d -> only(combo[d][1].indices), Val(N))
            copyto!(view(a_dest_b, dest_r...), src_data)
        end
    end
    return a_dest
end

# ---------------------------------------------------------------------------
#  Merging getindex: reindex by block permutation/merge
# ---------------------------------------------------------------------------

# Merging: each I[d] groups source blocks into destination blocks.
# Follows the same pattern as the old GradedArray getindex(::AbstractBlockVector...).
function Base.getindex(
        a::AbelianArray{T, N}, I::Vararg{AbstractBlockVector{<:Block{1}}, N}
    ) where {T, N}
    ax_dest = ntuple(d -> a.axes[d][I[d]], Val(N))
    a_dest = AbelianArray{T}(undef, ax_dest)
    ax = a.axes
    # Map source Block → BlockIndexRange encoding dest block + subrange within it
    src_to_dest = ntuple(Val(N)) do d
        key_type = eltype(I[d])
        range_type = UnitRange{Int}
        val_type = Base.promote_op(getindex, key_type, range_type)
        dict = Dict{key_type, val_type}()
        for j in eachindex(blocks(I[d]))
            sub_blocks = I[d][Block(j)]
            start = 1
            for b in sub_blocks
                blen = blocklengths(ax[d])[Int(b)]
                r = Base.OneTo(blen) .+ (start - 1)
                dict[b] = Block(j)[r]
                start += blen
            end
        end
        return dict
    end
    for bI_src in eachblockstoredindex(a)
        src_tuple = Tuple(bI_src)
        dest_info = ntuple(d -> src_to_dest[d][src_tuple[d]], Val(N))
        dest_b = Block(map(di -> only(Tuple(di.block)), dest_info))
        a_dest_b = @view!(a_dest[dest_b])
        dest_r = map(di -> only(di.indices), dest_info)
        copyto!(view(a_dest_b, dest_r...), a[bI_src])
    end
    return a_dest
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
        ::AbelianArray{<:Any, <:Any, I}, ::Type{S}, axs::NTuple{M, GradedUnitRange{I}}
    ) where {S, M, I}
    return AbelianArray{S}(undef, axs)
end

# ---------------------------------------------------------------------------
#  sector_type
# ---------------------------------------------------------------------------

sector_type(::Type{<:AbelianArray{T, N, I}}) where {T, N, I} = SectorRange{I}

# ---------------------------------------------------------------------------
#  permutedims
# ---------------------------------------------------------------------------

function Base.permutedims(a::AbelianArray{<:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> a.axes[perm[i]], Val(N))
    a_dest = AbelianArray{eltype(a)}(undef, dest_axes)
    return permutedims!(a_dest, a, perm)
end

function Base.permutedims!(
        y::AbelianArray{<:Any, N}, x::AbelianArray{<:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ---------------------------------------------------------------------------
#  fill! / zero!
# ---------------------------------------------------------------------------

function FI.zero!(a::AbelianArray)
    for bk in keys(a.blockdata)
        fill!(a.blockdata[bk], zero(eltype(a)))
    end
    return a
end

function Base.fill!(a::AbelianArray, v)
    iszero(v) || throw(
        ArgumentError("fill! with nonzero value is not supported for AbelianArray")
    )
    return FI.zero!(a)
end

# ---------------------------------------------------------------------------
#  Matrix multiplication (block-diagonal)
# ---------------------------------------------------------------------------

const AbelianMatrix{T, I, D} = AbelianArray{T, 2, I, D}

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
