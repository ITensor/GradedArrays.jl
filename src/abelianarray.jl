# ===========================================================================
#  AbelianArray — dict-of-keys graded array with GradedOneTo axes
# ===========================================================================

"""
    AbelianArray{T,N,D<:AbstractArray{T,N},I<:TKS.Sector} <: AbstractGradedArray{T,N}

A graded array that stores non-zero blocks in a dictionary keyed by block indices.
Each axis is a [`GradedOneTo`](@ref) carrying sector labels, multiplicities, and a dual flag.

Blocks are stored as plain dense arrays of type `D` (default `Array{T,N}`).
Accessing a block via `a[Block(i,j)]` returns a [`SectorArray`](@ref) wrapping the data
with the appropriate sector labels and dual flags.
"""
struct AbelianArray{T, N, D <: AbstractArray{T, N}, I <: TKS.Sector} <:
    AbstractGradedArray{T, N}
    blockdata::Dict{NTuple{N, Int}, D}
    axes::NTuple{N, GradedOneTo{I}}
end

# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

# Forward declaration — implementation in fusion.jl (needs fusion machinery)
function allowedblocks end

# Fully-parameterized undef constructor: finds allowed blocks, allocates, calls inner.
function AbelianArray{T, N, D, I}(
        ::UndefInitializer, axs::NTuple{N, GradedOneTo{I}}
    ) where {T, N, D <: AbstractArray{T, N}, I <: TKS.Sector}
    bks = allowedblocks(axs)
    blockdata = Dict{NTuple{N, Int}, D}(
        Int.(Tuple(bk)) =>
            D(undef, ntuple(d -> blocklengths(axs[d])[Int(Tuple(bk)[d])], Val(N)))
            for bk in bks
    )
    return AbelianArray{T, N, D, I}(blockdata, axs)
end

# Convenience: infer D = Array{T,N} and I from axes.
function AbelianArray{T}(
        ::UndefInitializer, axs::NTuple{N, GradedOneTo{I}}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T, N, Array{T, N}, I}(undef, axs)
end

function AbelianArray{T}(
        init::UndefInitializer, axs::Vararg{GradedOneTo{I}, N}
    ) where {T, N, I <: TKS.Sector}
    return AbelianArray{T}(init, axs)
end

# ---------------------------------------------------------------------------
#  AbstractArray interface
# ---------------------------------------------------------------------------

Base.size(a::AbelianArray) = map(length, a.axes)
Base.axes(a::AbelianArray) = a.axes
function BlockSparseArrays.blocktype(
        ::Type{<:AbelianArray{T, N, D, I}}
    ) where {T, N, D, I}
    return SectorArray{T, N, D, I}
end
BlockSparseArrays.blocktype(a::AbelianArray) = BlockSparseArrays.blocktype(typeof(a))

# ---------------------------------------------------------------------------
#  view (primitive): returns SectorArray sharing data with blockdata
# ---------------------------------------------------------------------------

# view: returns a SectorArray sharing data (errors for unstored blocks)
function Base.view(a::AbelianArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    haskey(a.blockdata, bk) || error("Block $bk is not stored. Use view! to create it.")
    sects = ntuple(d -> sectors(axes(a, d))[bk[d]], Val(N))
    return SectorArray(sects, a.blockdata[bk])
end
function Base.view(a::AbelianArray{T, N}, I::Block{N}) where {T, N}
    return view(a, Tuple(I)...)
end

# view!: get or create, then view
function BlockSparseArrays.view!(
        a::AbelianArray{T, N}, I::Vararg{Block{1}, N}
    ) where {T, N}
    bk = ntuple(d -> Int(I[d]), Val(N))
    if !haskey(a.blockdata, bk)
        block_dims = ntuple(d -> blocklengths(axes(a, d))[bk[d]], Val(N))
        a.blockdata[bk] = zeros(T, block_dims)
    end
    sects = ntuple(d -> sectors(axes(a, d))[bk[d]], Val(N))
    return SectorArray(sects, a.blockdata[bk])
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
Base.size(b::AbelianBlocks) = Tuple(blocklength.(axes(b.parent)))

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
        block_dims = ntuple(d -> blocklengths(axes(a, d))[bk[d]], Val(N))
        sects = ntuple(d -> sectors(axes(a, d))[bk[d]], Val(N))
        return SectorArray(sects, zeros(T, block_dims))
    end
end
function Base.getindex(a::AbelianArray{T, N}, I::Block{N}) where {T, N}
    return a[Tuple(I)...]
end

# setindex!: Block{N} unpacks to Vararg{Block{1}, N} (following BlockArrays convention)
function Base.setindex!(a::AbelianArray{<:Any, N}, value, I::Block{N}) where {N}
    return setindex!(a, value, Tuple(I)...)
end

# Primitive: get-or-create block view, then broadcast in.
# Handles both SectorArray and raw data values.
function Base.setindex!(
        a::AbelianArray{<:Any, N}, value::AbstractArray{<:Any, N}, I::Vararg{Block{1}, N}
    ) where {N}
    BlockSparseArrays.view!(a, I...) .= value
    return a
end

# ---------------------------------------------------------------------------
#  Splitting getindex: each I[d][k] = Block(b)[r] means dest block k comes
#  from source block b at subrange r. Inverse of the merging getindex.
# ---------------------------------------------------------------------------

# Ported from the old GradedArray getindex(::AbstractVector{<:BlockIndexRange{1}}...).
function Base.getindex(
        a::AbelianArray{T, N}, I::Vararg{AbstractVector{<:BlockIndexRange{1}}, N}
    ) where {T, N}
    ax_dest = ntuple(d -> axes(a, d)[I[d]], Val(N))
    a_dest = FI.zero!(similar(a, ax_dest))
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
            a_dest_b = view(a_dest, dest_b)
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
    ax_dest = ntuple(d -> axes(a, d)[I[d]], Val(N))
    a_dest = FI.zero!(similar(a, ax_dest))
    ax = axes(a)
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
        a_dest_b = view(a_dest, dest_b)
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

# similar with GradedOneTo axes: allocates all allowed blocks (uninitialized).
# Defined on AbstractGradedArray so FusedSectorMatrix can use it too.
function Base.similar(
        a::AbstractGradedArray,
        ::Type{S},
        axes::Tuple{GradedOneTo{I}, Vararg{GradedOneTo{I}}}
    ) where {S, I}
    N = length(axes)
    D = datatype(BlockSparseArrays.blocktype(a))
    data_ax_types = Tuple{ntuple(d -> dataaxistype(typeof(axes[d])), Val(N))...}
    D_N = Base.promote_op(similar, D, Type{S}, data_ax_types)
    D_N′ = isconcretetype(D_N) ? D_N : Array{S, N}
    return AbelianArray{S, N, D_N′, I}(undef, axes)
end
function Base.similar(
        a::AbstractGradedArray{T}, axes::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    return similar(a, T, axes)
end
function Base.similar(a::AbelianArray{T}, ::Type{S}) where {T, S}
    return similar(a, S, axes(a))
end
function Base.similar(a::AbelianArray{T}) where {T}
    return similar(a, T)
end

# ---------------------------------------------------------------------------
#  sector_type
# ---------------------------------------------------------------------------

sector_type(::Type{<:AbelianArray{T, N, D, I}}) where {T, N, D, I} = SectorRange{I}

# ---------------------------------------------------------------------------
#  permutedims
# ---------------------------------------------------------------------------

function Base.permutedims(a::AbelianArray{<:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> axes(a)[perm[i]], Val(N))
    a_dest = FI.zero!(similar(a, dest_axes))
    return permutedims!(a_dest, a, perm)
end

function Base.permutedims!(
        y::AbelianArray{<:Any, N}, x::AbelianArray{<:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ---------------------------------------------------------------------------
#  fill! / zero! / scale!
# ---------------------------------------------------------------------------

scale!(a::SectorArray, β::Number) = (a.data .*= β; a)
FI.zero!(a::SectorArray) = (fill!(a.data, zero(eltype(a))); a)

function scale!(a::AbstractGradedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

function FI.zero!(a::AbstractGradedArray)
    for bI in eachblockstoredindex(a)
        FI.zero!(view(a, bI))
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

const AbelianVector{T, D, I} = AbelianArray{T, 1, D, I}
const AbelianMatrix{T, D, I} = AbelianArray{T, 2, D, I}

# ---------------------------------------------------------------------------
#  show
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", a::AbelianArray{T, N}) where {T, N}
    block_str = join(map(g -> string(blocklength(g)), axes(a)), "×")
    size_str = join(map(string, size(a)), "×")
    nstored = length(collect(eachblockstoredindex(a)))
    print(io, block_str, "-blocked ", size_str, " AbelianArray{", T, "}")
    print(io, " with ", nstored, " stored block", nstored == 1 ? "" : "s")
    return nothing
end

function Base.show(io::IO, a::AbelianArray{T, N}) where {T, N}
    block_str = join(map(g -> string(blocklength(g)), axes(a)), "×")
    size_str = join(map(string, size(a)), "×")
    print(io, block_str, "-blocked ", size_str, " AbelianArray{", T, "}")
    return nothing
end
