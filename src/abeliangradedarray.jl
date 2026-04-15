# ===========================================================================
#  AbelianGradedArray — dict-of-keys graded array with GradedOneTo axes
# ===========================================================================

"""
    AbelianGradedArray{T,N,D<:AbstractArray{T,N},S<:SectorRange} <: AbstractGradedArray{T,N}

A graded array that stores non-zero blocks in a dictionary keyed by block indices.
Each axis is a [`GradedOneTo`](@ref) carrying sectors, sector lengths, and a dual flag.

Blocks are stored as plain dense arrays of type `D` (default `Array{T,N}`).
Accessing a block via `a[Block(i,j)]` returns a [`AbelianSectorArray`](@ref) wrapping the data
with the appropriate sectors.
"""
struct AbelianGradedArray{T, N, D <: AbstractArray{T, N}, S <: SectorRange} <:
    AbstractGradedArray{T, N}
    blockdata::Dict{NTuple{N, Int}, D}
    axes::NTuple{N, GradedOneTo{S}}
end

# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

# Fully-parameterized undef constructor: finds allowed blocks, allocates, calls inner.
# (allowedblocks is defined in fusion.jl)
function AbelianGradedArray{T, N, D, S}(
        ::UndefInitializer, axs::NTuple{N, GradedOneTo{S}}
    ) where {T, N, D <: AbstractArray{T, N}, S <: SectorRange}
    block_axes = map(eachdataaxis, axs)
    function allocate_block(bk)
        bk_inds = Int.(Tuple(bk))
        return similar(D, ntuple(d -> block_axes[d][bk_inds[d]], Val(N)))
    end
    bks = allowedblocks(axs)
    blockdata = Dict{NTuple{N, Int}, D}(
        Int.(Tuple(bk)) => allocate_block(bk) for bk in bks
    )
    return AbelianGradedArray{T, N, D, S}(blockdata, axs)
end

# Convenience: infer D = Array{T,N} and S from axes.
function AbelianGradedArray{T}(
        ::UndefInitializer, axs::NTuple{N, GradedOneTo{S}}
    ) where {T, N, S <: SectorRange}
    return AbelianGradedArray{T, N, Array{T, N}, S}(undef, axs)
end

function AbelianGradedArray{T}(
        init::UndefInitializer, axs::Vararg{GradedOneTo{S}, N}
    ) where {T, N, S <: SectorRange}
    return AbelianGradedArray{T}(init, axs)
end

# Convert any `AbstractGradedMatrix` (e.g. a `FusedGradedMatrix`) to an
# `AbelianGradedArray` with the same axes and stored blocks. Assumes each
# allowed block of the target is also stored in `m` — every `similar`
# allocation is overwritten by the loop below, so no `zero!` is needed.
function AbelianGradedArray(m::AbstractGradedMatrix)
    a = similar(m, axes(m))
    for I in eachblockstoredindex(m)
        a[Data(I)] = view(m, Data(I))
    end
    return a
end

# ---------------------------------------------------------------------------
#  AbstractArray interface
# ---------------------------------------------------------------------------

Base.size(a::AbelianGradedArray) = map(length, a.axes)
Base.axes(a::AbelianGradedArray) = a.axes
function BlockSparseArrays.blocktype(
        ::Type{<:AbelianGradedArray{T, N, D, S}}
    ) where {T, N, D, S}
    return AbelianSectorArray{T, N, D, S}
end
BlockSparseArrays.blocktype(a::AbelianGradedArray) = BlockSparseArrays.blocktype(typeof(a))

# ---------------------------------------------------------------------------
#  view (primitive): returns AbelianSectorArray sharing data with blockdata
# ---------------------------------------------------------------------------

function Base.view(a::AbelianGradedArray{T, N}, I::Block{N}) where {T, N}
    bk = Int.(Tuple(I))
    haskey(a.blockdata, bk) || error("Block $bk is not stored.")
    sects = ntuple(d -> sectors(axes(a, d))[bk[d]], Val(N))
    return AbelianSectorArray(sects, a.blockdata[bk])
end

# ---------------------------------------------------------------------------
#  blocks — lazy view delegating to view (following BlockArrays convention)
# ---------------------------------------------------------------------------

"""
    AbelianBlocks{T,N,A<:AbelianGradedArray{T,N}} <: AbstractArray{AbelianSectorArray,N}

Lazy view of an `AbelianGradedArray`'s block storage, following the BlockArrays
convention: `getindex` delegates to `view(parent, Block.(I)...)` (shares data),
`setindex!` copies into the existing view.
"""
struct AbelianBlocks{T, N, A <: AbelianGradedArray{T, N}} <:
    AbstractArray{AbelianSectorArray, N}
    parent::A
end

BlockArrays.blocks(a::AbelianGradedArray) = AbelianBlocks(a)
Base.size(b::AbelianBlocks) = Tuple(blocklength.(axes(b.parent)))

function Base.getindex(b::AbelianBlocks{T, N}, I::Vararg{Int, N}) where {T, N}
    return view(b.parent, Block.(I)...)
end

function Base.setindex!(
        b::AbelianBlocks{T, N}, value, I::Vararg{Int, N}
    ) where {T, N}
    dest = view(b.parent, Block.(I)...)
    copyto!(dest, value)
    return b
end

# ---------------------------------------------------------------------------
#  Splitting getindex: each I[d][k] = Block(b)[r] means dest block k comes
#  from source block b at subrange r. Inverse of the merging getindex.
# ---------------------------------------------------------------------------

# Ported from the old GradedArray getindex(::AbstractVector{<:BlockIndexRange{1}}...).
function Base.getindex(
        a::AbelianGradedArray{T, N}, I::Vararg{AbstractVector{<:BlockIndexRange{1}}, N}
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
        a::AbelianGradedArray{T, N}, I::Vararg{AbstractBlockVector{<:Block{1}}, N}
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

function BlockSparseArrays.eachblockstoredindex(a::AbelianGradedArray{T, N}) where {T, N}
    return (Block(k) for k in keys(a.blockdata))
end

# Implement the `SparseArraysBase` interface on `AbelianBlocks` (the lazy
# block view) so that `storedlength(blocks(a))` — and by extension
# `blockstoredlength(a)` — reflects the dict-of-keys storage rather than
# treating every slot as stored.
function SparseArraysBase.eachstoredindex(b::AbelianBlocks{T, N}) where {T, N}
    return (CartesianIndex(k) for k in keys(b.parent.blockdata))
end
SparseArraysBase.storedvalues(b::AbelianBlocks) = values(b.parent.blockdata)
function SparseArraysBase.isstored(b::AbelianBlocks{T, N}, I::Vararg{Int, N}) where {T, N}
    return haskey(b.parent.blockdata, I)
end

# ---------------------------------------------------------------------------
#  similar
# ---------------------------------------------------------------------------

# similar with GradedOneTo axes: allocates all allowed blocks (uninitialized).
# Defined on AbstractGradedArray so FusedGradedMatrix can use it too.
function Base.similar(
        a::AbstractGradedArray,
        ::Type{T},
        axes::Tuple{GradedOneTo{S}, Vararg{GradedOneTo{S}}}
    ) where {T, S}
    N = length(axes)
    D = datatype(BlockSparseArrays.blocktype(a))
    data_ax_types = Tuple{ntuple(d -> dataaxistype(typeof(axes[d])), Val(N))...}
    D_N = Base.promote_op(similar, D, Type{T}, data_ax_types)
    D_N′ = isconcretetype(D_N) ? D_N : Array{T, N}
    return AbelianGradedArray{T, N, D_N′, S}(undef, axes)
end
function Base.similar(
        a::AbstractGradedArray{T}, axes::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    return similar(a, T, axes)
end
function Base.similar(a::AbelianGradedArray{T}, ::Type{Tv}) where {T, Tv}
    return similar(a, Tv, axes(a))
end
function Base.similar(a::AbelianGradedArray{T}) where {T}
    return similar(a, T)
end

# ---------------------------------------------------------------------------
#  sectortype
# ---------------------------------------------------------------------------

sectortype(::Type{<:AbelianGradedArray{T, N, D, S}}) where {T, N, D, S} = S

# ---------------------------------------------------------------------------
#  permutedims
# ---------------------------------------------------------------------------

function Base.permutedims(a::AbelianGradedArray{<:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> axes(a)[perm[i]], Val(N))
    a_dest = FI.zero!(similar(a, dest_axes))
    return permutedims!(a_dest, a, perm)
end

function Base.permutedims!(
        y::AbelianGradedArray{<:Any, N}, x::AbelianGradedArray{<:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# ---------------------------------------------------------------------------
#  fill! / zero! / scale!
# ---------------------------------------------------------------------------

scale!(a::AbstractArray, β::Number) = (a .*= β; a)
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

function Base.fill!(a::AbelianGradedArray, v)
    iszero(v) || throw(
        ArgumentError("fill! with nonzero value is not supported for AbelianGradedArray")
    )
    return FI.zero!(a)
end

# ---------------------------------------------------------------------------
#  Matrix multiplication (block-diagonal)
# ---------------------------------------------------------------------------

const AbelianGradedVector{T, D, S} = AbelianGradedArray{T, 1, D, S}
const AbelianGradedMatrix{T, D, S} = AbelianGradedArray{T, 2, D, S}

# ---------------------------------------------------------------------------
#  show
# ---------------------------------------------------------------------------

function Base.summary(io::IO, a::AbelianGradedArray)
    block_str = join(map(g -> string(blocklength(g)), axes(a)), "×")
    size_str = join(map(string, size(a)), "×")
    nstored = blockstoredlength(a)
    print(io, block_str, "-blocked ", size_str, " ", typeof(a))
    print(io, " with ", nstored, " stored block", nstored == 1 ? "" : "s")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", a::AbelianGradedArray)
    summary(io, a)
    println(io, ":")
    for (d, g) in pairs(axes(a))
        print(io, "  Dim $d: ")
        show(io, g)
        println(io)
    end
    isempty(a) && return nothing
    Base.print_array(io, a)
    return nothing
end

function Base.show(io::IO, a::AbelianGradedArray)
    block_str = join(map(g -> string(blocklength(g)), axes(a)), "×")
    size_str = join(map(string, size(a)), "×")
    print(io, block_str, "-blocked ", size_str, " ", typeof(a))
    return nothing
end

# ---------------------------------------------------------------------------
#  zeros / rand  (allowedblocks is defined in fusion.jl)
# ---------------------------------------------------------------------------

"""
    zeros(T, axs::GradedOneTo...)

Create an `AbelianGradedArray{T}` with all allowed (zero-flux) blocks filled with zeros.
"""
function Base.zeros(::Type{T}, axs::GradedOneTo{S}...) where {T, S <: SectorRange}
    return FI.zero!(AbelianGradedArray{T}(undef, axs...))
end

function Base.zeros(axs::GradedOneTo...)
    return zeros(Float64, axs...)
end

function Base.zeros(
        ::Type{T}, axs::NTuple{N, GradedOneTo{S}}
    ) where {T, N, S <: SectorRange}
    return zeros(T, axs...)
end
