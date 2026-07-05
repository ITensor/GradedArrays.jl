# ===========================================================================
#  AbelianGradedArray — dict-of-keys graded array with GradedOneTo axes
# ===========================================================================

"""
    AbelianGradedArray{T,S<:SectorRange,N,D<:AbstractArray{T,N}} <: AbstractGradedArray{T,S,N}

A graded array that stores non-zero blocks in a dictionary keyed by block indices.
Each axis is a [`GradedOneTo`](@ref) carrying sectors, sector lengths, and a dual flag.

Blocks are stored as plain dense arrays of type `D` (default `Array{T,N}`).
Accessing a block via `a[Block(i,j)]` returns a [`AbelianSectorArray`](@ref) wrapping the data
with the appropriate sectors.
"""
struct AbelianGradedArray{T, S <: SectorRange, N, D <: AbstractArray{T, N}} <:
    AbstractGradedArray{T, S, N}
    blockdata::Dict{NTuple{N, Int}, D}
    axes::NTuple{N, GradedOneTo{S}}
end

const AbelianGradedMatrix{T, S, D} = AbelianGradedArray{T, S, 2, D}

# ---------------------------------------------------------------------------
#  Constructors
# ---------------------------------------------------------------------------

# Fully-parameterized undef constructor: finds allowed blocks, allocates, calls inner.
# (allowedblocks is defined in fusion.jl). The axes element type is left unparameterized so
# `S` binds from the type parameters rather than from `axs`, which is empty (and so carries
# no `S`) for a rank-0 array; `allowedblocks` returns the single empty block in that case.
function AbelianGradedArray{T, S, N, D}(
        ::UndefInitializer, axs::NTuple{N, GradedOneTo}
    ) where {T, S <: SectorRange, N, D <: AbstractArray{T, N}}
    block_axes = map(eachdataaxis, axs)
    function allocate_block(bk)
        bk_inds = Int.(Tuple(bk))
        return similar(D, ntuple(d -> block_axes[d][bk_inds[d]], Val(N)))
    end
    bks = allowedblocks(axs)
    blockdata = Dict{NTuple{N, Int}, D}(
        Int.(Tuple(bk)) => allocate_block(bk) for bk in bks
    )
    return AbelianGradedArray{T, S, N, D}(blockdata, axs)
end

# Convenience: infer D = Array{T,N} and S from the axes. Requires at least one axis: the
# sector type of a rank-0 array cannot be inferred from empty axes (there is no symmetry to
# read it from), so a rank-0 array is built through the fully-parameterized constructor with
# an explicit `S`.
function AbelianGradedArray{T}(
        ::UndefInitializer, axs::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    ) where {T}
    N = length(axs)
    return AbelianGradedArray{T, sectortype(eltype(axs)), N, Array{T, N}}(undef, axs)
end

function AbelianGradedArray{T}(
        init::UndefInitializer, ax1::GradedOneTo, axs::GradedOneTo...
    ) where {T}
    return AbelianGradedArray{T}(init, (ax1, axs...))
end

# Convert any `AbstractGradedMatrix` (e.g. a `FusedGradedMatrix`) to an
# `AbelianGradedArray` with the same axes and stored blocks.
function AbelianGradedArray(m::AbstractGradedMatrix)
    # Assumes each allowed block of the target is also stored in `m` — every
    # `similar` allocation is overwritten by the loop below, so no `zero!`
    # is needed.
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
function blocktype(
        ::Type{<:AbelianGradedArray{T, S, N, D}}
    ) where {T, S, N, D}
    return AbelianSectorArray{T, S, N, D}
end
blocktype(a::AbelianGradedArray) = blocktype(typeof(a))

# ---------------------------------------------------------------------------
#  view (primitive): returns AbelianSectorArray sharing data with blockdata
# ---------------------------------------------------------------------------

# Shared implementation: build the `AbelianSectorArray` view for a stored block.
# Construct through `blocktype(a)` so the sector type `S` comes from the parent rather
# than being inferred from `sects`, which is empty (and so carries no `S`) for a rank-0
# array.
function view_abelian(a::AbelianGradedArray{T, <:Any, N}, I::Block{N}) where {T, N}
    bk = Int.(Tuple(I))
    haskey(a.blockdata, bk) || error("Block $bk is not stored.")
    sects = ntuple(d -> eachsectoraxis(axes(a, d))[bk[d]], Val(N))
    return blocktype(a)(sects, a.blockdata[bk])
end

Base.view(a::AbelianGradedArray{T, <:Any, N}, I::Block{N}) where {T, N} = view_abelian(a, I)

# Disambiguate against `view(::AbstractGradedArray{T, <:Any, N}, ::Vararg{Block{1}, N})` for
# N=1, where the splatted form collapses to a single Block{1} argument.
Base.view(a::AbelianGradedArray{T, <:Any, 1}, I::Block{1}) where {T} = view_abelian(a, I)

# Rank-0 (scalar) element access: a rank-0 graded array (e.g. the result of a full
# contraction to a scalar) holds a single trivial-sector value. `a[]` is unambiguous —
# there are no coordinates and exactly one element — unlike the banned higher-rank scalar
# indexing. Defined on the concrete type to take precedence over the N-D block-indexing
# methods, whose `Vararg` signatures also match a no-argument call when N=0.
Base.getindex(a::AbelianGradedArray{T, <:Any, 0}) where {T} = view(a, Block())[]
function Base.setindex!(a::AbelianGradedArray{T, <:Any, 0}, value) where {T}
    view(a, Block())[] = value
    return a
end
# Block assignment copies the sector-array block in, matching the N≥1 path. Distinct from the
# scalar `a[] = value` above: block access carries a `Block` index, scalar access takes none,
# which for a rank-0 array would otherwise collide on the same no-coordinate `setindex!`.
function Base.setindex!(a::AbelianGradedArray{T, <:Any, 0}, value, ::Block{0}) where {T}
    copy!(view(a, Block()), value)
    return a
end

# ---------------------------------------------------------------------------
#  blocks — lazy view delegating to view (following BlockArrays convention)
# ---------------------------------------------------------------------------

"""
    AbelianBlocks{T,N,A<:AbelianGradedArray{T,<:Any,N}} <: AbstractArray{AbelianSectorArray,N}

Lazy view of an `AbelianGradedArray`'s block storage, following the BlockArrays
convention: `getindex` delegates to `view(parent, Block.(I)...)` (shares data),
`setindex!` copies into the existing view.
"""
struct AbelianBlocks{T, N, A <: AbelianGradedArray{T, <:Any, N}} <:
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
        a::AbelianGradedArray{T, <:Any, N}, I::Vararg{AbstractVector{<:BlockIndexRange{1}}, N}
    ) where {T, N}
    ax_dest = ntuple(d -> axes(a, d)[I[d]], Val(N))
    # `zero!` is needed: we only copy sub-ranges from stored source blocks,
    # so destination block regions outside those sub-ranges (and destination
    # blocks with no source counterpart) must start at 0.
    a_dest = zero!(similar(a, ax_dest))
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
        a::AbelianGradedArray{T, <:Any, N}, I::Vararg{AbstractBlockVector{<:Block{1}}, N}
    ) where {T, N}
    ax_dest = ntuple(d -> axes(a, d)[I[d]], Val(N))
    # `zero!` is needed: each source block writes into a sub-range of one
    # destination block, so remaining sub-ranges (and destination blocks
    # with no source counterpart) must start at 0.
    a_dest = zero!(similar(a, ax_dest))
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

function eachblockstoredindex(a::AbelianGradedArray{T, <:Any, N}) where {T, N}
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
    D = datatype(a)
    data_ax_types = Tuple{ntuple(d -> dataaxistype(typeof(axes[d])), Val(N))...}
    D_N = Base.promote_op(similar, D, Type{T}, data_ax_types)
    D_N′ = isconcretetype(D_N) ? D_N : Array{T, N}
    return AbelianGradedArray{T, S, N, D_N′}(undef, axes)
end

# Allocate a graded array from any prototype when the requested axes are
# `GradedOneTo`. The axes carry the structure; the prototype only fixes a
# fallback datatype. Two overloads to resolve ambiguity with `BlockArrays`'
# `StridedArray`-specific methods.
function Base.similar(
        ::AbstractArray, ::Type{T},
        axes::Tuple{GradedOneTo{S}, Vararg{GradedOneTo{S}}}
    ) where {T, S}
    return AbelianGradedArray{T}(undef, axes)
end
function Base.similar(
        ::StridedArray, ::Type{T},
        axes::Tuple{GradedOneTo{S}, Vararg{GradedOneTo{S}}}
    ) where {T, S}
    return AbelianGradedArray{T}(undef, axes)
end
function Base.similar(
        a::AbstractGradedArray{T}, axes::Tuple{Vararg{GradedOneTo}}
    ) where {T}
    return similar(a, T, axes)
end
# Rank-0 destination: the empty axes carry no sector type, so unlike the `GradedOneTo`
# `similar` above (which binds `S` from the axes), both the sector type and the block
# datatype are taken from the prototype `a`.
function Base.similar(a::AbstractGradedArray, ::Type{T}, ::Tuple{}) where {T}
    D = datatype(a)
    D_0 = Base.promote_op(similar, D, Type{T}, Tuple{})
    D_0′ = isconcretetype(D_0) ? D_0 : Array{T, 0}
    return AbelianGradedArray{T, sectortype(a), 0, D_0′}(undef, ())
end
function Base.similar(a::AbelianGradedArray{T}, ::Type{Tv}) where {T, Tv}
    return similar(a, Tv, axes(a))
end
function Base.similar(a::AbelianGradedArray{T}) where {T}
    return similar(a, T)
end

# Block-wise copy; the default falls through to scalar `copyto!`.
function Base.copy(a::AbelianGradedArray{T, S, N, D}) where {T, S, N, D}
    return AbelianGradedArray{T, S, N, D}(
        Dict{NTuple{N, Int}, D}(k => copy(v) for (k, v) in a.blockdata),
        a.axes
    )
end

function Base.copyto!(
        dest::AbelianGradedArray{<:Any, <:Any, N},
        src::AbelianGradedArray{<:Any, <:Any, N}
    ) where {N}
    axes(dest) == axes(src) ||
        throw(
        DimensionMismatch(
            "copyto! axes mismatch: dest $(axes(dest)), src $(axes(src))"
        )
    )
    # Matching axes mean matching allowed-block keys (every allowed block is
    # allocated), so copy each block into the existing destination buffer.
    for (k, v) in src.blockdata
        copyto!(dest.blockdata[k], v)
    end
    return dest
end

# Route eager `conj` through the lazy conjugating broadcast so there is a single
# implementation: `conj.` lowers to a `ConjArray` (dualizing the axes) and materializes via
# `bipermutedimsopadd!` with `op = conj`, which carries the fermionic reversal phase. This
# also overrides Base's `conj(::AbstractArray{<:Real}) = A` short-circuit, so a real-eltype
# graded array still dualizes its axes.
Base.conj(a::AbelianGradedArray) = conj.(a)

# ---------------------------------------------------------------------------
#  permutedims
# ---------------------------------------------------------------------------

function Base.permutedims(a::AbelianGradedArray{<:Any, <:Any, N}, perm) where {N}
    dest_axes = ntuple(i -> axes(a)[perm[i]], Val(N))
    # No `zero!` here: `permutedims!` → `permutedimsopadd!(β=0)` already
    # zeros the destination before writing.
    a_dest = similar(a, dest_axes)
    return permutedims!(a_dest, a, perm)
end

function Base.permutedims!(
        y::AbelianGradedArray{<:Any, <:Any, N}, x::AbelianGradedArray{<:Any, <:Any, N}, perm
    ) where {N}
    TensorAlgebra.permutedimsopadd!(y, identity, x, perm, true, false)
    return y
end

# Block-aware iszero: non-stored blocks are implicitly zero, so an
# `AbelianGradedArray` is zero iff every stored block is zero. The generic
# `Base.iszero(::AbstractArray) = all(iszero, x)` path iterates elements,
# which throws on the no-scalar-indexing guard.
function Base.iszero(a::AbelianGradedArray)
    return all(iszero, values(a.blockdata))
end

# Block-aware random fills: dispatch to the underlying block's `rand!`/`randn!`,
# bypassing the generic `AbstractArray` fallbacks that go through scalar indexing.
# The 3-arg `Random.rand!(rng, A, sp::Sampler)` form is what Random's `rand!(A)`
# / `rand!(A, X)` / `rand!(rng, A, X)` shims ultimately call, so overriding it
# covers every entry point.
function Random.rand!(rng::AbstractRNG, a::AbelianGradedArray, sp::Random.Sampler)
    for b in values(a.blockdata)
        Random.rand!(rng, b, sp)
    end
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbelianGradedArray)
    for b in values(a.blockdata)
        Random.randn!(rng, b)
    end
    return a
end

# Constructors `rand(rng, T, axes)` / `randn(rng, T, axes)` for graded axes:
# allocate an `AbelianGradedArray` from the axes, then fill via the block-aware
# in-place methods above. The generic `Base.rand`/`randn` fallbacks build a
# `Matrix` from `length.(axes)`, which loses the graded structure.
function Base.rand(
        rng::AbstractRNG, ::Type{T},
        ax::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    ) where {T}
    return Random.rand!(rng, AbelianGradedArray{T}(undef, ax))
end
function Base.randn(
        rng::AbstractRNG, ::Type{T},
        ax::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    ) where {T}
    return Random.randn!(rng, AbelianGradedArray{T}(undef, ax))
end

# Shorthand shims forwarding to the canonical `(rng, T, tuple)` forms above. Base's
# `rand`/`randn` defaulting chain hardcodes `Integer` / `Dims` argument shapes, so the
# non-canonical forms (`rand(i, j)`, `rand(T, i, j)`, `rand((i, j))`, ...) never reach the
# graded constructor on their own. A leading `GradedOneTo` keeps these from shadowing the
# zero-argument `rand()` / `randn()`.
function Base.rand(ax::GradedOneTo, axs::GradedOneTo...)
    return rand(Random.default_rng(), Float64, (ax, axs...))
end
function Base.rand(::Type{T}, ax::GradedOneTo, axs::GradedOneTo...) where {T}
    return rand(Random.default_rng(), T, (ax, axs...))
end
function Base.rand(rng::AbstractRNG, ax::GradedOneTo, axs::GradedOneTo...)
    return rand(rng, Float64, (ax, axs...))
end
function Base.rand(
        rng::AbstractRNG,
        ::Type{T},
        ax::GradedOneTo,
        axs::GradedOneTo...
    ) where {T}
    return rand(rng, T, (ax, axs...))
end
function Base.rand(ax::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return rand(Random.default_rng(), Float64, ax)
end
function Base.rand(::Type{T}, ax::Tuple{GradedOneTo, Vararg{GradedOneTo}}) where {T}
    return rand(Random.default_rng(), T, ax)
end
function Base.rand(rng::AbstractRNG, ax::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return rand(rng, Float64, ax)
end

function Base.randn(ax::GradedOneTo, axs::GradedOneTo...)
    return randn(Random.default_rng(), Float64, (ax, axs...))
end
function Base.randn(::Type{T}, ax::GradedOneTo, axs::GradedOneTo...) where {T}
    return randn(Random.default_rng(), T, (ax, axs...))
end
function Base.randn(rng::AbstractRNG, ax::GradedOneTo, axs::GradedOneTo...)
    return randn(rng, Float64, (ax, axs...))
end
function Base.randn(
        rng::AbstractRNG,
        ::Type{T},
        ax::GradedOneTo,
        axs::GradedOneTo...
    ) where {T}
    return randn(rng, T, (ax, axs...))
end
function Base.randn(ax::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return randn(Random.default_rng(), Float64, ax)
end
function Base.randn(::Type{T}, ax::Tuple{GradedOneTo, Vararg{GradedOneTo}}) where {T}
    return randn(Random.default_rng(), T, ax)
end
function Base.randn(rng::AbstractRNG, ax::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return randn(rng, Float64, ax)
end

# Block-aware diagonal check: block-diagonal (no off-diagonal stored blocks), and each
# stored diagonal block is itself diagonal. Bypasses the generic scalar-indexing path.
function LinearAlgebra.isdiag(A::AbelianGradedMatrix)
    isblockdiagonal(A) || return false
    for bI in eachblockstoredindex(A)
        LinearAlgebra.isdiag(view(A, bI)) || return false
    end
    return true
end

# Orthogonal projection of a dense source into the symmetry-allowed subspace.
# Magnitude-blind: forbidden-block entries of `src` are dropped without inspection.
# The `TensorAlgebra.project` wrapper verifies the discarded weight is small.
function TensorAlgebra.projectto!(dest::AbelianGradedArray, src::AbstractArray)
    # Reshape `src` to `size(dest)` (a no-op when the ranks already match), so a lower-rank
    # `src` may omit trailing length-1 axes (e.g. an auxiliary flux-canceling leg); a genuine
    # size mismatch still errors in `reshape`.
    src = reshape(src, size(dest))
    zero!(dest)
    for b in allowedblocks(axes(dest))
        block_ranges = ntuple(d -> axes(dest, d)[Block(Int(Tuple(b)[d]))], ndims(dest))
        view(dest, b) .= view(src, block_ranges...)
    end
    return dest
end

# Net charge of a dense operator, read from its dominant-magnitude entry: find the block
# holding that entry over the stored axes (domain dualized to match `zeros_map`/`similar_map`)
# and fuse that block's per-axis sectors, each with its axis's arrow applied (the same fusion
# `allowedblocks` is built on, so the charge lines up with which blocks `project` keeps). This
# is the abelian fast path for aux derivation; the general (possibly multi-sector) derivation
# lives in the `TensorMap` backend.
function projected_charge(src::AbstractArray, codomain_axes, domain_axes)
    stored = (codomain_axes..., conj.(domain_axes)...)
    src = reshape(src, length.(stored))
    I = Tuple(findmax(abs, src)[2])
    secs = map(stored, I) do ax, i
        return eachsectoraxis(ax)[Int(BlockArrays.findblock(ax, i))]
    end
    return reduce(tensor_product, secs)
end

# `allocate_project` with graded axes: the destination allocation, which is where a trailing
# surplus axis gets its space derived. With one axis more
# in `src` than the given axes account for, that trailing surplus axis is an auxiliary leg
# appended as the last domain axis: derive its space so the result is symmetry-allowed. The
# trailing position matches how `stack` lays out an operator multiplet and the Julia convention
# that trailing length-1 axes are implicit/flexible. Abelian sectors are one-dimensional, so each
# length-1 slice along the aux axis gets its own projected charge — the per-slice lookup is the
# whole derivation, and a multi-slice aux comes out as a direct sum (e.g. stacking `[S⁺, S⁻]`
# gives `[U1(2), U1(-2)]`, an MPO-virtual-leg structure). For the matricized result `X` the Gram
# matrix `X * X'` contracts the aux away (for a spin multiplet, `X * X' = S·S`, the Casimir).
# The aux is a genuine axis of the result, read off `axes(dest)`.
function TensorAlgebra.allocate_project(
        src::AbstractArray, codomain_axes::Tuple{GradedOneTo, Vararg{GradedOneTo}}, domain_axes
    )
    nphys = length(codomain_axes) + length(domain_axes)
    if ndims(src) > nphys
        ndims(src) == nphys + 1 || throw(
            ArgumentError(
                "`project`: expected at most one trailing auxiliary axis beyond the \
                $nphys given axes, got a rank-$(ndims(src)) input"
            )
        )
        qs = map(eachslice(src; dims = nphys + 1)) do slice
            return projected_charge(slice, codomain_axes, domain_axes)
        end
        # Merge contiguous equal charges into one sector of that multiplicity (matching the
        # `TensorMap` backend); non-contiguous repeats stay separate to preserve slice order.
        ps = Pair{eltype(qs), Int}[]
        for q in qs
            if isempty(ps) || first(ps[end]) != q
                push!(ps, q => 1)
            else
                ps[end] = q => (last(ps[end]) + 1)
            end
        end
        domain_axes = (domain_axes..., gradedrange(ps))
    end
    return TensorAlgebra.similar_map(src, codomain_axes, domain_axes)
end

# Materialize the graded array into a dense `Array`. The checked projection verbs reach this
# through `convert(Array, dest)` to compare elementwise against a dense source, which would
# otherwise fall back to a `src - dest` broadcast that scalar-indexes the block storage.
function Base.Array(a::AbelianGradedArray{T, <:Any, N}) where {T, N}
    dest = zeros(T, size(a))
    for bI in eachblockstoredindex(a)
        block_ranges = ntuple(d -> axes(a, d)[Block(Int(Tuple(bI)[d]))], N)
        copyto!(view(dest, block_ranges...), view(a, bI))
    end
    return dest
end

function LinearAlgebra.norm(a::AbelianGradedArray, p::Real = 2)
    if p == Inf
        isempty(a.blockdata) && return zero(float(real(eltype(a))))
        return maximum(Base.Fix2(LinearAlgebra.norm, p), values(a.blockdata))
    elseif p > 0
        s = zero(float(real(eltype(a))))
        for b in values(a.blockdata)
            s += LinearAlgebra.norm(b, p)^p
        end
        return s^inv(p)
    else
        throw(ArgumentError("Norm with non-positive p ($p) is not defined"))
    end
end

function LinearAlgebra.dot(a::AbelianGradedArray, b::AbelianGradedArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    # Matching axes mean matching allowed-block keys, so each `a` block has a
    # counterpart in `b`.
    s = zero(LinearAlgebra.dot(zero(eltype(a)), zero(eltype(b))))
    for (k, ablk) in pairs(a.blockdata)
        s += LinearAlgebra.dot(ablk, b.blockdata[k])
    end
    return s
end

# Forbidden blocks are zero, so the total is the sum over the stored blocks.
function Base.sum(a::AbelianGradedArray)
    s = zero(eltype(a))
    for b in values(a.blockdata)
        s += sum(b)
    end
    return s
end

# `maximum`/`minimum` fold over the stored blocks, but unlike `sum` they must also see the
# implicit zeros from forbidden and allowed-but-unstored blocks: those zeros are real
# elements, so unless every block is stored (`blockstoredlength == blocklength`) the
# reduction folds in `f(0)` (this is what makes `maximum(abs, a)` and `minimum(a)` correct
# on sign-definite data). Reductions over opaque block storage would otherwise scalar-index
# it and error.
function Base.maximum(f, a::AbelianGradedArray)
    blocks = values(a.blockdata)
    m = isempty(blocks) ? f(zero(eltype(a))) : maximum(b -> maximum(f, b), blocks)
    return blockstoredlength(a) == blocklength(a) ? m : max(m, f(zero(eltype(a))))
end
function Base.minimum(f, a::AbelianGradedArray)
    blocks = values(a.blockdata)
    m = isempty(blocks) ? f(zero(eltype(a))) : minimum(b -> minimum(f, b), blocks)
    return blockstoredlength(a) == blocklength(a) ? m : min(m, f(zero(eltype(a))))
end
Base.maximum(a::AbelianGradedArray) = maximum(identity, a)
Base.minimum(a::AbelianGradedArray) = minimum(identity, a)
Base.extrema(a::AbelianGradedArray) = extrema(identity, a)
Base.extrema(f, a::AbelianGradedArray) = (minimum(f, a), maximum(f, a))

# Scalar `*` / `/` are inherited from Base's `AbstractArray`-scalar methods, which
# forward to broadcasting (`a .* x` / `a ./ x`). `AbelianGradedArray` supports the
# linear-broadcast path, so no dedicated overrides are needed here.

# `LinearAlgebra.normalize` infers its result eltype via `typeof(first(a)/nrm)`,
# which scalar-indexes opaque storage.
function LinearAlgebra.normalize(a::AbelianGradedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# ---------------------------------------------------------------------------
#  twist!
# ---------------------------------------------------------------------------

"""
    twist!(a::AbstractGradedArray, dims) -> a

Scale `data(a)` in place by `prod(twist(sectoraxes(a, i)) for i in dims)`.
Here, `twist` is defined as `-1` for odd-parity fermionic charges and `+1` otherwise.

This is a no-op unless `BraidingStyle(sectortype(a))` is `Fermionic`.

See also [`contraction_twist!`](@ref).
"""
function twist!(a::AbstractGradedArray, dims)
    TKS.BraidingStyle(sectortype(a)) isa TKS.Fermionic || return a
    for bI in eachblockstoredindex(a)
        twist!(view(a, bI), dims)
    end
    return a
end

# ---------------------------------------------------------------------------
#  Matrix multiplication (block-diagonal)
# ---------------------------------------------------------------------------

const AbelianGradedVector{T, S, D} = AbelianGradedArray{T, S, 1, D}
const AbelianGradedMatrix{T, S, D} = AbelianGradedArray{T, S, 2, D}

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

# A leading mandatory `GradedOneTo` on every vararg form (and `Tuple{GradedOneTo,
# Vararg{GradedOneTo}}` on every tuple form) keeps these from matching the zero-argument
# calls (`zeros()`, `ones()`, `fill(v)`, `zeros(T, ())`), which would pirate Base for calls
# that involve no GradedArrays-owned type.

"""
    zeros(T, ax1::GradedOneTo, axs::GradedOneTo...)

Create an `AbelianGradedArray{T}` with all allowed (zero-flux) blocks filled with zeros.
"""
function Base.zeros(
        ::Type{T}, ax1::GradedOneTo{S}, axs::GradedOneTo{S}...
    ) where {T, S <: SectorRange}
    return zero!(AbelianGradedArray{T}(undef, ax1, axs...))
end

function Base.zeros(ax1::GradedOneTo, axs::GradedOneTo...)
    return zeros(Float64, ax1, axs...)
end

function Base.zeros(::Type{T}, axs::Tuple{GradedOneTo, Vararg{GradedOneTo}}) where {T}
    return zeros(T, axs...)
end

function Base.zeros(axs::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return zeros(Float64, axs...)
end

"""
    ones(T, ax1::GradedOneTo, axs::GradedOneTo...)

Create an `AbelianGradedArray{T}` with all allowed (zero-flux) blocks filled with ones.
"""
function Base.ones(
        ::Type{T}, ax1::GradedOneTo{S}, axs::GradedOneTo{S}...
    ) where {T, S <: SectorRange}
    return fill!(AbelianGradedArray{T}(undef, ax1, axs...), one(T))
end

function Base.ones(ax1::GradedOneTo, axs::GradedOneTo...)
    return ones(Float64, ax1, axs...)
end

function Base.ones(::Type{T}, axs::Tuple{GradedOneTo, Vararg{GradedOneTo}}) where {T}
    return ones(T, axs...)
end

function Base.ones(axs::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return ones(Float64, axs...)
end

"""
    fill(v, ax1::GradedOneTo, axs::GradedOneTo...)

Create an `AbelianGradedArray{typeof(v)}` with all allowed (zero-flux) blocks filled with `v`.
"""
function Base.fill(v, ax1::GradedOneTo{S}, axs::GradedOneTo{S}...) where {S <: SectorRange}
    return fill!(AbelianGradedArray{typeof(v)}(undef, ax1, axs...), v)
end

function Base.fill(v, axs::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return fill(v, axs...)
end

"""
    getindex(a::AbstractArray, ax1::GradedOneTo, axs::GradedOneTo...)

Construct an `AbelianGradedArray` by projecting the dense data of `a` onto the
symmetry-allowed blocks of the graded axes `(ax1, axs...)`, via
`TensorAlgebra.project` (which errors if `a` has weight outside
the allowed blocks). `a` is reshaped to `length.((ax1, axs...))` first, so a
trailing size-1 bond can be supplied implicitly. Each axis carries its own arrow,
so index with `dual`/`conj` axes to set duality.
"""
function Base.getindex(a::AbstractArray, ax1::GradedOneTo, axs::GradedOneTo...)
    dest_axes = (ax1, axs...)
    # Reshape first so the rank matches the requested axes: indexing selects exactly
    # these axes, so the surplus-axis derivation branch of `project` must not trigger.
    return TensorAlgebra.project(reshape(a, length.(dest_axes)), dest_axes)
end
# Disambiguate the single-axis case for a concrete `Array`: `Base.getindex(::Array,
# ::AbstractUnitRange{<:Integer})` and the projection method above are otherwise equally
# specific, so `dense[graded_axis]` (e.g. building a one-leg graded tensor) is ambiguous.
function Base.getindex(a::Array, ax1::GradedOneTo)
    return invoke(getindex, Tuple{AbstractArray, GradedOneTo, Vararg{GradedOneTo}}, a, ax1)
end
