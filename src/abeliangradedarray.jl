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

const AbelianGradedMatrix{T, D, S} = AbelianGradedArray{T, 2, D, S}

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
        ::Type{<:AbelianGradedArray{T, N, D, S}}
    ) where {T, N, D, S}
    return AbelianSectorArray{T, N, D, S}
end
blocktype(a::AbelianGradedArray) = blocktype(typeof(a))
datatype(::Type{<:AbelianGradedArray{T, N, D, S}}) where {T, N, D, S} = D

# ---------------------------------------------------------------------------
#  view (primitive): returns AbelianSectorArray sharing data with blockdata
# ---------------------------------------------------------------------------

# Shared implementation: build the `AbelianSectorArray` view for a stored block.
function view_abelian(a::AbelianGradedArray{T, N}, I::Block{N}) where {T, N}
    bk = Int.(Tuple(I))
    haskey(a.blockdata, bk) || error("Block $bk is not stored.")
    sects = ntuple(d -> sectors(axes(a, d))[bk[d]], Val(N))
    return AbelianSectorArray(sects, a.blockdata[bk])
end

Base.view(a::AbelianGradedArray{T, N}, I::Block{N}) where {T, N} = view_abelian(a, I)

# Disambiguate against `view(::AbstractGradedArray{T, N}, ::Vararg{Block{1}, N})` for
# N=1, where the splatted form collapses to a single Block{1} argument.
Base.view(a::AbelianGradedArray{T, 1}, I::Block{1}) where {T} = view_abelian(a, I)

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
        a::AbelianGradedArray{T, N}, I::Vararg{AbstractBlockVector{<:Block{1}}, N}
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

function eachblockstoredindex(a::AbelianGradedArray{T, N}) where {T, N}
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
    D = datatype(blocktype(a))
    data_ax_types = Tuple{ntuple(d -> dataaxistype(typeof(axes[d])), Val(N))...}
    D_N = Base.promote_op(similar, D, Type{T}, data_ax_types)
    D_N′ = isconcretetype(D_N) ? D_N : Array{T, N}
    return AbelianGradedArray{T, N, D_N′, S}(undef, axes)
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
function Base.similar(a::AbelianGradedArray{T}, ::Type{Tv}) where {T, Tv}
    return similar(a, Tv, axes(a))
end
function Base.similar(a::AbelianGradedArray{T}) where {T}
    return similar(a, T)
end

# Block-wise copy; the default falls through to scalar `copyto!`.
function Base.copy(a::AbelianGradedArray{T, N, D, S}) where {T, N, D, S}
    return AbelianGradedArray{T, N, D, S}(
        Dict{NTuple{N, Int}, D}(k => copy(v) for (k, v) in a.blockdata),
        a.axes
    )
end

function Base.copyto!(
        dest::AbelianGradedArray{<:Any, N},
        src::AbelianGradedArray{<:Any, N}
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

# Conjugate element-wise *and* flip axis duality, mirroring `Base.conj` on the
# axis types (`SectorRange`/`GradedOneTo`/`SectorOneTo`). Without the axis flip,
# `conj(t)` would leave bra-layer tensors with the same duality as the ket, and
# any contraction between them would silently pair non-dual against non-dual.
# Delegate per block to `conj(::AbelianSectorArray)`, which carries the fermionic
# reversal phase that a bare block-wise data conjugation would drop.
function Base.conj(a::AbelianGradedArray{T, N, D, S}) where {T, N, D, S}
    return AbelianGradedArray{T, N, D, S}(
        Dict{NTuple{N, Int}, D}(
            k => data(conj(view(a, Block(k...)))) for k in keys(a.blockdata)
        ),
        map(conj, a.axes)
    )
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
    # No `zero!` here: `permutedims!` → `permutedimsopadd!(β=0)` already
    # zeros the destination before writing.
    a_dest = similar(a, dest_axes)
    return permutedims!(a_dest, a, perm)
end

function Base.permutedims!(
        y::AbelianGradedArray{<:Any, N}, x::AbelianGradedArray{<:Any, N}, perm
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
# Use `TensorAlgebra.checked_projectto!` to verify the discarded weight is small.
function TensorAlgebra.projectto!(dest::AbelianGradedArray, src::AbstractArray)
    size(dest) == size(src) || throw(
        DimensionMismatch(
            "projectto!: dest has size $(size(dest)), src has size $(size(src))"
        )
    )
    zero!(dest)
    for b in allowedblocks(axes(dest))
        block_ranges = ntuple(d -> axes(dest, d)[Block(Int(Tuple(b)[d]))], ndims(dest))
        view(dest, b) .= view(src, block_ranges...)
    end
    return dest
end

# Compare via `Array(dest)` so the generic `isapprox(::AbstractArray, ::AbelianGradedArray)`
# path doesn't fall back to a `src - dest` broadcast that scalar-indexes the block storage.
# `kwargs` (e.g. `atol`/`rtol`) are forwarded to `isapprox`, matching the generic verb.
function TensorAlgebra.checked_projectto!(
        dest::AbelianGradedArray, src::AbstractArray; kwargs...
    )
    TensorAlgebra.projectto!(dest, src)
    isapprox(src, Array(dest); kwargs...) ||
        throw(InexactError(:checked_projectto!, typeof(dest), src))
    return dest
end

# Materialize the graded array into a dense `Array` for the default
# `checked_projectto!`/`isapprox`-after path.
function Base.Array(a::AbelianGradedArray{T, N}) where {T, N}
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
