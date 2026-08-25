# ===========================================================================
#  SectorData — lazy dictionary of a fused graded array's per-sector block data
# ===========================================================================

using Dictionaries:
    AbstractDictionary, Dictionary, gettoken, gettokenvalue, istokenassigned, istokenizable

# A single block's reduced data as a `view` into the contiguous buffer: a `(len,)` size gives a 1-D
# vector block, a `(rows, cols)` size a reshaped matrix block. Both share storage with the buffer,
# so writes land in it. `offset` is 0-based (the count of buffer entries before this block).
_dataview(buffer, offset::Int, sz::Tuple{Int}) = view(buffer, (offset + 1):(offset + sz[1]))
function _dataview(buffer, offset::Int, sz::Tuple{Int, Int})
    return reshape(view(buffer, (offset + 1):(offset + prod(sz))), sz)
end

# `sectordatalayout` builds (2-arg and 1-arg axis forms, below) or reads off a fused graded array
# (the accessor each buffer-backed array defines) the sector → block-layout dictionary locating
# each stored block's data in the contiguous buffer: a 0-based buffer `offset` plus the block's
# data `size` (`(len,)` for a vector block, `(rows, cols)` for a matrix block). A
# `SortedArrayDictionary` keyed by the lazy sector view over the sorted coupled-sector labels.
# Computed once at array construction and carried by the array, so `sectordata` wraps it on each
# call without rebuilding it.

# The concrete type of a canonical `N`-dimensional layout for sector type `S`: it involves
# `labeltype(S)`, so the `datalayout` struct fields are loosely typed and their `sectordatalayout`
# accessors typeassert against this.
function datalayouttype(::Type{S}, ::Val{N}) where {S <: SectorRange, N}
    return SortedArrayDictionary{
        S, @NamedTuple{offset::Int, size::NTuple{N, Int}},
        sectorstype(S), Vector{@NamedTuple{offset::Int, size::NTuple{N, Int}}}
    }
end

# Matrix form: one block per coupled sector (present on both codomain and domain), in sorted
# coupled-sector order and column-major within each block (TensorKit's `.data` layout). A single
# sorted merge over the two axes' sorted label vectors finds the coupled sectors and accumulates
# the offsets.
function sectordatalayout(
        codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {S <: SectorRange}
    codl, codd = sectorlabels(codomain), datalengths(codomain)
    doml, domd = sectorlabels(domain), datalengths(domain)
    labels = labeltype(S)[]
    layouts = @NamedTuple{offset::Int, size::NTuple{2, Int}}[]
    offset = 0
    i = j = 1
    while i <= length(codl) && j <= length(doml)
        if isless(codl[i], doml[j])
            i += 1
        elseif isless(doml[j], codl[i])
            j += 1
        else
            sz = (codd[i], domd[j])
            push!(labels, codl[i])
            push!(layouts, (offset = offset, size = sz))
            offset += prod(sz)
            i += 1
            j += 1
        end
    end
    return SortedArrayDictionary(mappedarray(S, labels), layouts)
end

# Vector form: one block per axis sector, in sorted-sector order; the offsets are the prefix sums
# of the axis's stored data lengths. The label vector is shared with the axis.
function sectordatalayout(axis::FusedGradedOneTo)
    lens = datalengths(axis)
    layouts = Vector{@NamedTuple{offset::Int, size::NTuple{1, Int}}}(undef, length(lens))
    offset = 0
    for k in eachindex(lens)
        layouts[k] = (offset = offset, size = (lens[k],))
        offset += lens[k]
    end
    return SortedArrayDictionary(sectors(axis), layouts)
end

# Total buffer length the blocks of a layout tile; the fused array constructors validate their
# buffer against it. The blocks are contiguous, so it is the sum of the block sizes (well defined
# for any layout dictionary, canonical or not).
fusedbufferlength(datalayout) = sum(layout -> prod(layout.size), datalayout; init = 0)

"""
    SectorData{S,T,P,I} <: Dictionaries.AbstractDictionary{S,T}

Lazy dictionary of the per-coupled-sector block data of a fused graded array, wrapping the array
itself. Keys are the coupled sectors; each value materializes on access as a `view` into the array's
contiguous buffer (a 1-D view for a [`FusedGradedVector`](@ref), a reshaped 2-D view for a
[`FusedGradedMatrix`](@ref)), so no block-shaped storage is held and writes through a value land in
the buffer. The value type is `datatype(parent)`. The `datalayout` field is the array's carried
sector → offset/size layout (see `sectordatalayout`), so constructing a `SectorData` is a plain
wrap with no per-call work.
"""
struct SectorData{S, T, P <: AbstractFusedGradedArray, I <: AbstractDictionary{S}} <:
    AbstractDictionary{S, T}
    parent::P
    datalayout::I
end

function SectorData(
        parent::AbstractFusedGradedArray,
        datalayout::AbstractDictionary{S}
    ) where {S}
    return SectorData{S, datatype(parent), typeof(parent), typeof(datalayout)}(
        parent, datalayout
    )
end

# The construction entry point (what `sectordata` wraps): read the carried layout off the parent
# via the `sectordatalayout` accessor, which each buffer-backed fused array defines.
SectorData(parent::AbstractFusedGradedArray) = SectorData(parent, sectordatalayout(parent))

# --- AbstractDictionary interface (read-only; values are views, so they mutate through) ---

Base.keys(sd::SectorData) = keys(sd.datalayout)
Base.isassigned(sd::SectorData{S}, s::S) where {S} = haskey(sd.datalayout, s)
Base.@propagate_inbounds function Base.getindex(sd::SectorData{S}, s::S) where {S}
    layout = sd.datalayout[s]
    return _dataview(sd.parent.buffer, layout.offset, layout.size)
end

# Share tokens with the backing `datalayout` so iteration/`values`/`pairs` stay O(1) per step.
Dictionaries.istokenizable(::SectorData) = true
Dictionaries.gettoken(sd::SectorData, s) = gettoken(sd.datalayout, s)
Dictionaries.istokenassigned(sd::SectorData, t) = istokenassigned(sd.datalayout, t)
function Dictionaries.gettokenvalue(sd::SectorData, t)
    layout = gettokenvalue(sd.datalayout, t)
    return _dataview(sd.parent.buffer, layout.offset, layout.size)
end
