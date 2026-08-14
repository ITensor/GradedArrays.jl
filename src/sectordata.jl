# ===========================================================================
#  SectorData — lazy dictionary of a fused graded array's per-sector block data
# ===========================================================================

using Dictionaries:
    AbstractDictionary, Dictionary, gettoken, gettokenvalue, istokenassigned, istokenizable

# A single block's reduced data as a `view` into the contiguous buffer: a 1-D range gives a vector
# block, a `(rows, cols)` shape a reshaped matrix block. Both share storage with the buffer, so writes
# land in it. `offset` is 0-based (the count of buffer entries before this block).
_dataview(buffer, offset::Int, len::Int) = view(buffer, (offset + 1):(offset + len))
function _dataview(buffer, offset::Int, sz::Tuple{Int, Int})
    return reshape(view(buffer, (offset + 1):(offset + prod(sz))), sz)
end

"""
    SectorData{S,T,P,I} <: Dictionaries.AbstractDictionary{S,T}

Lazy dictionary of the per-coupled-sector block data of a fused graded array, wrapping the array
itself. Keys are the coupled sectors; each value materializes on access as a `view` into the array's
contiguous buffer (a 1-D view for a [`FusedGradedVector`](@ref), a reshaped 2-D view for a
[`FusedGradedMatrix`](@ref)), so no block-shaped storage is held and writes through a value land in
the buffer. The value type is `datatype(parent)`. A `sectorindices` dictionary (sector →
offset/shape), the per-sector slice-and-reshape into the buffer, is precomputed from the fused axes so
each block lookup is O(1).
"""
struct SectorData{S, T, P <: AbstractFusedGradedArray, I <: AbstractDictionary{S}} <:
    AbstractDictionary{S, T}
    parent::P
    sectorindices::I
end

function SectorData(
        parent::AbstractFusedGradedArray,
        sectorindices::AbstractDictionary{S}
    ) where {S}
    return SectorData{S, datatype(parent), typeof(parent), typeof(sectorindices)}(
        parent, sectorindices
    )
end

# Vector form: one block per axis sector, in sorted-sector order.
function SectorData(
        parent::AbstractFusedGradedArray,
        datalengths::AbstractDictionary{S, Int}
    ) where {S}
    total = sum(datalengths; init = 0)
    length(parent.buffer) == total ||
        throw(
        DimensionMismatch(
            "buffer length $(length(parent.buffer)) does not match block total $total"
        )
    )
    sectorindices = Dictionary{S, Tuple{Int, Int}}()
    offset = 0
    for s in keys(datalengths)
        len = datalengths[s]
        insert!(sectorindices, s, (offset, len))
        offset += len
    end
    return SectorData(parent, sectorindices)
end

# Matrix form: one block per coupled sector (present on both codomain and domain), in sorted
# coupled-sector order and column-major within each block (TensorKit's `.data` layout).
function SectorData(
        parent::AbstractFusedGradedArray,
        codomain::AbstractDictionary{S, Int}, domain::AbstractDictionary{S, Int}
    ) where {S}
    coupled = intersect(keys(codomain), keys(domain))
    total = sum(c -> codomain[c] * domain[c], coupled; init = 0)
    length(parent.buffer) == total ||
        throw(
        DimensionMismatch(
            "buffer length $(length(parent.buffer)) does not match block total $total"
        )
    )
    sectorindices = Dictionary{S, Tuple{Int, Tuple{Int, Int}}}()
    offset = 0
    for c in coupled
        sz = (codomain[c], domain[c])
        insert!(sectorindices, c, (offset, sz))
        offset += prod(sz)
    end
    return SectorData(parent, sectorindices)
end

# --- AbstractDictionary interface (read-only; values are views, so they mutate through) ---

Base.keys(sd::SectorData) = keys(sd.sectorindices)
Base.isassigned(sd::SectorData{S}, s::S) where {S} = haskey(sd.sectorindices, s)
Base.@propagate_inbounds function Base.getindex(sd::SectorData{S}, s::S) where {S}
    offset, shape = sd.sectorindices[s]
    return _dataview(sd.parent.buffer, offset, shape)
end

# Share tokens with the backing `sectorindices` so iteration/`values`/`pairs` stay O(1) per step.
Dictionaries.istokenizable(::SectorData) = true
Dictionaries.gettoken(sd::SectorData, s) = gettoken(sd.sectorindices, s)
Dictionaries.istokenassigned(sd::SectorData, t) = istokenassigned(sd.sectorindices, t)
function Dictionaries.gettokenvalue(sd::SectorData, t)
    offset, shape = gettokenvalue(sd.sectorindices, t)
    return _dataview(sd.parent.buffer, offset, shape)
end
