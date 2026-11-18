# ===========================================================================
#  SectorData
# ===========================================================================

using Dictionaries:
    AbstractDictionary, Dictionary, gettoken, gettokenvalue, istokenassigned, istokenizable

# `offset` is 0-based.
_dataview(buffer, offset::Int, sz::Tuple{Int}) = view(buffer, (offset + 1):(offset + sz[1]))
function _dataview(buffer, offset::Int, sz::Tuple{Int, Int})
    return reshape(view(buffer, (offset + 1):(offset + prod(sz))), sz)
end

# Maps each stored sector to the 0-based `offset` of its block in the buffer and the block `size`.
const SectorDataLayout{S, N} = SortedArrayDictionary{
    S, @NamedTuple{offset::Int, size::NTuple{N, Int}},
    Vector{S}, Vector{@NamedTuple{offset::Int, size::NTuple{N, Int}}}
}

# One block per sector shared by the codomain and domain, in sorted order, as in TensorKit's
# `.data` layout.
function sectordatalayout(
        codomain::FusedGradedOneTo{S}, domain::FusedGradedOneTo{S}
    ) where {S <: Sector}
    cods, codd = sectors(codomain), datalengths(codomain)
    doms, domd = sectors(domain), datalengths(domain)
    coupled = S[]
    layouts = @NamedTuple{offset::Int, size::NTuple{2, Int}}[]
    offset = 0
    i = j = 1
    while i <= length(cods) && j <= length(doms)
        if isless(cods[i], doms[j])
            i += 1
        elseif isless(doms[j], cods[i])
            j += 1
        else
            sz = (codd[i], domd[j])
            push!(coupled, cods[i])
            push!(layouts, (offset = offset, size = sz))
            offset += prod(sz)
            i += 1
            j += 1
        end
    end
    return SortedArrayDictionary(coupled, layouts)
end

function sectordatalayout(axis::FusedGradedOneTo)
    lens = datalengths(axis)
    layouts = Vector{@NamedTuple{offset::Int, size::NTuple{1, Int}}}(undef, length(lens))
    offset = 0
    for k in eachindex(lens)
        layouts[k] = (offset = offset, size = (lens[k],))
        offset += lens[k]
    end
    return SortedArrayDictionary(collect(sectors(axis)), layouts)
end

bufferlength(datalayout) = sum(layout -> prod(layout.size), datalayout; init = 0)

# Lazy dictionary from sector to block data of a fused graded array, where each block is a `view`
# into the buffer.
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

# --- AbstractDictionary interface ---

Base.keys(sd::SectorData) = keys(sd.datalayout)
# The first method avoids an ambiguity with the Dictionaries fallback, which is defined at the key
# type.
Base.isassigned(sd::SectorData{S}, s::S) where {S} = haskey(sd.datalayout, s)
Base.isassigned(sd::SectorData, s) = haskey(sd.datalayout, s)
Base.@propagate_inbounds function Base.getindex(sd::SectorData{S}, s::S) where {S}
    layout = sd.datalayout[s]
    return _dataview(sd.parent.buffer, layout.offset, layout.size)
end

# Share tokens with `datalayout` so iteration is O(1) per step.
Dictionaries.istokenizable(::SectorData) = true
Dictionaries.gettoken(sd::SectorData, s) = gettoken(sd.datalayout, s)
Dictionaries.istokenassigned(sd::SectorData, t) = istokenassigned(sd.datalayout, t)
function Dictionaries.gettokenvalue(sd::SectorData, t)
    layout = gettokenvalue(sd.datalayout, t)
    return _dataview(sd.parent.buffer, layout.offset, layout.size)
end
