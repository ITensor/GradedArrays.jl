# ===========================================================================
#  SortedArrayIndices / SortedArrayDictionary — read-only sorted-array dictionaries
# ===========================================================================

using Dictionaries: Dictionaries, AbstractDictionary, AbstractIndices, gettoken

# Self-keyed index set over a sorted, unique keys vector, in the naming family of Dictionaries'
# `ArrayIndices` plus the sorted invariant: a lookup binary-searches the keys, and the token of a
# key is its position. Read-only; the sorted+unique invariant is the caller's (e.g. a
# `FusedGradedOneTo`'s stored labels), and callers must not mutate the vectors they wrap.
struct SortedArrayIndices{K, KS <: AbstractVector{K}} <: AbstractIndices{K}
    keys::KS
end

Base.length(inds::SortedArrayIndices) = length(inds.keys)
Dictionaries.istokenizable(::SortedArrayIndices) = true
Dictionaries.tokentype(::SortedArrayIndices) = Int
function Dictionaries.iteratetoken(inds::SortedArrayIndices, s...)
    return iterate(eachindex(inds.keys), s...)
end
function Dictionaries.iteratetoken_reverse(inds::SortedArrayIndices)
    isempty(inds.keys) && return nothing
    t = lastindex(inds.keys)
    return (t, t)
end
function Dictionaries.iteratetoken_reverse(inds::SortedArrayIndices, t)
    t -= 1
    t < firstindex(inds.keys) && return nothing
    return (t, t)
end
function Dictionaries.gettoken(inds::SortedArrayIndices{K}, i) where {K}
    i isa K || return (false, 0)
    t = searchsortedfirst(inds.keys, i)
    t <= length(inds.keys) && isequal(inds.keys[t], i) || return (false, 0)
    return (true, t)
end
Dictionaries.gettokenvalue(inds::SortedArrayIndices, t::Int) = inds.keys[t]
Dictionaries.istokenassigned(::SortedArrayIndices, ::Int) = true

# Dictionary over parallel sorted-keys / values vectors (`ArrayDictionary` plus the sorted
# invariant); its key set is the matching `SortedArrayIndices`, whose positional tokens it shares.
struct SortedArrayDictionary{K, V, KS <: AbstractVector{K}, VS <: AbstractVector{V}} <:
    AbstractDictionary{K, V}
    keys::KS
    values::VS
    function SortedArrayDictionary(keys::AbstractVector, values::AbstractVector)
        length(keys) == length(values) ||
            throw(ArgumentError("keys and values must have the same length"))
        return new{eltype(keys), eltype(values), typeof(keys), typeof(values)}(keys, values)
    end
end

Base.keys(d::SortedArrayDictionary) = SortedArrayIndices(d.keys)
Dictionaries.istokenizable(::SortedArrayDictionary) = true
Dictionaries.gettoken(d::SortedArrayDictionary, i) = gettoken(keys(d), i)
Dictionaries.gettokenvalue(d::SortedArrayDictionary, t::Int) = d.values[t]
Dictionaries.istokenassigned(::SortedArrayDictionary, ::Int) = true

# Canonicalize any dictionary into the sorted parallel-vector form: collect the keys, sort them,
# and permute the values to match. The identity method makes the already-canonical case a no-op
# (no copy), so trusted paths that pass one through keep sharing it.
function Base.convert(
        ::Type{SortedArrayDictionary{K, V, KS, VS}}, d::SortedArrayDictionary{K, V, KS, VS}
    ) where {K, V, KS <: AbstractVector{K}, VS <: AbstractVector{V}}
    return d
end
function Base.convert(
        ::Type{SortedArrayDictionary{K, V, KS, VS}}, d::AbstractDictionary
    ) where {K, V, KS <: AbstractVector{K}, VS <: AbstractVector{V}}
    ks = collect(K, keys(d))
    vs = collect(V, d)
    perm = sortperm(ks)
    return SortedArrayDictionary(convert(KS, ks[perm])::KS, convert(VS, vs[perm])::VS)
end
