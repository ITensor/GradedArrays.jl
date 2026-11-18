# ===========================================================================
#  SortedArrayIndices / SortedArrayDictionary
# ===========================================================================

using Dictionaries: Dictionaries, AbstractDictionary, AbstractIndices, gettoken

# Like Dictionaries' `ArrayIndices` over sorted, unique keys, so lookup is a binary search and a
# token is a position. Callers must not mutate the wrapped vector.
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
# Not restricted by key type, so a key that is `isequal` to a stored one but of another type is
# found.
function Dictionaries.gettoken(inds::SortedArrayIndices, i)
    t = searchsortedfirst(inds.keys, i)
    t <= length(inds.keys) && isequal(inds.keys[t], i) || return (false, 0)
    return (true, t)
end
Dictionaries.gettokenvalue(inds::SortedArrayIndices, t::Int) = inds.keys[t]
Dictionaries.istokenassigned(::SortedArrayIndices, ::Int) = true

# Like Dictionaries' `ArrayDictionary` over sorted, unique keys.
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
