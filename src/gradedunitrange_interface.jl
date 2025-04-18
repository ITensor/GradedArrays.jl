using BlockArrays: AbstractBlockVector, BlockRange, blocklength
using FillArrays: Fill

"""
    dual(x)

Take the dual of the symmetry sector, graded unit range, etc.
By default, it just returns `x`, i.e. it assumes the object
is self-dual.
"""
dual(x) = x

nondual(r::AbstractUnitRange) = r
isdual(::AbstractUnitRange) = false

flip(a::AbstractUnitRange) = dual(map_blocklabels(dual, a))

"""
    dag(r::AbstractUnitRange)

Same as `dual(r)`.
"""
dag(r::AbstractUnitRange) = dual(r)

"""
    dag(a::AbstractArray)

Complex conjugates `a` and takes the dual of the axes.
"""
function dag(a::AbstractArray)
  a′ = similar(a, dual.(axes(a)))
  a′ .= conj.(a)
  return a′
end

map_blocklabels(::Any, a::AbstractUnitRange) = a

to_sector(x) = x

sector_type(x) = sector_type(typeof(x))
sector_type(::Type) = error("Not implemented")

struct NoLabel end
blocklabels(r::AbstractUnitRange) = Fill(NoLabel(), blocklength(r))
blocklabels(v::AbstractBlockVector) = mapreduce(blocklabels, vcat, blocks(v))

# == is just a range comparison that ignores labels. Need dedicated function to check equality.
function space_isequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return (isdual(a1) == isdual(a2)) &&
         blocklabels(a1) == blocklabels(a2) &&
         blockisequal(a1, a2)
end
