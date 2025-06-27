using BlockArrays: AbstractBlockVector, blockisequal, blocklength
using FillArrays: Fill

"""
    dual(x)

Take the dual of the symmetry sector, graded unit range, etc.
By default, it just returns `x`, i.e. it assumes the object
is self-dual.
"""
dual(x) = x

isdual(::AbstractUnitRange) = false

flip(a::AbstractUnitRange) = dual(map_sectors(dual, a))

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
  a′ .= conj.(ungrade(a))
  return a′
end

"""
    ungrade(a::AbstractArray)

Return an array without attached sectors. Avoid copying data when possible.
"""
ungrade(a::AbstractArray) = a

map_sectors(::Any, a::AbstractUnitRange) = a

to_sector(x) = x

sector_type(x) = sector_type(typeof(x))
sector_type(::Type) = error("Not implemented")

struct NoSector end
sectors(r::AbstractUnitRange) = Fill(NoSector(), blocklength(r))
sectors(v::AbstractBlockVector) = mapreduce(sectors, vcat, blocks(v))

# == is just a range comparison that ignores labels. Need dedicated function to check equality.
function space_isequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return (isdual(a1) == isdual(a2)) && sectors(a1) == sectors(a2) && blockisequal(a1, a2)
end

function checkspaces(::Type{Bool}, axes1, axes2)
  return length(axes1) == length(axes2) && all(space_isequal.(axes1, axes2))
end

function checkspaces(ax1, ax2)
  return checkspaces(Bool, ax1, ax2) || throw(ArgumentError(lazy"$ax1 does not match $ax2"))
end

checkspaces_dual(::Type{Bool}, axes1, axes2) = checkspaces(Bool, axes1, dual.(axes2))
checkspaces_dual(axes1, axes2) = checkspaces(axes1, dual.(axes2))
