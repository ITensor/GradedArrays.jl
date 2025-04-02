using LabelledNumbers: LabelledInteger, label, labelled, unlabel

"""
    dual(x)

Take the dual of the symmetry sector, graded unit range, etc.
By default, it just returns `x`, i.e. it assumes the object
is self-dual.
"""
dual(x) = x

nondual(r::AbstractUnitRange) = r
isdual(::AbstractUnitRange) = false

dual_type(x) = dual_type(typeof(x))
dual_type(T::Type) = T
nondual_type(x) = nondual_type(typeof(x))
nondual_type(T::Type) = T

dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))
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
