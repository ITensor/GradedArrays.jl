using LabelledNumbers:
  LabelledNumbers, IsLabelled, LabelledUnitRange, label, label_type, unlabel

# LabelledUnitRangeDual is obtained by slicing a GradedUnitRangeDual with a block

struct LabelledUnitRangeDual{T,NondualUnitRange<:AbstractUnitRange{T}} <:
       AbstractUnitRange{T}
  nondual_unitrange::NondualUnitRange
end

dual(a::LabelledUnitRange) = LabelledUnitRangeDual(a)
nondual(a::LabelledUnitRangeDual) = a.nondual_unitrange
dual(a::LabelledUnitRangeDual) = nondual(a)
isdual(::LabelledUnitRangeDual) = true
blocklabels(la::LabelledUnitRangeDual) = [label(la)]

map_blocklabels(f, la::LabelledUnitRange) = labelled(unlabel(la), f(label(la)))
map_blocklabels(f, lad::LabelledUnitRangeDual) = dual(map_blocklabels(f, nondual(lad)))

function nondual_type(
  ::Type{<:LabelledUnitRangeDual{<:Any,NondualUnitRange}}
) where {NondualUnitRange}
  return NondualUnitRange
end
dual_type(T::Type{<:LabelledUnitRangeDual}) = nondual_type(T)
function dual_type(T::Type{<:LabelledUnitRange})
  return LabelledUnitRangeDual{eltype(T),T}
end

LabelledNumbers.label(a::LabelledUnitRangeDual) = dual(label(nondual(a)))
LabelledNumbers.unlabel(a::LabelledUnitRangeDual) = unlabel(nondual(a))
LabelledNumbers.LabelledStyle(::LabelledUnitRangeDual) = IsLabelled()
function LabelledNumbers.label_type(type::Type{<:LabelledUnitRangeDual})
  # `dual_type` right now doesn't do anything but anticipates defining `SectorDual`.
  return dual_type(label_type(nondual_type(type)))
end

for f in [:first, :getindex, :last, :length, :step]
  @eval Base.$f(a::LabelledUnitRangeDual, args...) =
    labelled($f(unlabel(a), args...), label(a))
end

# fix ambiguities
Base.getindex(a::LabelledUnitRangeDual, i::Integer) = dual(nondual(a)[i])
function Base.getindex(a::LabelledUnitRangeDual, indices::AbstractUnitRange{<:Integer})
  return dual(nondual(a)[indices])
end

function Base.iterate(a::LabelledUnitRangeDual, i)
  i == last(a) && return nothing
  next = convert(eltype(a), labelled(i + step(a), label(a)))
  return (next, next)
end

function Base.show(io::IO, ::MIME"text/plain", a::LabelledUnitRangeDual)
  println(io, typeof(a))
  return print(io, label(a), " => ", unlabel(a))
end

function Base.show(io::IO, a::LabelledUnitRangeDual)
  return print(io, nameof(typeof(a)), " ", label(a), " => ", unlabel(a))
end

function Base.AbstractUnitRange{T}(a::LabelledUnitRangeDual) where {T}
  return AbstractUnitRange{T}(nondual(a))
end
