# This file defines the interface for type Sector
# all fusion categories (Z{2}, SU2, Ising...) are subtypes of Sector
import TensorKitSectors as TKS

"""
    SectorRange(sector::TKS.Sector)

Unit range with elements of type `Int` that additionally stores a sector to denote the grading.
Equivalent to `Base.OneTo(length(sector))`.
"""
struct SectorRange{I <: TKS.Sector} <: AbstractUnitRange{Int}
    label::I
end

label(r::SectorRange) = r.label
sector_type(I::Type{<:SectorRange}) = I

# ===================================  Base interface  =====================================

Base.length(r::SectorRange) = quantum_dimension(r)

Base.isless(r1::SectorRange, r2::SectorRange) = isless(label(r1), label(r2))
Base.isless(r1::SectorRange, r2::TKS.Sector) = isless(label(r1), r2)
Base.isless(r1::TKS.Sector, r2::SectorRange) = isless(r1, label(r2))

Base.isequal(r1::SectorRange, r2::SectorRange) = isequal(label(r1), label(r2))
Base.:(==)(r1::SectorRange, r2::SectorRange) = label(r1) == label(r2)
Base.:(==)(r1::SectorRange, r2::TKS.Sector) = label(r1) == r2
Base.:(==)(r1::TKS.Sector, r2::SectorRange) = r1 == label(r2)

Base.hash(r::SectorRange, h::UInt) = hash(label(r), h)

Base.OneTo(r::SectorRange) = Base.OneTo(length(r))
Base.first(r::SectorRange) = first(Base.OneTo(r))
Base.last(r::SectorRange) = last(Base.OneTo(r))

function Base.show(io::IO, r::SectorRange{I}) where {I}
    show(io, typeof(r))
    print(io, '(')
    l = sector_label(r)
    isnothing(l) || show(io, l)
    print(io, ')')
    return nothing
end

# =================================  Sectors interface  ====================================

trivial(x) = trivial(typeof(x))
function trivial(axis_type::Type{<:AbstractUnitRange})
    return gradedrange([trivial(sector_type(axis_type)) => 1])  # always returns nondual
end
function trivial(type::Type)
    return error("`trivial` not defined for type $(type).")
end
trivial(::Type{SectorRange{I}}) where {I} = SectorRange{I}(one(I))
trivial(::Type{I}) where {I <: TKS.Sector} = one(I)

istrivial(r::SectorRange) = isone(label(r))
istrivial(r) = (r == trivial(r))

sector_label(r::SectorRange) = sector_label(label(r))
function sector_label(c::TKS.Sector)
    return map(fieldnames(typeof(c))) do f
        return getfield(c, f)
    end
    return c
end

quantum_dimension(g::AbstractUnitRange) = length(g)
quantum_dimension(r::SectorRange) = TKS.dim(label(r))
quantum_dimension(s::TKS.Sector) = TKS.dim(s)

to_sector(x::TKS.Sector) = SectorRange(x)

# convert to range
to_gradedrange(c::SectorRange) = gradedrange([c => 1])
to_gradedrange(c::TKS.Sector) = to_gradedrange(SectorRange(c))

function nsymbol(s1::SectorRange, s2::SectorRange, s3::SectorRange)
    return TKS.Nsymbol(label(s1), label(s2), label(s3))
end

dual(c::TKS.Sector) = TKS.dual(c)
dual(r1::SectorRange) = typeof(r1)(dual(label(r1)))

# ===============================  Fusion rule interface  ==================================

TKS.FusionStyle(::Type{SectorRange{I}}) where {I} = TKS.FusionStyle(I)
TKS.BraidingStyle(::Type{SectorRange{I}}) where {I} = TKS.BraidingStyle(I)

abstract type SymmetryStyle end

struct AbelianStyle <: SymmetryStyle end
struct NotAbelianStyle <: SymmetryStyle end

SymmetryStyle(x) = SymmetryStyle(typeof(x))

# default SymmetryStyle to AbelianStyle
# allows for abelian-like slicing style for GradedUnitRange: assume length(::label) = 1
# and preserve labels in any slicing operation
SymmetryStyle(T::Type) = AbelianStyle()
function SymmetryStyle(::Type{T}) where {T <: SectorRange}
    if TKS.FusionStyle(T) === TKS.UniqueFusion() && TKS.BraidingStyle(T) === TKS.Bosonic()
        return AbelianStyle()
    else
        return NotAbelianStyle()
    end
end
SymmetryStyle(G::Type{<:AbstractUnitRange}) = SymmetryStyle(sector_type(G))

combine_styles(::AbelianStyle, ::AbelianStyle) = AbelianStyle()
combine_styles(::SymmetryStyle, ::SymmetryStyle) = NotAbelianStyle()

function fusion_rule(r1::SectorRange, r2::SectorRange)
    a = label(r1)
    b = label(r2)
    fstyle = TKS.FusionStyle(typeof(r1)) & TKS.FusionStyle(typeof(r2))
    fstyle === TKS.UniqueFusion() && return SectorRange(only(TKS.otimes(a, b)))
    return gradedrange(
        vec([SectorRange(c) => TKS.Nsymbol(a, b, c) for c in TKS.otimes(a, b)])
    )
end

# =============================  Tensor products  ==========================================

# TODO: Overload `TensorAlgebra.tensor_product_axis` for `SectorFusion`.
function tensor_product end
const ⊗ = tensor_product
tensor_product(s::SectorRange) = s
tensor_product(c1::SectorRange, c2::SectorRange) = fusion_rule(c1, c2)
function tensor_product(c1::TKS.Sector, c2::TKS.Sector)
    return tensor_product(to_sector(c1), to_sector(c2))
end
function tensor_product(c1::SectorRange, c2::TKS.Sector)
    return tensor_product(c1, to_sector(c2))
end
function tensor_product(c1::TKS.Sector, c2::SectorRange)
    return tensor_product(to_sector(c1), c2)
end

# =====================================  Sectors ===========================================

const TrivialSector = SectorRange{TKS.Trivial}
TrivialSector() = TrivialSector(TKS.Trivial())
sector_label(::TKS.Trivial) = nothing
function fusion_rule(::TrivialSector, r::SectorRange)
    return TKS.FusionStyle(label(r)) === TKS.UniqueFusion() ? r : to_gradedrange(r)
end
function fusion_rule(r::SectorRange, ::TrivialSector)
    return TKS.FusionStyle(label(r)) === TKS.UniqueFusion() ? r : to_gradedrange(r)
end
fusion_rule(r::TrivialSector, ::TrivialSector) = r

Base.:(==)(::TrivialSector, ::TrivialSector) = true
Base.:(==)(::TrivialSector, r::SectorRange) = isone(label(r))
Base.:(==)(r::SectorRange, ::TrivialSector) = isone(label(r))
Base.isless(::TrivialSector, ::TrivialSector) = false
Base.isless(::TrivialSector, r::SectorRange) = trivial(r) < r
Base.isless(r::SectorRange, ::TrivialSector) = r < trivial(r)

# use promotion to handle trivial sectors in tensor products
Base.promote_rule(::Type{TrivialSector}, ::Type{T}) where {T <: SectorRange} = T
Base.convert(::Type{T}, ::TrivialSector) where {T <: SectorRange} = trivial(T)

const Z{N} = SectorRange{TKS.ZNIrrep{N}}
sector_label(c::TKS.ZNIrrep) = c.n
modulus(::Z{N}) where {N} = N
const Z2 = Z{2}

const U1 = SectorRange{TKS.U1Irrep}
sector_label(c::TKS.U1Irrep) = c.charge
Base.isless(r1::U1, r2::U1) = isless(sector_label(r1), sector_label(r2))

const O2 = SectorRange{TKS.CU1Irrep}
function O2(l::Real)
    j = max(l, zero(l))
    s = if l == 0
        0
    elseif l == -1
        1
    else
        2
    end
    return O2(TKS.CU1Irrep(j, s))
end
function sector_label(c::TKS.CU1Irrep)
    return if c.s == 0
        c.j
    elseif c.s == 1
        oftype(c.j, -1)
    else
        c.j
    end
end
zero_odd(::Type{O2}) = O2(-1)

const SU2 = SectorRange{TKS.SU2Irrep}
sector_label(c::TKS.SU2Irrep) = c.j

const Fib = SectorRange{TKS.FibonacciAnyon}
function Fib(s::AbstractString)
    s == "1" && return Fib(:I)
    s == "τ" && return Fib(:τ)
    throw(ArgumentError("Unrecognized input `$s`"))
end
sector_label(c::TKS.FibonacciAnyon) = c.isone ? "1" : "τ"

const Ising = SectorRange{TKS.IsingAnyon}
function Ising(s::AbstractString)
    s in ("1", "σ", "ψ") || throw(ArgumentError("Unrecognized input `$s`"))
    sym = s == "1" ? :I : Symbol(s)
    return Ising(sym)
end
sector_label(c::TKS.IsingAnyon) = Symbol(c.s) == :I ? "1" : String(c.s)
