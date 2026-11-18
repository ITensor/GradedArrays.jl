# ===========================================================================
#  AdjointFusedGradedArray
# ===========================================================================

# Lazy adjoint of a fused graded matrix, like `LinearAlgebra.Adjoint`.
struct AdjointFusedGradedArray{
        T,
        S <: Sector,
        N,
        P <: AbstractFusedGradedArray{T, S, N},
    } <: AbstractFusedGradedArray{T, S, N}
    parent::P
end

const AdjointFusedGradedMatrix{T, S, P} = AdjointFusedGradedArray{T, S, 2, P}
const AdjointFusedGradedVector{T, S, P} = AdjointFusedGradedArray{T, S, 1, P}

function AdjointFusedGradedArray(
        parent::P
    ) where {T, S, N, P <: AbstractFusedGradedArray{T, S, N}}
    return AdjointFusedGradedArray{T, S, N, P}(parent)
end

Base.parent(a::AdjointFusedGradedArray) = a.parent

# `Iterators.map` is the lazy map from Dictionaries, so no dictionary is materialized.
sectordata(a::AdjointFusedGradedArray) = Iterators.map(adjoint, sectordata(parent(a)))

Base.adjoint(a::AbstractFusedGradedMatrix) = AdjointFusedGradedArray(a)
Base.adjoint(a::AdjointFusedGradedArray) = parent(a)

# ---- accessors ----

function datatype(::Type{<:AdjointFusedGradedArray{T, S, N, P}}) where {T, S, N, P}
    return Base.promote_op(adjoint, datatype(P))
end

axes_codomain(a::AdjointFusedGradedArray) = axes_domain(parent(a))
axes_domain(a::AdjointFusedGradedArray) = axes_codomain(parent(a))

Base.dataids(a::AdjointFusedGradedArray) = Base.dataids(parent(a))

# Allocate through the parent so the result keeps its structure, for example a diagonal's adjoint
# stays diagonal.
function Base.similar(a::AdjointFusedGradedArray, ::Type{T}) where {T}
    return similar(parent(a), T, axis_codomain(a), axis_domain(a))
end

# ---- show ----

function Base.summary(io::IO, a::AdjointFusedGradedArray)
    print(io, "adjoint of ")
    summary(io, parent(a))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", a::AdjointFusedGradedArray)
    summary(io, a)
    println(io, ":")
    show_biaxes(io, a)
    isempty(sectordata(parent(a))) && return nothing
    println(io)
    Base.print_array(io, a)
    return nothing
end

Base.show(io::IO, a::AdjointFusedGradedArray) = print(io, "adjoint(", parent(a), ")")
