# ===========================================================================
#  AdjointFusedGradedArray — lazy adjoint of a fused graded array
# ===========================================================================

"""
    AdjointFusedGradedArray{T,S<:SectorRange,N,P<:AbstractFusedGradedArray{T,S,N}} <: AbstractFusedGradedArray{T,S,N}

Lazy adjoint (conjugate transpose) of a fused graded array, produced by `adjoint`/`'` on a fused
graded matrix. Analogous to TensorKit's `AdjointTensorMap` and `LinearAlgebra.Adjoint`.
"""
struct AdjointFusedGradedArray{
        T,
        S <: SectorRange,
        N,
        P <: AbstractFusedGradedArray{T, S, N},
    } <: AbstractFusedGradedArray{T, S, N}
    parent::P
end

function AdjointFusedGradedArray(
        parent::P
    ) where {T, S, N, P <: AbstractFusedGradedArray{T, S, N}}
    return AdjointFusedGradedArray{T, S, N, P}(parent)
end

Base.parent(a::AdjointFusedGradedArray) = a.parent

# `adjoint` swaps codomain and domain and adjoints each block. All three are computed on demand from
# the parent (like `LinearAlgebra.Adjoint`), so the wrapper stays a trivial parent-only type.
sectordata(a::AdjointFusedGradedArray) = map(adjoint, sectordata(parent(a)))

# `adjoint` of a fused graded matrix is the lazy wrapper; a second `adjoint` unwraps to the parent.
Base.adjoint(a::AbstractFusedGradedMatrix) = AdjointFusedGradedArray(a)
Base.adjoint(a::AdjointFusedGradedArray) = parent(a)

# ---- accessors ----

# The block-data type is the `adjoint` of the parent's block data (adjointed lazily).
function datatype(::Type{<:AdjointFusedGradedArray{T, S, N, P}}) where {T, S, N, P}
    return Base.promote_op(adjoint, datatype(P))
end

# The adjoint swaps codomain and domain: its codomain is the parent's domain and its domain is the
# parent's codomain (`axes(m') == (dual(axes(m, 2)), dual(axes(m, 1)))`). `biaxes` is the core axis
# accessor; block indexing, reductions, `copy`, and the like derive from it and `sectordata`
# generically on `AbstractFusedGradedMatrix`.
function biaxes(a::AdjointFusedGradedArray)
    b = biaxes(parent(a))
    return bispace(domain(b), codomain(b))
end

function Base.similar(a::AdjointFusedGradedArray, ::Type{T}) where {T}
    return FusedGradedMatrix{T}(undef, axis_codomain(a), axis_domain(a))
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
    Base.print_array(io, a)
    return nothing
end

Base.show(io::IO, a::AdjointFusedGradedArray) = print(io, "adjoint(", parent(a), ")")
