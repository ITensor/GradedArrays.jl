# ===========================================================================
#  AdjointFusedGradedArray — lazy adjoint of a fused graded array
# ===========================================================================

using Dictionaries: gettoken, gettokenvalue

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
sectordatalengths_codomain(a::AdjointFusedGradedArray) = sectordatalengths_domain(parent(a))
sectordatalengths_domain(a::AdjointFusedGradedArray) = sectordatalengths_codomain(parent(a))

# `adjoint` of a fused graded matrix is the lazy wrapper; a second `adjoint` unwraps to the parent.
Base.adjoint(a::AbstractFusedGradedMatrix) = AdjointFusedGradedArray(a)
Base.adjoint(a::AdjointFusedGradedArray) = parent(a)

# ---- accessors ----

BlockArrays.blocklength(a::AdjointFusedGradedArray) = blocklength(parent(a))

# The block is the `adjoint` of the parent's block (its reduced data adjointed lazily).
function blocktype(::Type{<:AdjointFusedGradedArray{T, S, N, P}}) where {T, S, N, P}
    return FusedSectorMatrix{T, S, Base.promote_op(adjoint, datatype(P))}
end
blocktype(a::AdjointFusedGradedArray) = blocktype(typeof(a))

# The adjoint swaps codomain and domain: its codomain is the parent's domain and its domain is the
# parent's codomain (`axes(m') == (dual(axes(m, 2)), dual(axes(m, 1)))`).
function biaxes(a::AdjointFusedGradedArray)
    b = biaxes(parent(a))
    return bispace(domain(b), codomain(b))
end
Base.axes(a::AdjointFusedGradedArray) = Tuple(biaxes(a))
Base.size(a::AdjointFusedGradedArray) = map(length, axes(a))

function Base.view(a::AdjointFusedGradedArray, I::Block{2})
    i, j = Int.(Tuple(I))
    cod = sectordatalengths_codomain(a)
    dom = sectordatalengths_domain(a)
    @boundscheck begin
        i in 1:length(cod) && j in 1:length(dom) || throw(BoundsError(a, I))
    end
    s_cod = gettokenvalue(keys(cod), i)
    s_dom = gettokenvalue(keys(dom), j)
    s_cod == s_dom ||
        error("Off-diagonal access not supported for AdjointFusedGradedArray")
    return FusedSectorMatrix(adjoint(sectordata(parent(a))[s_cod]), s_cod)
end

function eachblockstoredindex(a::AdjointFusedGradedArray)
    cod = sectordatalengths_codomain(a)
    dom = sectordatalengths_domain(a)
    return (
        Block(gettoken(cod, c)[2][2], gettoken(dom, c)[2][2]) for
            c in keys(sectordata(parent(a)))
    )
end

# Materialize into an owned `FusedGradedMatrix`, copying each adjoint block into a fresh buffer.
function Base.copy(a::AdjointFusedGradedArray)
    return FusedGradedMatrix(
        sectordata(a), sectordatalengths_codomain(a), sectordatalengths_domain(a)
    )
end
function Base.similar(a::AdjointFusedGradedArray, ::Type{T}) where {T}
    return FusedGradedMatrix{T}(undef, only(axes_codomain(a)), only(axes_domain(a)))
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
