# ===========================================================================
#  AdjointFusedGradedArray — lazy adjoint of a fused graded array
# ===========================================================================

using Dictionaries: gettoken, gettokenvalue

"""
    AdjointFusedGradedArray{T,S<:SectorRange,N,P<:AbstractFusedGradedArray{T,S,N}} <: AbstractFusedGradedArray{T,S,N}

Lazy adjoint (conjugate transpose) of a fused graded array, mirroring TensorKit's `AdjointTensorMap`
and `LinearAlgebra.Adjoint`. Stores only the parent (a trivial wrapper); each block, and the swapped
codomain/domain, are computed on demand as lazy `adjoint` views, so the adjoint shares the parent's
contiguous buffer (no copy). `adjoint`/`'` on a fused graded matrix produces this; a second `adjoint`
unwraps to the parent.
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

Base.parent(a::AdjointFusedGradedArray) = getfield(a, :parent)

# `adjoint` swaps codomain and domain and adjoints each block. Compute all three on demand from the
# parent (like `LinearAlgebra.Adjoint`), so the wrapper stays a trivial parent-only type. `.codomain`,
# `.domain`, and `.blocks` are exposed so the shared block-wise matrix operations (`mul!`, `lmul!`,
# `rmul!`, `allocate_output`) that read those fields on a `FusedGradedMatrix` work here unchanged.
function Base.getproperty(a::AdjointFusedGradedArray, name::Symbol)
    name === :codomain && return parent(a).domain
    name === :domain && return parent(a).codomain
    name === :blocks && return map(adjoint, parent(a).blocks)
    return getfield(a, name)
end

# `adjoint` of a fused graded matrix is the lazy wrapper; a second `adjoint` unwraps to the parent.
Base.adjoint(a::AbstractFusedGradedMatrix) = AdjointFusedGradedArray(a)
Base.adjoint(a::AdjointFusedGradedArray) = parent(a)

# ---- accessors ----

BlockArrays.blocklength(a::AdjointFusedGradedArray) = blocklength(parent(a))

# The block is the `adjoint` of the parent's block (its reduced data adjointed lazily).
function blocktype(::Type{<:AdjointFusedGradedArray{T, S, N, P}}) where {T, S, N, P}
    return FusedSectorMatrix{T, S, Base.promote_op(adjoint, datatype(blocktype(P)))}
end
blocktype(a::AdjointFusedGradedArray) = blocktype(typeof(a))

function biaxes(a::AdjointFusedGradedArray)
    cod = gradedrange(collect(pairs(a.codomain)))
    dom = gradedrange([dual(s) => l for (s, l) in pairs(a.domain)])
    return BiTuple((cod,), (dom,))
end
Base.axes(a::AdjointFusedGradedArray) = Tuple(biaxes(a))
Base.size(a::AdjointFusedGradedArray) = map(length, axes(a))
Base.eltype(::Type{<:AdjointFusedGradedArray{T}}) where {T} = T

function Base.view(a::AdjointFusedGradedArray, I::Block{2})
    i, j = Int.(Tuple(I))
    @boundscheck begin
        i in 1:length(a.codomain) && j in 1:length(a.domain) ||
            throw(BoundsError(a, I))
    end
    s_cod = gettokenvalue(keys(a.codomain), i)
    s_dom = gettokenvalue(keys(a.domain), j)
    s_cod == s_dom ||
        error("Off-diagonal access not supported for AdjointFusedGradedArray")
    return FusedSectorMatrix(adjoint(parent(a).blocks[s_cod]), s_cod)
end

function eachblockstoredindex(a::AdjointFusedGradedArray)
    return (
        Block(gettoken(a.codomain, c)[2][2], gettoken(a.domain, c)[2][2]) for
            c in keys(parent(a).blocks)
    )
end

# Materialize into an owned `FusedGradedMatrix`, copying each adjoint block into a fresh buffer.
Base.copy(a::AdjointFusedGradedArray) = FusedGradedMatrix(a.blocks, a.codomain, a.domain)
function Base.similar(a::AdjointFusedGradedArray, ::Type{T}) where {T}
    return FusedGradedMatrix{T}(undef, a.codomain, a.domain)
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
