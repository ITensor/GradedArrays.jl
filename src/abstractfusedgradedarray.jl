# Supertype of the fused graded arrays, `FusedGradedMatrix` and `FusedGradedVector`.
abstract type AbstractFusedGradedArray{T, S, N} <: AbstractArray{T, N} end
const AbstractFusedGradedMatrix{T, S} = AbstractFusedGradedArray{T, S, 2}
const AbstractFusedGradedVector{T, S} = AbstractFusedGradedArray{T, S, 1}

TensorAlgebra.ndims_codomain(::AbstractFusedGradedArray) = 1
TensorAlgebra.has_bipartition(::AbstractFusedGradedArray) = true

TensorAlgebra.matricize(m::AbstractFusedGradedMatrix) = m

using BlockArrays: mortar
using FillArrays: Zeros

# `datatype` is the per-sector block-data array type, defined by each concrete fused array.
datatype(a::AbstractFusedGradedArray) = datatype(typeof(a))
sectortype(::Type{<:AbstractFusedGradedArray{T, S}}) where {T, S} = S

function blocktype(::Type{A}) where {A <: AbstractFusedGradedMatrix}
    return FusedSectorMatrix{eltype(A), sectortype(A), datatype(A)}
end
function blocktype(::Type{A}) where {A <: AbstractFusedGradedVector}
    return FusedSectorVector{eltype(A), sectortype(A), datatype(A)}
end
blocktype(a::AbstractFusedGradedArray) = blocktype(typeof(a))

# The sector → block-data map, overloaded by each concrete fused array. Shared code reads storage
# through it rather than through fields, so lazy wrappers like the adjoint work too.
function sectordata end

# The sorted union of the sectors of every axis of the arguments, analogous to `eachindex(A...)`.
# Sectors carry no arrow, so one union covers dual and non-dual axes alike.
function tensorkit_sectorsupport(
        a::AbstractFusedGradedArray, as::AbstractFusedGradedArray...
    )
    return mapreduce(mergesortedunique, (a, as...)) do x
        return mapreduce(
            tensorkit_sectors,
            mergesortedunique,
            (axes_codomain(x)..., axes_domain(x)...)
        )
    end
end
function sectorsupport(a::AbstractFusedGradedArray, as::AbstractFusedGradedArray...)
    return map(Sector, tensorkit_sectorsupport(a, as...))
end

# Bring the arguments onto the union of their axis supports, so that one position indexes the
# same sector in every one of them. Each result shares its argument's buffer.
function promote_sectorsupport(
        a::AbstractFusedGradedArray, as::AbstractFusedGradedArray...
    )
    cs = tensorkit_sectorsupport(a, as...)
    return map(x -> setsectors(x, cs), (a, as...))
end

# Union of two sorted, unique vectors, in sorted order.
function mergesortedunique(a::AbstractVector{S}, b::AbstractVector{S}) where {S}
    out = S[]
    i = j = 1
    while i <= length(a) && j <= length(b)
        if isless(a[i], b[j])
            push!(out, a[i])
            i += 1
        elseif isless(b[j], a[i])
            push!(out, b[j])
            j += 1
        else
            push!(out, a[i])
            i += 1
            j += 1
        end
    end
    append!(out, @view a[i:end])
    append!(out, @view b[j:end])
    return out
end

# `setsectors(a, ss)` sets the sectors of every axis to exactly `ss`, keeping the stored lengths
# and giving the added sectors length zero, so the result can share `a`'s buffer. `ss` must be
# sorted and cover the sectors of every axis.
setsectors(a::AbstractFusedGradedArray) = setsectors(a, sectorsupport(a))

sectordata(a::AbstractFusedGradedArray, c) = sectordata(a)[c]

# The block of the `i`-th stored sector. Sector keys are never `Int`s, so this does not clash with
# the keyed form.
sectordata(a::AbstractFusedGradedArray, i::Int) = gettokenvalue(sectordata(a), i)

function isblockdiag(A::AbstractFusedGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

# Checking blocks avoids densifying through the generic fallback.
function LinearAlgebra.isdiag(A::AbstractFusedGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        (row == col && LinearAlgebra.isdiag(view(A, bI))) || return false
    end
    return true
end

# Equal codomain and domain axes, so every coupled block is square. Stronger than equal overall
# dimensions, since a graded matrix can be square with rectangular blocks.
issquare(A::AbstractFusedGradedMatrix) = axis_codomain(A) == axis_domain(A)

# Not `LinearAlgebra.checksquare`, which returns the matrix size.
function checksquare(A::AbstractFusedGradedMatrix)
    issquare(A) || throw(
        DimensionMismatch(
            "graded matrix is not square: its codomain and domain axes differ"
        )
    )
    return nothing
end

# ---------------------------------------------------------------------------
#  axes / size
# ---------------------------------------------------------------------------

Base.axes(a::AbstractFusedGradedArray) = Tuple(biaxes(a))
Base.size(a::AbstractFusedGradedArray) = map(length, axes(a))

# ---------------------------------------------------------------------------
#  Block indexing
# ---------------------------------------------------------------------------

function Base.view(m::AbstractFusedGradedMatrix, I::Block{2})
    i, j = Int.(Tuple(I))
    cod, dom = axis_codomain(m), axis_domain(m)
    @boundscheck begin
        i in 1:blocklength(cod) && j in 1:blocklength(dom) || throw(BoundsError(m, I))
    end
    s_cod = sectors(cod)[i]
    s_dom = sectors(dom)[j]
    s_cod == s_dom ||
        error("Off-diagonal access not supported for block-sparse fused graded matrix")
    return FusedSectorMatrix(sectordata(m)[s_cod], s_cod)
end

function eachblockstoredindex(m::AbstractFusedGradedMatrix)
    cod = axis_codomain(m)
    dom = axis_domain(m)
    return (
        Block(sectorindex(cod, c), sectorindex(dom, c)) for
            c in keys(sectordata(m))
    )
end

# ---------------------------------------------------------------------------
#  Block-diagonal reductions and predicates
# ---------------------------------------------------------------------------

function storedlength(A::AbstractFusedGradedMatrix)
    return sum(B -> storedlength(view(A, B)), eachblockstoredindex(A); init = 0)
end

# `sum` requires a zero-preserving `f`, since each forbidden position would add `f(0)`.
# `maximum`/`minimum` fold in a single `f(0)` when any forbidden position is present.
Base.sum(A::AbstractFusedGradedMatrix) = sum(identity, A)
function Base.sum(f, A::AbstractFusedGradedMatrix)
    z = f(zero(eltype(A)))
    iszero(z) || throw_not_zero_preserving_sum(z)
    return sum(B -> sum(f, view(A, B)), eachblockstoredindex(A); init = z)
end

Base.maximum(A::AbstractFusedGradedMatrix) = maximum(identity, A)
function Base.maximum(f, A::AbstractFusedGradedMatrix)
    iszero(blockstoredlength(A)) && return f(zero(eltype(A)))
    m = maximum(B -> maximum(f, view(A, B)), eachblockstoredindex(A))
    return length(A) > storedlength(A) ? max(m, f(zero(eltype(A)))) : m
end

Base.minimum(A::AbstractFusedGradedMatrix) = minimum(identity, A)
function Base.minimum(f, A::AbstractFusedGradedMatrix)
    iszero(blockstoredlength(A)) && return f(zero(eltype(A)))
    m = minimum(B -> minimum(f, view(A, B)), eachblockstoredindex(A))
    return length(A) > storedlength(A) ? min(m, f(zero(eltype(A)))) : m
end

Base.extrema(A::AbstractFusedGradedMatrix) = extrema(identity, A)
Base.extrema(f, A::AbstractFusedGradedMatrix) = (minimum(f, A), maximum(f, A))

function LinearAlgebra.tr(A::AbstractFusedGradedMatrix)
    return sum(
        bI -> LinearAlgebra.tr(view(A, bI)), eachblockstoredindex(A);
        init = zero(eltype(A))
    )
end

function LinearAlgebra.istriu(A::AbstractFusedGradedMatrix)
    return all(LinearAlgebra.istriu, sectordata(A))
end
function LinearAlgebra.istril(A::AbstractFusedGradedMatrix)
    return all(LinearAlgebra.istril, sectordata(A))
end
function LinearAlgebra.isposdef(A::AbstractFusedGradedMatrix)
    return all(LinearAlgebra.isposdef, sectordata(A))
end
Base.iszero(A::AbstractFusedGradedMatrix) = all(iszero, sectordata(A))

# ---------------------------------------------------------------------------
#  copy / copyto!
# ---------------------------------------------------------------------------

function Base.copyto!(dest::AbstractFusedGradedArray, src::AbstractFusedGradedArray)
    axes(dest) == axes(src) || throw(DimensionMismatch("`copyto!` requires matching axes"))
    dsd, ssd = sectordata(dest), sectordata(src)
    for c in keys(ssd)
        copyto!(dsd[c], ssd[c])
    end
    return dest
end

Base.copy(a::AbstractFusedGradedArray) = copyto!(similar(a), a)

# ---------------------------------------------------------------------------
#  Matrix display
# ---------------------------------------------------------------------------

function Base.summary(io::IO, m::AbstractFusedGradedMatrix)
    sd = sectordata(m)
    print(
        io, blocklength(axis_codomain(m)), "×", blocklength(axis_domain(m)), " ",
        summary_typename(io, typeof(m)),
        " with ", length(sd), " stored block", length(sd) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(sd), ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::AbstractFusedGradedMatrix)
    summary(io, m)
    println(io, ":")
    show_biaxes(io, m)
    isempty(sectordata(m)) && return nothing
    println(io)
    Base.print_array(io, m)
    return nothing
end

function Base.show(io::IO, m::AbstractFusedGradedMatrix)
    print(
        io, blocklength(axis_codomain(m)), "×", blocklength(axis_domain(m)), " ",
        summary_typename(io, typeof(m)), " (", length(sectordata(m)), " stored)"
    )
    return nothing
end

# ---------------------------------------------------------------------------
#  Matrix functions
# ---------------------------------------------------------------------------

# A separate `::Type{T}` method so the `convert` target is concrete. Inlined into the loop below,
# Julia 1.10 widens the block container to `AbstractArray` and reconstruction throws (#175).
function convert_eltypes(::Type{T}, arrays) where {T}
    return map(array -> convert(AbstractArray{T}, array), arrays)
end

for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval function Base.$f(A::AbstractFusedGradedMatrix)
        raw = map(Base.$f, sectordata(A))
        # Block eltypes can differ, for example `sqrt` of a real block can be complex.
        T = mapreduce(eltype, promote_type, raw; init = eltype(A))
        return fusedgradedmatrix(
            convert_eltypes(T, raw), axis_codomain(A), axis_domain(A)
        )
    end
end

# ---------------------------------------------------------------------------
#  fill! / zero! / scale!
#
#  These only touch stored (symmetry-allowed) blocks, so a nonzero `fill!`
#  value leaves the forbidden positions at zero.
# ---------------------------------------------------------------------------

function TensorAlgebra.scale!(a::AbstractFusedGradedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

LinearAlgebra.rmul!(a::AbstractFusedGradedArray, β::Number) = TensorAlgebra.scale!(a, β)
LinearAlgebra.lmul!(β::Number, a::AbstractFusedGradedArray) = TensorAlgebra.scale!(a, β)

function TensorAlgebra.zero!(a::AbstractFusedGradedArray)
    for bI in eachblockstoredindex(a)
        zero!(view(a, bI))
    end
    return a
end

function Base.fill!(a::AbstractFusedGradedArray, v)
    for bI in eachblockstoredindex(a)
        fill!(view(a, bI), v)
    end
    return a
end

Base.:+(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray) = a .+ b
Base.:-(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray) = a .- b
Base.:*(a::AbstractFusedGradedArray, x::Number) = a .* x
Base.:*(x::Number, a::AbstractFusedGradedArray) = x .* a
Base.:/(a::AbstractFusedGradedArray, x::Number) = a ./ x

# ---------------------------------------------------------------------------
#  Display
# ---------------------------------------------------------------------------

# Type name for the summary line, with the trailing TensorKitSectors sector type parameter shown as
# `…`. Uses `wrapper` rather than `nameof`, which would always drop the module qualification.
function summary_typename(
        io::IO, type::Type{<:Union{AbstractFusedGradedArray, AbstractGradedOneTo}}
    )
    base = sprint(show, type.name.wrapper; context = io)
    params = collect(type.parameters)
    isempty(params) && return base
    strs = map(params) do p
        p isa Type && p <: SectorProduct &&
            return sprint(show_sectorproduct_type, p; context = io)
        return sprint(show, p; context = io)
    end
    if params[end] isa Type && params[end] <: TKS.Sector
        strs[end] = "…"
    end
    return string(base, "{", join(strs, ", "), "}")
end

function _to_blockarray(a::AbstractFusedGradedArray{T, <:Any, N}) where {T, N}
    blens = map(blocklengths, axes(a))
    blockmat = Array{AbstractArray{T, N}, N}(undef, map(length, blens)...)
    # Unstored blocks are `Zeros`, which print as `⋅`.
    for I in CartesianIndices(blockmat)
        b = Tuple(I)
        blockmat[I] = Zeros{T}(ntuple(d -> blens[d][b[d]], N)...)
    end
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        blockmat[CartesianIndex(Int.(Tuple(bI)))] =
            kron_nd(collect(data(blk)), Array(structure(blk)))
    end
    return mortar(blockmat)
end

function Base.print_array(io::IO, a::AbstractFusedGradedArray)
    return Base.print_array(io, _to_blockarray(a))
end

function Base.Array{T, N}(a::AbstractFusedGradedArray{<:Any, <:Any, N}) where {T, N}
    return Array{T, N}(_to_blockarray(a))
end

# Each block's own `dot` carries the quantum-dimension weight of its sector. Equal axes mean equal
# stored blocks, so iterating one operand's stored indices covers both.
function LinearAlgebra.dot(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    init = zero(LinearAlgebra.dot(zero(eltype(a)), zero(eltype(b))))
    return sum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.dot(view(a, I), view(b, I))
    end
end

# The stored blocks have disjoint support, so the `p`-th powers add, with `p == Inf` handled
# separately since the sum formula collapses to `1` there.
function LinearAlgebra.norm(a::AbstractFusedGradedArray, p::Real = 2)
    p > 0 || throw(ArgumentError("norm with non-positive p ($p) is not defined"))
    init = zero(float(real(eltype(a))))
    p == Inf && return maximum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.norm(view(a, I), p)
    end
    s = sum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.norm(view(a, I), p)^p
    end
    return s^inv(p)
end

# `LinearAlgebra.normalize` scalar-indexes to infer its result eltype.
function LinearAlgebra.normalize(a::AbstractFusedGradedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# `conj` would make the first axis dual, which the fused storage types disallow.
Base.conj(a::AbstractFusedGradedArray) = throw_flips_first_axis(conj, a)

# Unlike `conj`, `real`/`imag` are not semilinear, so they cannot go through the linear-broadcast
# fold.
function Base.real(a::AbstractFusedGradedArray)
    eltype(a) <: Real && return a
    r = similar(a, real(eltype(a)))
    for I in eachblockstoredindex(a)
        copy!(view(r, I), real(view(a, I)))
    end
    return r
end
function Base.imag(a::AbstractFusedGradedArray)
    r = similar(a, real(eltype(a)))
    for I in eachblockstoredindex(a)
        copy!(view(r, I), imag(view(a, I)))
    end
    return r
end

# The 3-arg `rand!(rng, a, sp)` form is what Random's `rand!` entry points ultimately call.
function Random.rand!(rng::AbstractRNG, a::AbstractFusedGradedArray, sp::Random.Sampler)
    for I in eachblockstoredindex(a)
        Random.rand!(rng, view(a, I), sp)
    end
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbstractFusedGradedArray)
    for I in eachblockstoredindex(a)
        Random.randn!(rng, view(a, I))
    end
    return a
end

# ============================  matrix operations (guarded)  ============================
# On a fused vector these would fall through to the generic `AbstractArray` methods, which
# scalar-index into a dense result. `transpose` is unavailable on the matrix too.
Base.transpose(A::AbstractFusedGradedArray) = _matrix_op_error(transpose, A)
Base.adjoint(A::AbstractFusedGradedVector) = _matrix_op_error(adjoint, A)
Base.:*(A::AbstractFusedGradedVector, B::AbstractFusedGradedVector) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::AbstractFusedGradedVector) = _matrix_op_error(LinearAlgebra.tr, A)
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::AbstractFusedGradedVector) = _matrix_op_error($f, A)
end

# ============================  similar_map  ============================
# Undefined on purpose, so that a fused prototype never implicitly allocates a `GradedArray`.
function TensorAlgebra.similar_map(
        ::AbstractFusedGradedArray, ::Type, ::Tuple, ::Tuple
    )
    return throw(
        ArgumentError(
            "`similar_map` with explicit axes is not defined for a fused graded array"
        )
    )
end

# ============================  matrix algebra  ============================

function TensorAlgebra.check_input(
        ::typeof(*),
        A::AbstractFusedGradedMatrix,
        B::AbstractFusedGradedMatrix
    )
    axes(A, 2) == dual(axes(B, 1)) ||
        throw(DimensionMismatch("sector mismatch in contracted dimension"))
    return nothing
end

function TensorAlgebra.check_input(
        ::typeof(mul!),
        C::AbstractFusedGradedMatrix, A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix
    )
    check_input(*, A, B)
    axes(C, 1) == axes(A, 1) || throw(DimensionMismatch())
    axes(C, 2) == axes(B, 2) || throw(DimensionMismatch())
    return nothing
end

# Only an identity copy or an adjoint keeps the first axis non-dual, which fused storage requires.
function TensorAlgebra.check_input(
        ::typeof(TA.bipermutedimsconj), isconj::Bool,
        ::AbstractFusedGradedMatrix, perm_codomain, perm_domain
    )
    is_copy = !isconj && perm_codomain == (1,) && perm_domain == (2,)
    is_adjoint = isconj && perm_codomain == (2,) && perm_domain == (1,)
    (is_copy || is_adjoint) || throw(
        ArgumentError(
            "a fused graded matrix permute-add allows only an identity copy (`isconj = false`, " *
                "`perm_codomain = (1,)`, `perm_domain = (2,)`) or an adjoint (`isconj = true`, " *
                "`perm_codomain = (2,)`, `perm_domain = (1,)`); got `isconj = $isconj`, " *
                "`perm_codomain = $perm_codomain`, `perm_domain = $perm_domain`"
        )
    )
    return nothing
end

function TensorAlgebra.allocate_output(
        ::typeof(TA.bipermutedimsconj), isconj::Bool, src::AbstractFusedGradedMatrix,
        perm_codomain, perm_domain
    )
    check_input(TA.bipermutedimsconj, isconj, src, perm_codomain, perm_domain)
    perm_codomain == (1,) && return similar(src)
    return similar(src')
end

# Only the identity copy is valid: `conj` would dualize the axis.
function TensorAlgebra.check_input(
        ::typeof(TA.bipermutedimsconj), isconj::Bool,
        ::AbstractFusedGradedVector, perm_codomain, perm_domain
    )
    (!isconj && perm_codomain == (1,) && perm_domain == ()) || throw(
        ArgumentError(
            "a fused graded vector permute-add allows only the identity copy (`isconj = false`, " *
                "`perm_codomain = (1,)`, `perm_domain = ()`); got `isconj = $isconj`, " *
                "`perm_codomain = $perm_codomain`, `perm_domain = $perm_domain`"
        )
    )
    return nothing
end

Base.permutedims(a::AbstractFusedGradedArray, perm) = TensorAlgebra.permutedims(a, perm)

function LinearAlgebra.mul!(
        C::AbstractFusedGradedMatrix, A::AbstractFusedGradedMatrix,
        B::AbstractFusedGradedMatrix,
        α::Number, β::Number
    )
    check_input(mul!, C, A, B)
    dA, dB = sectordata(A), sectordata(B)
    for (s, c) in pairs(sectordata(C))
        if haskey(dA, s) && haskey(dB, s)
            mul!(c, dA[s], dB[s], α, β)
        else
            iszero(β) ? fill!(c, β) : scale!(c, β)
        end
    end
    return C
end

function allocate_output(
        ::typeof(*),
        A::AbstractFusedGradedMatrix,
        B::AbstractFusedGradedMatrix
    )
    cod = axis_codomain(A)
    dom = axis_domain(B)
    Tout = Base.promote_op(*, eltype(A), eltype(B))
    return FusedGradedMatrix{Tout}(undef, cod, dom)
end

function Base.:(*)(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    check_input(*, A, B)
    C = allocate_output(*, A, B)
    return mul!(C, A, B)
end

# Used by MatrixAlgebraKit's SVD-based `left_orth!` / `right_orth!`, which scale the orthogonal
# factor in place by the singular values with `lmul!(S, C)` / `rmul!(C, S)`.
function LinearAlgebra.lmul!(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    check_input(mul!, B, A, B)
    dA = sectordata(A)
    for (s, b) in pairs(sectordata(B))
        LinearAlgebra.lmul!(dA[s], b)
    end
    return B
end
function LinearAlgebra.rmul!(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    check_input(mul!, A, A, B)
    dB = sectordata(B)
    for (s, a) in pairs(sectordata(A))
        LinearAlgebra.rmul!(a, dB[s])
    end
    return A
end

# Equal axes guarantee the same stored sectors.
function Base.:(==)(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    axes(A) == axes(B) || return false
    dA, dB = sectordata(A), sectordata(B)
    return all(dA[c] == dB[c] for c in keys(dA))
end
