"""
    AbstractFusedGradedArray{T,S,N} <: AbstractArray{T,N}

Supertype of the fused (coupled-sector-block) graded arrays, [`FusedGradedMatrix`](@ref) and
[`FusedGradedVector`](@ref). Holds the code shared between the two.
"""
abstract type AbstractFusedGradedArray{T, S, N} <: AbstractArray{T, N} end
const AbstractFusedGradedMatrix{T, S} = AbstractFusedGradedArray{T, S, 2}
const AbstractFusedGradedVector{T, S} = AbstractFusedGradedArray{T, S, 1}

using BlockArrays: mortar
using FillArrays: Zeros

# `datatype` (the per-sector block-data array type) is the primitive each concrete fused array
# defines; `blocktype` wraps it in the sector-block wrapper (`FusedSectorMatrix` for a matrix,
# `FusedSectorVector` for a vector), uniform across the family.
datatype(a::AbstractFusedGradedArray) = datatype(typeof(a))
sectortype(::Type{<:AbstractFusedGradedArray{T, S}}) where {T, S} = S

function blocktype(::Type{A}) where {A <: AbstractFusedGradedMatrix}
    return FusedSectorMatrix{eltype(A), sectortype(A), datatype(A)}
end
function blocktype(::Type{A}) where {A <: AbstractFusedGradedVector}
    return FusedSectorVector{eltype(A), sectortype(A), datatype(A)}
end
blocktype(a::AbstractFusedGradedArray) = blocktype(typeof(a))

# The one storage accessor each variant overloads: the sector → block-data map. The shared matrix
# algebra reads through it instead of raw fields, so a lazy adjoint or diagonal factor needs no faked
# fields. Everything axis-related derives from `biaxes` (the per-variant core), below.
function sectordata end

# Sectors to iterate. Single arg: the stored sectors. Varargs: their union across arguments
# (analogous to `eachindex(A...)`). Returns an iterator; `sectors` is the vector-returning query.
eachsector(a::AbstractFusedGradedArray) = keys(sectordata(a))
function eachsector(a::AbstractFusedGradedArray, as::AbstractFusedGradedArray...)
    return union(eachsector(a), eachsector.(as)...)
end

# Per-sector data, strict and lenient. Strict `sectordata(a, c)` returns the stored block's data and
# throws if the sector is absent; lenient `getsectordata(a, c)` allocates a zero-size block on a miss
# (the `get` prefix marks the possible allocation).
sectordata(a::AbstractFusedGradedArray, c) = sectordata(a)[c]
function getsectordata(a::AbstractFusedGradedArray, c)
    return get(sectordata(a), c) do
        return similar(valtype(sectordata(a)), getsectordataaxes(a, c))
    end
end

# The data axes of sector `c`'s block, one lenient per-dimension data axis, used only by
# `getsectordata` to size an absent (zero) block. Dual-invariant: `sectordatalengths` reads the
# stored per-sector lengths, which `dual` leaves unchanged, so `axes(a)` (with a possibly dualized
# domain) gives the same sizes as the un-dualized codomain/domain ranges.
function getsectordataaxes(a::AbstractFusedGradedArray, c)
    return map(ax -> getsectordataaxis(ax, c), axes(a))
end

# Each concrete type implements the bipartite-axes primitives `axes_codomain`/`axes_domain` (its
# codomain and domain axis groups, un-dualized). The derived `biaxes`/`axis_codomain`/`axis_domain`
# generics live in `tensoralgebra.jl` (overloaded on `AbstractArray`, since `GradedArray` shares them
# but is not an `AbstractFusedGradedArray`).

function isblockdiagonal(A::AbstractFusedGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        row == col || return false
    end
    return true
end

# Diagonal when block-diagonal and every stored (diagonal) block is itself diagonal. Checking blocks
# avoids densifying a block-sparse matrix through the generic dense-slicing fallback.
function LinearAlgebra.isdiag(A::AbstractFusedGradedMatrix)
    for bI in eachblockstoredindex(A)
        row, col = Tuple(bI)
        (row == col && LinearAlgebra.isdiag(view(A, bI))) || return false
    end
    return true
end

# ---------------------------------------------------------------------------
#  axes / size — derived from the per-variant `biaxes` core
# ---------------------------------------------------------------------------

Base.axes(a::AbstractFusedGradedArray) = Tuple(biaxes(a))
Base.size(a::AbstractFusedGradedArray) = map(length, axes(a))

# ---------------------------------------------------------------------------
#  Block indexing — one stored block per coupled sector (block-diagonal)
#
#  A fused graded matrix is block-diagonal in sector space: block (i, j) is stored only when the
#  codomain sector at `i` equals the domain sector at `j`. Both the block view and the stored-index
#  set read through the `sectordata` / `axis_codomain` / `axis_domain` accessors, so every matrix
#  variant (dense, diagonal, lazy adjoint) is covered by these.
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
    cod = sectordatalengths(axis_codomain(m))
    dom = sectordatalengths(axis_domain(m))
    return (
        Block(gettoken(cod, c)[2][2], gettoken(dom, c)[2][2]) for
            c in keys(sectordata(m))
    )
end

# ---------------------------------------------------------------------------
#  Block-diagonal reductions and predicates
#
#  Fold over the stored blocks (each a `FusedSectorMatrix` that already carries its coupled sector's
#  quantum-dimension weight and within-block structural zeros). The remaining structural zeros are the
#  off-sector (symmetry-forbidden) positions.
# ---------------------------------------------------------------------------

# Sum the per-block stored counts; the rest of `length` are structural zeros. Without this the
# `AbstractArray` fallback reports `length` (i.e. fully dense).
function storedlength(A::AbstractFusedGradedMatrix)
    return sum(B -> storedlength(view(A, B)), eachblockstoredindex(A); init = 0)
end

# `sum` restricts to zero-preserving `f` (the forbidden positions would each add `f(0)`);
# `maximum`/`minimum` fold a single `f(0)` when any forbidden position is present.
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

# The full-matrix trace is the sum of the per-coupled-sector block traces (each `FusedSectorMatrix`
# trace carries the sector's quantum-dimension weight), matching `tr(Array(A))` without scalar-indexing.
function LinearAlgebra.tr(A::AbstractFusedGradedMatrix)
    return sum(
        bI -> LinearAlgebra.tr(view(A, bI)), eachblockstoredindex(A);
        init = zero(eltype(A))
    )
end

# Block-wise predicates over the stored (block-diagonal) blocks.
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
#  copy / copyto! — block-wise (the generic AbstractArray fallbacks scalar-index)
# ---------------------------------------------------------------------------

# Block-wise `copyto!`: copy each stored block's data across. It conservatively requires equal axes,
# so it is self-guarding on a direct call; Base's generic `copy!(::AbstractArray, ::AbstractArray)`
# also checks axes before delegating here, so `copy!` is available for free with the same contract.
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
#  Matrix display — `N×M` block grid with the stored coupled sectors listed
# ---------------------------------------------------------------------------

function Base.summary(io::IO, m::AbstractFusedGradedMatrix)
    sd = sectordata(m)
    print(
        io, blocklength(axis_codomain(m)), "×", blocklength(axis_domain(m)), " ",
        summary_typename(typeof(m)),
        " with ", length(sd), " stored block", length(sd) == 1 ? "" : "s", " at sectors ["
    )
    join(io, keys(sd), ", ")
    print(io, "]")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::AbstractFusedGradedMatrix)
    summary(io, m)
    println(io, ":")
    for (d, g) in pairs(axes(m))
        print(io, "  Dim $d: ")
        show_axis(io, g)
        println(io)
    end
    isempty(sectordata(m)) && return nothing
    Base.print_array(io, m)
    return nothing
end

function Base.show(io::IO, m::AbstractFusedGradedMatrix)
    print(
        io, blocklength(axis_codomain(m)), "×", blocklength(axis_domain(m)), " ",
        summary_typename(typeof(m)), " (", length(sectordata(m)), " stored)"
    )
    return nothing
end

# ---------------------------------------------------------------------------
#  Matrix functions — block-diagonal, so `f(A) = blkdiag(f(blk_i))`
#
#  Any matrix function applies block-wise (`sqrt`, `exp`, `log`, …), routing around the generic
#  `LinearAlgebra` impls that scalar-index for triangular / Hermitian detection. Returns a
#  materialized `FusedGradedMatrix`. Per-block result eltypes may differ (e.g. `sqrt(::Matrix{Float64})`
#  returns `Matrix{ComplexF64}` via Schur even when each block is real-PSD), so unify to the
#  `promote_type` of all returned blocks before reconstructing.
# ---------------------------------------------------------------------------

# `T` must arrive as a type parameter so the `convert` target is concrete. Splicing a runtime `T`
# straight into the `convert` (inlining this into the loop body below) makes Julia 1.10 widen the
# block container to an abstract `AbstractArray` and reconstruction throws a `TypeError` (#175), so
# this stays a separate `::Type{T}` method.
function convert_eltypes(::Type{T}, arrays) where {T}
    return map(array -> convert(AbstractArray{T}, array), arrays)
end

for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval function Base.$f(A::AbstractFusedGradedMatrix)
        raw = map(Base.$f, sectordata(A))
        T = mapreduce(eltype, promote_type, raw; init = eltype(A))
        return fusedgradedmatrix(
            convert_eltypes(T, raw), axis_codomain(A), axis_domain(A)
        )
    end
end

# ---------------------------------------------------------------------------
#  fill! / zero! / scale! — block-wise over the stored blocks
#
#  Defined once via the `eachblockstoredindex`/`view` interface every
#  `AbstractFusedGradedArray` implements, so both fused subtypes are covered.
#  These only touch stored (symmetry-allowed) blocks, so a nonzero `fill!`
#  value leaves the forbidden positions at zero.
# ---------------------------------------------------------------------------

function TensorAlgebra.scale!(a::AbstractFusedGradedArray, β::Number)
    for bI in eachblockstoredindex(a)
        scale!(view(a, bI), β)
    end
    return a
end

# The `LinearAlgebra` spelling of blockwise scaling (the generic fallback
# scalar-indexes).
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

# Linear-combination arithmetic, as `.`-broadcasts over the stored blocks. `GradedArray` defines its
# own split-preserving versions in `gradedarray.jl`.
Base.:+(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray) = a .+ b
Base.:-(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray) = a .- b
Base.:*(a::AbstractFusedGradedArray, x::Number) = a .* x
Base.:*(x::Number, a::AbstractFusedGradedArray) = x .* a
Base.:/(a::AbstractFusedGradedArray, x::Number) = a ./ x

# ---------------------------------------------------------------------------
#  Display — render through a BlockArrays block array. BlockArrays draws the
#  block grid; unstored blocks become `Zeros`, which print as `⋅`.
# ---------------------------------------------------------------------------

# Compact type name for the summary line. The buffer-backed fused arrays carry `{T,S,V}` — element,
# sector, and storage-buffer types. Only the element `T` and the storage buffer `V` are informative in
# the header (the sector is spelled out in the `Dim` lines below), so keep the first and last type
# parameters and elide the middle to `…`.
function summary_typename(type::Type{<:AbstractFusedGradedArray})
    alias = Base.make_typealias(type)
    base, params = if isnothing(alias)
        string(nameof(type)), collect(type.parameters)
    else
        globalref, alias_params = alias
        string(globalref.name), collect(alias_params)
    end
    isempty(params) && return base
    strs = if length(params) <= 2
        map(string, params)
    else
        [string(first(params)), "…", string(last(params))]
    end
    return string(base, "{", join(strs, ", "), "}")
end

function _to_blockarray(a::AbstractFusedGradedArray{T, <:Any, N}) where {T, N}
    blens = map(blocklengths, axes(a))
    blockmat = Array{AbstractArray{T, N}, N}(undef, map(length, blens)...)
    # Unstored blocks render as `Zeros` (printed as `⋅`); stored blocks carry their data.
    for I in CartesianIndices(blockmat)
        b = Tuple(I)
        blockmat[I] = Zeros{T}(ntuple(d -> blens[d][b[d]], N)...)
    end
    for bI in eachblockstoredindex(a)
        blk = view(a, bI)
        blockmat[CartesianIndex(Int.(Tuple(bI)))] =
            kron_nd(collect(data(blk)), Array(sector(blk)))
    end
    return mortar(blockmat)
end

function Base.print_array(io::IO, a::AbstractFusedGradedArray)
    return Base.print_array(io, _to_blockarray(a))
end

# Materialize into a dense `Array` (the generic fallback copies elementwise, which scalar-indexes).
# `_to_blockarray` reintroduces each block's structural factor (`reduced ⊗ I`), the identity for
# abelian sectors but a repeat over the irrep's quantum dimension for non-abelian ones.
Base.Array(a::AbstractFusedGradedArray) = Array(_to_blockarray(a))

# Block-diagonal inner product: sum the inner products of the stored blocks, each a sector array
# whose own `dot` carries the quantum-dimension weight of its coupled sector (unit weight for
# abelian sectors). Matching axes mean matching allocated blocks (every allowed block is stored),
# so iterating one operand's stored indices lines up one-to-one with the other's.
function LinearAlgebra.dot(a::AbstractFusedGradedArray, b::AbstractFusedGradedArray)
    axes(a) == axes(b) ||
        throw(DimensionMismatch("dot axes mismatch: a $(axes(a)), b $(axes(b))"))
    init = zero(LinearAlgebra.dot(zero(eltype(a)), zero(eltype(b))))
    return sum(eachblockstoredindex(a); init) do I
        return LinearAlgebra.dot(view(a, I), view(b, I))
    end
end

# Block-diagonal `p`-norm: the stored blocks have disjoint support, so the `p`-th powers add (a
# `max` at `p == Inf`), and each block is a sector array carrying its own quantum-dimension weight.
# This is the `BlockSparseArrays` block reduction, with the `Inf` case handled correctly (unlike the
# `p`-sum formula, which collapses to `1` there).
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

# `LinearAlgebra.normalize` infers its result eltype via `typeof(first(a)/nrm)`, which scalar-indexes
# opaque block storage; route through the graded `/` instead.
function LinearAlgebra.normalize(a::AbstractFusedGradedArray, p::Real = 2)
    return a / LinearAlgebra.norm(a, p)
end

# `conj` would dualize the sectors and axes, flipping the first axis to dual, which the fused storage
# types disallow. Conjugate the `GradedArray` (or matricize) instead.
Base.conj(a::AbstractFusedGradedArray) = throw_flips_first_axis(conj, a)

# `real`/`imag` act on the reduced data of each stored block, leaving the (real) structural sector
# factor untouched (`f(I ⊗ A) = I ⊗ f(A)`). Unlike `conj` they are not semilinear, so they cannot go
# through the linear-broadcast fold; each block delegates to the block-level `AbstractSectorArray`
# method.
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

# Block-aware random fills: dispatch to each stored block's `rand!`/`randn!`, bypassing the generic
# `AbstractArray` fallbacks that go through (disallowed) scalar indexing. The 3-arg
# `rand!(rng, a, sp)` form is what Random's `rand!` entry points ultimately call.
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
# Matrix / linear-algebra operations live on the matrix storage type `FusedGradedMatrix`. On a fused
# vector they would otherwise fall through to the generic `AbstractArray` methods, which scalar-index
# into a dense, non-graded result. Error instead. `transpose` is guarded on the whole
# `AbstractFusedGradedArray` because, unlike `adjoint`, it is unavailable on the matrix too.
Base.transpose(A::AbstractFusedGradedArray) = _matrix_op_error(transpose, A)
Base.adjoint(A::AbstractFusedGradedVector) = _matrix_op_error(adjoint, A)
Base.:*(A::AbstractFusedGradedVector, B::AbstractFusedGradedVector) = _matrix_op_error(*, A)
LinearAlgebra.tr(A::AbstractFusedGradedVector) = _matrix_op_error(LinearAlgebra.tr, A)
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(A::AbstractFusedGradedVector) = _matrix_op_error($f, A)
end

# ============================  similar_map  ============================
# `similar_map` with explicit axes off a fused prototype previously allocated a `GradedArray`; that
# implicitly crossed the `FusedGradedMatrix` (internal) / `GradedArray` (external) boundary, so it is
# now undefined. A fused matrix permutes through `permutedims` / `permutedimsop` (staying fused via
# `allocate_output(permutedimsop, ::AbstractFusedGradedMatrix, …)` below).
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
# Block-wise matrix operations on the abstract fused graded matrix. They read the storage through the
# `sectordata` / `axis_codomain` / `axis_domain` accessors (which every storage variant provides), so
# a lazy adjoint or a diagonal factor works as an operand without materializing. The mutated operand
# may be any fused graded matrix; an incompatible one (e.g. a diagonal destination for a non-diagonal
# product) fails at the block level.

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

# A fused graded matrix permute-add keeps the first (codomain) axis non-dual only for an identity copy or
# an adjoint; any other `(op, perm)` would bend or dualize it, which fused storage cannot represent.
function TensorAlgebra.check_input(
        ::typeof(TA.permutedimsop), op,
        ::AbstractFusedGradedMatrix, perm_codomain, perm_domain
    )
    is_copy = op === identity && perm_codomain == (1,) && perm_domain == (2,)
    is_adjoint = op === conj && perm_codomain == (2,) && perm_domain == (1,)
    (is_copy || is_adjoint) || throw(
        ArgumentError(
            "a fused graded matrix permute-add allows only an identity copy (`op = identity`, " *
                "`perm_codomain = (1,)`, `perm_domain = (2,)`) or an adjoint (`op = conj`, " *
                "`perm_codomain = (2,)`, `perm_domain = (1,)`); got `op = $op`, " *
                "`perm_codomain = $perm_codomain`, `perm_domain = $perm_domain`"
        )
    )
    return nothing
end

# A fused graded matrix permute stays fused with the same axes and eltype (`check_input` allows only
# the identity copy or, for a square operand, the adjoint), so allocate with `similar`. Covers the
# diagonal too, via `AbstractFusedGradedMatrix`.
function TensorAlgebra.allocate_output(
        ::typeof(TA.permutedimsop), op, src::AbstractFusedGradedMatrix, perm_codomain,
        perm_domain
    )
    check_input(TA.permutedimsop, op, src, perm_codomain, perm_domain)
    return similar(src)
end

# A fused graded vector has a single canonical non-dual axis, so only the identity copy is valid: `conj`
# would dualize it, and there is no rank-preserving transpose.
function TensorAlgebra.check_input(
        ::typeof(TA.permutedimsop), op,
        ::AbstractFusedGradedVector, perm_codomain, perm_domain
    )
    (op === identity && perm_codomain == (1,) && perm_domain == ()) || throw(
        ArgumentError(
            "a fused graded vector permute-add allows only the identity copy (`op = identity`, " *
                "`perm_codomain = (1,)`, `perm_domain = ()`); got `op = $op`, " *
                "`perm_codomain = $perm_codomain`, `perm_domain = $perm_domain`"
        )
    )
    return nothing
end

# Route through `permutedimsop` so the permute allocates via `allocate_output` and validates via
# `check_input`, instead of the generic path through `similar_map`.
function TensorAlgebra.permutedims(a::AbstractFusedGradedArray, perm)
    return TA.permutedimsop(identity, a, perm, ())
end
function TensorAlgebra.permutedims(a::AbstractFusedGradedArray, perm_codomain, perm_domain)
    return TA.permutedimsop(identity, a, perm_codomain, perm_domain)
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

# MatrixAlgebraKit's SVD-based `left_orth!` / `right_orth!` fold the singular values into the
# orthogonal factor in place with `lmul!(S, C)` / `rmul!(C, S)`, where `S` is the (diagonal)
# singular-value matrix. The scalar-argument `lmul!` / `rmul!` above do not cover this two-matrix
# form, so define it block-wise: each stored sector block delegates to the `LinearAlgebra` method for
# that block pair, an in-place row / column scaling for the diagonal `S` blocks the factorizations
# feed in. The `check_input(mul!, ...)` call validates the contracted axes and that the product fits
# the mutated operand (the operand plays the role of the `mul!` destination `C`: `B` for `lmul!`, `A`
# for `rmul!`), so the block sectors line up by construction.
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

# Compare coupled-sector blocks directly instead of falling back to element-wise iteration (which
# would index forbidden blocks). Equal axes guarantee the same stored (allowed) sectors. Broad enough
# to compare a lazy adjoint or diagonal factor (whose blocks are lazy views).
function Base.:(==)(A::AbstractFusedGradedMatrix, B::AbstractFusedGradedMatrix)
    axes(A) == axes(B) || return false
    dA, dB = sectordata(A), sectordata(B)
    return all(dA[c] == dB[c] for c in keys(dA))
end
