# A symmetric block that factorizes as the Kronecker product of a structural factor fixed by the
# symmetry (`structure`) and the reduced matrix elements (`data`). Its `size` is the full extent of
# the product, not of the stored data.
abstract type AbstractSectorArray{T, S, N} <: AbstractArray{T, N} end

sectortype(::Type{<:AbstractSectorArray{T, S}}) where {T, S} = S

data(sa::AbstractSectorArray) = sa.data

# The structural factor of a single-sector object, the part fixed by the symmetry, so that
# `x == structure(x) ⊗ data(x)`.
function structure end

# Inverse of the `structure` / `data` split.
function sector_kron end

structureaxes(x) = axes(structure(x))
structureaxes(x, d::Int) = structureaxes(x)[d]
dataaxes(sa::AbstractSectorArray) = axes(data(sa))
function biaxes(sa::AbstractSectorArray)
    sbi = biaxes(structure(sa))
    datacod, datadom = bipartition(dataaxes(sa), Val(length(sbi.t1)))
    return BiTuple(map(SectorOneTo, sbi.t1, datacod), map(SectorOneTo, sbi.t2, datadom))
end
Base.axes(sa::AbstractSectorArray) = Tuple(biaxes(sa))

Base.size(sa::AbstractSectorArray) = map(length, axes(sa))

Base.similar(a::AbstractSectorArray, ::Type{T}) where {T} = similar(a, T, axes(a))

# Scalar indexing reads the reduced data, which equals the full array only for unique (abelian)
# fusion. Otherwise it would read out of bounds.
Base.@propagate_inbounds function Base.getindex(
        A::AbstractSectorArray{T, <:Any, N},
        I::Vararg{Int, N}
    ) where {T, N}
    assert_scalar_indexing()
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    return @inbounds data(A)[I...]
end
Base.@propagate_inbounds function Base.setindex!(
        A::AbstractSectorArray{T, <:Any, N},
        v,
        I::Vararg{Int, N}
    ) where {T, N}
    assert_scalar_indexing()
    require_unique_fusion(A)
    @boundscheck checkbounds(A, I...)
    @inbounds data(A)[I...] = v
    return A
end

# A function of our own rather than `Base.copyto!` methods against `AbstractArray`, to avoid
# method ambiguities with other array types.

# The flat axes carry no codomain/domain split, so a `(2, 0)` block can copy into a `(1, 1)` one.
function copy_sector!(dest::AbstractSectorArray, src::AbstractSectorArray)
    axes(dest) == axes(src) || throw(
        DimensionMismatch("sector axes mismatch in copy: $(axes(dest)) vs $(axes(src))")
    )
    copyto!(data(dest), data(src))
    return dest
end
# A plain array is the full array, which equals the reduced data only under unique fusion.
function copy_sector!(dest::AbstractSectorArray, src::AbstractArray)
    require_unique_fusion(dest)
    copyto!(data(dest), src)
    return dest
end

Base.copyto!(dest::AbstractSectorArray, src::AbstractSectorArray) = copy_sector!(dest, src)
Base.copy!(dest::AbstractSectorArray, src::AbstractSectorArray) = copy_sector!(dest, src)

# The structural factor is diagonal for any fusion, so the matrix is diagonal iff the reduced
# data is.
function LinearAlgebra.isdiag(A::AbstractSectorArray{<:Any, <:Any, 2})
    return LinearAlgebra.isdiag(data(A))
end

# The full array equals the reduced data only under unique fusion.
function Base.all(f::Function, A::AbstractSectorArray)
    require_unique_fusion(A)
    return all(f, data(A))
end
function Base.any(f::Function, A::AbstractSectorArray)
    require_unique_fusion(A)
    return any(f, data(A))
end

# ========================  Shared utilities  ========================

isuniquefusion(A) = TKS.FusionStyle(sectortype(A)) === TKS.UniqueFusion()
function require_unique_fusion(A)
    return isuniquefusion(A) || error("not implemented for non-abelian tensors")
end

# ========================  scale! / zero!  ========================

function TensorAlgebra.scale!(a::AbstractSectorArray, β::Number)
    scale!(data(a), β)
    return a
end

function TensorAlgebra.zero!(a::AbstractSectorArray)
    zero!(data(a))
    return a
end

# Fills the reduced data, like `rand!`/`randn!`, not the dense array. A zero fill uses `zero!`,
# which storage backends can implement more efficiently.
function Base.fill!(a::AbstractSectorArray, v)
    iszero(v) && return zero!(a)
    fill!(data(a), v)
    return a
end

# ========================  matrix operations live on the matrix storage types  ========================
# Otherwise these fall through to the generic `AbstractArray` methods, which scalar-index into a
# dense result. `FusedSectorMatrix` defines its own, more specific methods.
Base.adjoint(a::AbstractSectorArray) = _matrix_op_error(adjoint, a)
Base.transpose(a::AbstractSectorArray) = _matrix_op_error(transpose, a)
Base.:*(a::AbstractSectorArray, b::AbstractSectorArray) = _matrix_op_error(*, a)
function LinearAlgebra.tr(a::AbstractSectorArray{<:Any, <:Any, 2})
    return _matrix_op_error(LinearAlgebra.tr, a)
end
for f in TensorAlgebra.MATRIX_FUNCTIONS
    @eval Base.$f(a::AbstractSectorArray) = _matrix_op_error($f, a)
end

# `dot(A ⊗ B, C ⊗ D) = dot(A, C) * dot(B, D)`, so the structural factor contributes a
# quantum-dimension weight.
function LinearAlgebra.dot(a::AbstractSectorArray, b::AbstractSectorArray)
    return LinearAlgebra.dot(structure(a), structure(b)) *
        LinearAlgebra.dot(data(a), data(b))
end

# `norm(A ⊗ B, p) = norm(A, p) * norm(B, p)` for every `p`, including `Inf`.
function LinearAlgebra.norm(a::AbstractSectorArray, p::Real = 2)
    return LinearAlgebra.norm(structure(a), p) * LinearAlgebra.norm(data(a), p)
end

# ========================  densification  ========================

# `data(a) ⊗ structure(a)`, with the reduced data as the outer (slower) index to match TensorKit's
# dense block layout.
function Base.Array{T, N}(a::AbstractSectorArray{<:Any, <:Any, N}) where {T, N}
    return kron_nd(Array{T, N}(data(a)), Array{T, N}(structure(a)))
end

# ========================  random fills  ========================

# The 3-arg `rand!(rng, a, sp)` form is what Random's `rand!` entry points ultimately call.
function Random.rand!(rng::AbstractRNG, a::AbstractSectorArray, sp::Random.Sampler)
    Random.rand!(rng, data(a), sp)
    return a
end
function Random.randn!(rng::AbstractRNG, a::AbstractSectorArray)
    Random.randn!(rng, data(a))
    return a
end

# ========================  conj  ========================

# Through the broadcast so that the fermionic phase from reversing the leg order is applied, which
# a bare data `conj` drops. Overrides Base's `conj` for real eltypes, which would not dualize the axes.
Base.conj(a::AbstractSectorArray) = conj.(a)

# ========================  real / imag  ========================

# `f(I ⊗ A) = I ⊗ f(A)` since the structural factor is real.
Base.real(a::AbstractSectorArray) = sector_kron(structure(a), real(data(a)))
Base.imag(a::AbstractSectorArray) = sector_kron(structure(a), imag(data(a)))

# ========================  display  ========================

function Base.print_array(io::IO, sa::AbstractSectorArray)
    Base.print_array(io, structure(sa))
    println(io, "\n ⊗")
    Base.print_array(io, data(sa))
    return nothing
end

function Base.show(io::IO, sa::AbstractSectorArray)
    show(io, structure(sa))
    print(io, " ⊗ ")
    show(io, data(sa))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sa::AbstractSectorArray)
    summary(io, sa)
    println(io, ":")
    Base.print_array(io, sa)
    return nothing
end
