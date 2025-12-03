using BlockArrays: blocks, eachblockaxes1
using BlockSparseArrays: BlockSparseArray, blockreshape
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, BlockReshapeFusion,
    BlockedTuple, FusionStyle, ReshapeFusion, matricize, tensor_product_axis, unmatricize

struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:GradedArray}) = SectorFusion()

function trivial_axis(t::Tuple{Vararg{G}}) where {G <: AbstractGradedUnitRange}
    return trivial(first(t))
end
# heterogeneous sectors
trivial_axis(t::Tuple{Vararg{AbstractGradedUnitRange}}) = ⊗(trivial.(t)...)
# trivial_axis from sector_type
function trivial_axis(::Type{S}) where {S <: SectorRange}
    return gradedrange([trivial(S) => 1])
end

# TODO: Use `TensorAlgebra.matricize_axes`.
function matricize_axes(
        blocked_axes::BlockedTuple{2, <:Any, <:Tuple{Vararg{AbstractUnitRange}}}
    )
    @assert !isempty(blocked_axes)
    default_axis = trivial_axis(Tuple(blocked_axes))
    codomain_axes, domain_axes = blocks(blocked_axes)
    codomain_axis = unmerged_tensor_product(default_axis, codomain_axes...)
    unflipped_domain_axis = unmerged_tensor_product(default_axis, domain_axes...)
    return codomain_axis, flip(unflipped_domain_axis)
end

function TensorAlgebra.trivial_axis(::BlockReshapeFusion, a::GradedArray)
    return trivial_axis(axes(a))
end
function TensorAlgebra.tensor_product_axis(
        ::ReshapeFusion, r1::SectorUnitRange, r2::SectorUnitRange
    )
    return r1 ⊗ r2
end
function TensorAlgebra.tensor_product_axis(
        ::BlockReshapeFusion, r1::AbstractGradedUnitRange, r2::AbstractGradedUnitRange
    )
    (isone(first(r1)) && isone(first(r2))) ||
        throw(ArgumentError("Only one-based axes are supported"))
    blockaxpairs = Iterators.product(eachblockaxes1(r1), eachblockaxes1(r2))
    blockaxs = vec(map(splat(tensor_product_axis), blockaxpairs))
    return mortar_axis(blockaxs)
end
function TensorAlgebra.matricize(
        ::SectorFusion, a::AbstractArray, length_codomain::Val
    )
    a_reshaped = matricize(BlockReshapeFusion(), a, length_codomain)
    return sectormergesort(a_reshaped)
end

using TensorAlgebra: tuplemortar
function TensorAlgebra.unmatricize(
        ::SectorFusion,
        m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    blocked_axes = tuplemortar((codomain_axes, domain_axes))
    if isempty(blocked_axes)
        # Handle edge case of empty blocked_axes, which can occur
        # when matricizing a 0-dimensional array (a scalar).
        a = similar(m, ())
        a[] = only(m)
        return a
    end

    # First, fuse axes to get `sectormergesortperm`.
    # Then unpermute the blocks.
    # TODO: Use `TensorAlgebra.matricize_axes`.
    fused_axes = matricize_axes(blocked_axes)

    blockperms = sectorsortperm.(fused_axes)
    sorted_axes = map((r, I) -> only(axes(r[I])), fused_axes, blockperms)

    # TODO: This is doing extra copies of the blocks,
    # use `@view a[axes_prod...]` instead.
    # That will require implementing some reindexing logic
    # for this combination of slicing.
    m_unblocked = m[sorted_axes...]
    m_blockpermed = m_unblocked[invblockperm.(blockperms)...]
    return unmatricize(FusionStyle(BlockSparseArray), m_blockpermed, blocked_axes)
end

# Sort the blocks by sector and then merge the common sectors.
function sectormergesort(a::AbstractArray)
    I = sectormergesortperm.(axes(a))
    return a[I...]
end
