struct SectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:SectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorArray}) = SectorFusion()

function TensorAlgebra.matricize(
        ::SectorFusion, a::SectorDelta, ndims_codomain::Val{Ncodomain}
    ) where {Ncodomain}
    biperm = trivialbiperm(ndims_codomain, Val(ndims(a)))
    ax_codomain, ax_domain = blocks(axes(a)[biperm])
    ax_codomain =
        isempty(ax_codomain) ? trivial(sector_type(a)) : tensor_product(ax_codomain...)
    ax_domain =
        isempty(ax_domain) ? trivial(sector_type(a)) : flip(tensor_product(ax_domain...))
    return SectorDelta{eltype(a)}((ax_codomain, ax_domain))
end

function TensorAlgebra.permutedimsopadd!(
        y::SectorArray, op, x::SectorArray, perm,
        α::Number, β::Number
    )
    xsectors = SectorDelta{eltype(x)}(sectors(x))
    ysectors = SectorDelta{eltype(y)}(sectors(y))
    ysectors == permutedims(xsectors, perm) || throw(DimensionMismatch())
    phase = fermion_permutation_phase(xsectors, perm)
    TensorAlgebra.permutedimsopadd!(y.data, op, x.data, perm, phase * α, β)
    return y
end
