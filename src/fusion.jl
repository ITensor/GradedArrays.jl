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

# ========================  permutedimsopadd!  ========================

function TensorAlgebra.permutedimsopadd!(
        y::SectorArray, op, x::SectorArray, perm,
        α::Number, β::Number
    )
    xdelta = SectorDelta{eltype(x)}(x.labels, x.isdual)
    ydelta = SectorDelta{eltype(y)}(y.labels, y.isdual)
    ydelta == permutedims(xdelta, perm) || throw(DimensionMismatch())
    phase = fermion_permutation_phase(xdelta, perm)
    TensorAlgebra.permutedimsopadd!(y.data, op, x.data, perm, phase * α, β)
    return y
end

function TensorAlgebra.permutedimsopadd!(
        y::AbelianArray{<:Any, N}, op, x::AbelianArray{<:Any, N}, perm,
        α::Number, β::Number
    ) where {N}
    if !iszero(β)
        for bI in eachblockstoredindex(y)
            y_b = @view!(y[bI])
            idperm = ntuple(identity, ndims(y_b))
            TensorAlgebra.permutedimsopadd!(y_b, identity, y_b, idperm, β, false)
        end
    end
    for bI in eachblockstoredindex(x)
        b = Tuple(bI)
        b_dest = Block(ntuple(i -> b[perm[i]], N))
        y_b = @view!(y[b_dest])
        x_b = x[bI]
        TensorAlgebra.permutedimsopadd!(y_b, op, x_b, perm, α, true)
    end
    return y
end
