using ArrayLayouts: ArrayLayouts
using Base.Broadcast: Broadcast as BC
using BlockArrays: AbstractBlockedUnitRange, BlockedOneTo, blockisequal
using BlockSparseArrays: BlockSparseArrays, AbstractBlockSparseMatrix,
    AnyAbstractBlockSparseArray, BlockSparseArray, BlockUnitRange, blocktype,
    eachblockstoredindex, sparsemortar
using FillArrays: Zeros, fillsimilar
using TensorAlgebra: TensorAlgebra, *ₗ, +ₗ, -ₗ, /ₗ, conjed
using TypeParameterAccessors: similartype, unwrap_array_type

# Axes
# ----
"""
    const GradedUnitRange{I, R1, R2} = BlockUnitRange{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

Type alias for the axis type of graded arrays. This represents the blocked combination of ranges,
where each block is a `SectorUnitRange` with sector labels of type `I` and underlying range types
`R1` and `R2`.

See also [`SectorUnitRange`](@ref) and [`GradedOneTo`](@ref).
"""
const GradedUnitRange{I, R1, R2} =
    BlockUnitRange{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

"""
    const GradedOneTo{I, R1, R2} = BlockOneTo{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

See also [`SectorUnitRange`](@ref) and [`GradedUnitRange`](@ref).
"""
const GradedOneTo{I, R1, R2} =
    BlockOneTo{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

sectors(r::GradedUnitRange) = sector.(eachblockaxis(r))
sector_multiplicities(r::GradedUnitRange) = sector_multiplicity.(eachblockaxis(r))
sector_type(::Type{<:GradedUnitRange{I}}) where {I} = I

# TODO: this should work:
dual(g::GradedUnitRange) = BlockSparseArrays.BlockUnitRange(g.r, dual.(eachblockaxis(g)))
flip(g::GradedUnitRange) = BlockSparseArrays.BlockUnitRange(g.r, flip.(eachblockaxis(g)))
isdual(g::GradedUnitRange) = isdual(first(eachblockaxis(g)))  # crash for empty. Should not be an issue.

flux(a::AbstractBlockedUnitRange, I::Block{1}) = flux(a[I])

function KroneckerArrays.:×(g1::GradedOneTo, g2::GradedOneTo)
    v = vec([a × b for a in eachblockaxis(g1), b in eachblockaxis(g2)])
    return mortar_axis(v)
end

KroneckerArrays.:×(g::GradedUnitRange, a::AbstractUnitRange) = ×(g, to_gradedrange(a))
KroneckerArrays.:×(a::AbstractUnitRange, g::GradedUnitRange) = ×(to_gradedrange(a), g)
KroneckerArrays.:×(g::GradedUnitRange, a::SectorRange) = ×(g, to_gradedrange(a))
KroneckerArrays.:×(a::SectorRange, g::GradedUnitRange) = ×(to_gradedrange(a), g)
function KroneckerArrays.:×(g1::GradedUnitRange, g2::GradedUnitRange)
    v = vec([a × b for a in eachblockaxis(g1), b in eachblockaxis(g2)])
    return mortar_axis(v)
end

function space_isequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
    return (isdual(a1) == isdual(a2)) && sectors(a1) == sectors(a2) && blockisequal(a1, a2)
end

function BlockSparseArrays.blockrange(xs::Vector{<:GradedUnitRange})
    baxis = mapreduce(eachblockaxis, vcat, xs)
    return blockrange(baxis) # FIXME this is probably ignoring information somewhere
end

"""
    gradedrange(xs::AbstractVector{<:Pair})

Construct a graded range from the provided list of `sector => range` pairs.
"""
gradedrange(xs::AbstractVector{<:Pair}) = blockrange(map(splat(sectorrange), xs))

function BlockSparseArrays.mortar_axis(geachblockaxis::AbstractVector{<:SectorUnitRange})
    allequal(isdual, geachblockaxis) ||
        throw(ArgumentError("Cannot combine sectors with different arrows"))
    return blockrange(geachblockaxis)
end

# Keep graded labels when slicing by block vectors.
function BlockSparseArrays.blockedunitrange_getindices(
        g::GradedUnitRange, indices::Vector{<:BlockArrays.Block{1}}
    )
    gblocks = map(index -> g[index], indices)
    new_multiplicities = sector_multiplicity.(gblocks)
    new_axis = mortar_axis(sectorrange.(sector.(gblocks), Base.OneTo.(new_multiplicities)))
    return BlockArrays.mortar(gblocks, (new_axis,))
end

function BlockSparseArrays.blockedunitrange_getindices(
        g::GradedUnitRange, indices::BlockArrays.AbstractBlockVector{<:BlockArrays.Block{1}}
    )
    blks = map(bs -> BlockArrays.mortar(map(b -> g[b], bs)), BlockArrays.blocks(indices))
    new_sectors = map(bs -> sectors(g)[Int.(bs)], BlockArrays.blocks(indices))
    @assert all(allequal.(new_sectors))
    new_multiplicities = map(BlockArrays.blocks(indices)) do bs
        return sum(b -> sector_multiplicity(g[b]), bs; init = 0)
    end
    new_axis = mortar_axis(
        sectorrange.(first.(new_sectors), Base.OneTo.(new_multiplicities))
    )
    return BlockArrays.mortar(blks, (new_axis,))
end
function Base.getindex(g::GradedUnitRange, indices::Vector{<:BlockArrays.Block{1}})
    return BlockSparseArrays.blockedunitrange_getindices(g, indices)
end
function Base.getindex(
        g::GradedUnitRange,
        indices::BlockArrays.AbstractBlockVector{<:BlockArrays.Block{1}}
    )
    return BlockSparseArrays.blockedunitrange_getindices(g, indices)
end

to_gradedrange(g::GradedUnitRange) = g

function Base.show(io::IO, g::GradedUnitRange)
    print(io, "GradedUnitRange[")
    join(io, repr.(sectors(g) .=> sector_multiplicities(g)), ", ")
    return print(io, ']')
end

function Base.show(io::IO, ::MIME"text/plain", g::GradedUnitRange)
    println(io, "GradedUnitRange{", sector_type(g), "}")
    return print(io, join(repr.(blocks(g)), '\n'))
end

# Array
# -----

const GradedArray{T, N, I, A, Blocks, Axes <: NTuple{N, GradedUnitRange{I}}} =
    BlockSparseArray{T, N, SectorArray{T, N, I, A}, Blocks, Axes}

const GradedMatrix{T, I, A, Blocks, Axes} = GradedArray{T, 2, A, Blocks, Axes}
const GradedVector{T, I, A, Blocks, Axes} = GradedArray{T, 1, A, Blocks, Axes}

struct GradedStyle{I, N, B <: BC.AbstractArrayStyle{N}} <: BC.AbstractArrayStyle{N}
    blockstyle::B
end
function GradedStyle{I, N}(blockstyle::BC.AbstractArrayStyle{N}) where {I, N}
    return GradedStyle{I, N, typeof(blockstyle)}(blockstyle)
end
function GradedStyle{I, N, B}() where {I, N, B <: BC.AbstractArrayStyle{N}}
    return GradedStyle{I, N}(B())
end
GradedStyle{I, N}(::Val{M}) where {I, N, M} = GradedStyle{I, M}(BC.DefaultArrayStyle{M}())

blockstyle(style::GradedStyle) = style.blockstyle

function BC.BroadcastStyle(arraytype::Type{<:GradedArray{<:Any, N, I}}) where {N, I}
    return GradedStyle{I, N}(BC.BroadcastStyle(blocktype(arraytype)))
end
BC.BroadcastStyle(style::GradedStyle, ::BC.DefaultArrayStyle{0}) = style
BC.BroadcastStyle(::BC.DefaultArrayStyle{0}, style::GradedStyle) = style
function BC.BroadcastStyle(::GradedStyle{I, N}, ::BC.DefaultArrayStyle{N}) where {I, N}
    return BC.DefaultArrayStyle{N}()
end
function BC.BroadcastStyle(::BC.DefaultArrayStyle{N}, ::GradedStyle{I, N}) where {I, N}
    return BC.DefaultArrayStyle{N}()
end
function BC.BroadcastStyle(
        style1::GradedStyle{I, N},
        style2::GradedStyle{I, N}
    ) where {I, N}
    style = BC.result_style(blockstyle(style1), blockstyle(style2))
    return GradedStyle{I, N}(style)
end

function Base.similar(bc::BC.Broadcasted{<:GradedStyle}, elt::Type, ax)
    bc′ = BC.flatten(bc)
    arg = bc′.args[findfirst(arg -> arg isa AbstractArray, bc′.args)]
    return graded_similar(arg, elt, ax)
end

# Specific overloads
# ------------------
# convert Array to SectorArray upon insertion
function Base.setindex!(
        A::GradedArray{T, N},
        value::AbstractArray{T, N},
        I::Vararg{Block{1}, N}
    ) where {T, N}
    # TODO: refactor into a function
    sectors = ntuple(dim -> kroneckerfactors(axes(A, dim)[I[dim]], 1), N)
    sarray = SectorArray(sectors, value)
    Base.setindex!(A, sarray, I...)
    return A
end
# this is a copy of the BlockSparseArrays implementation to ensure that is more specific
function Base.setindex!(
        A::GradedArray{T, N},
        value::SectorArray{<:Any, N},
        I::Vararg{Block{1}, N}
    ) where {T, N}
    if isstored(A, I...)
        # This writes into existing blocks, or constructs blocks
        # using the axes.
        aI = @view! A[I...]
        aI .= value
    else
        # Custom `_convert` works around the issue that
        # `convert(::Type{<:Diagonal}, ::AbstractMatrix)` isnt' defined
        # in Julia v1.10 (https://github.com/JuliaLang/julia/pull/48895,
        # https://github.com/JuliaLang/julia/pull/52487).
        # TODO: Delete `_convert` once we drop support for Julia v1.10.
        blocks(A)[Int.(I)...] = BlockSparseArrays._convert(blocktype(A), value)
    end
    return A
end

function check_graded_broadcast_axes(a::AbstractArray, b::AbstractArray)
    all(dim -> space_isequal(axes(a, dim), axes(b, dim)), 1:ndims(a)) ||
        throw(
        ArgumentError("GradedArray linear broadcasting requires matching graded axes")
    )
    return nothing
end

function graded_broadcast_error(f)
    throw(
        ArgumentError(
            "Only linear broadcast operations are supported for GradedArray, got `$f`."
        )
    )
end

function lazyblock(a::GradedArray{<:Any, N}, I::Vararg{Block{1}, N}) where {N}
    if isstored(a, I...)
        return blocks(a)[Int.(I)...]
    else
        block_ax = map((ax, i) -> eachblockaxis(ax)[Int(i)], axes(a), I)
        return fillsimilar(Zeros{eltype(a)}(block_ax), block_ax)
    end
end
lazyblock(a::GradedArray, I::Block) = lazyblock(a, Tuple(I)...)

TensorAlgebra.@scaledarray_type ScaledGradedArray
TensorAlgebra.@scaledarray ScaledGradedArray
TensorAlgebra.@conjarray_type ConjGradedArray
TensorAlgebra.@conjarray ConjGradedArray
TensorAlgebra.@addarray_type AddGradedArray
TensorAlgebra.@addarray AddGradedArray

const LazyGradedArray = Union{
    GradedArray, ScaledGradedArray, ConjGradedArray, AddGradedArray,
}

function TensorAlgebra.BroadcastStyle_scaled(arrayt::Type{<:ScaledGradedArray})
    return BC.BroadcastStyle(TensorAlgebra.unscaled_type(arrayt))
end
function TensorAlgebra.BroadcastStyle_conj(arrayt::Type{<:ConjGradedArray})
    return BC.BroadcastStyle(TensorAlgebra.conjed_type(arrayt))
end
function TensorAlgebra.BroadcastStyle_add(arrayt::Type{<:AddGradedArray})
    args_type = TensorAlgebra.addends_type(arrayt)
    return Base.promote_op(BC.combine_styles, fieldtypes(args_type)...)()
end

function lazyblock(a::ScaledGradedArray, I::Block)
    return TensorAlgebra.coeff(a) *ₗ lazyblock(TensorAlgebra.unscaled(a), I)
end
function lazyblock(a::ConjGradedArray, I::Block)
    return conjed(lazyblock(conjed(a), I))
end
function lazyblock(a::AddGradedArray, I::Block)
    return +ₗ(map(Base.Fix2(lazyblock, I), TensorAlgebra.addends(a))...)
end

Base.@propagate_inbounds function Base.getindex(a::ScaledGradedArray, I...)
    return TensorAlgebra.coeff(a) * getindex(TensorAlgebra.unscaled(a), I...)
end
Base.@propagate_inbounds function Base.getindex(a::ConjGradedArray, I...)
    return conj(getindex(conjed(a), I...))
end
Base.@propagate_inbounds function Base.getindex(a::AddGradedArray, I...)
    return sum(addend -> getindex(addend, I...), TensorAlgebra.addends(a))
end

graded_eachblockstoredindex(a::GradedArray) = collect(eachblockstoredindex(a))
function graded_eachblockstoredindex(a::ScaledGradedArray)
    return graded_eachblockstoredindex(TensorAlgebra.unscaled(a))
end
graded_eachblockstoredindex(a::ConjGradedArray) = graded_eachblockstoredindex(conjed(a))
function graded_eachblockstoredindex(a::AddGradedArray)
    return unique!(vcat(map(graded_eachblockstoredindex, TensorAlgebra.addends(a))...))
end

function graded_similar(
        a::GradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return similar(a, elt, ax)
end
function graded_similar(
        a::ScaledGradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return graded_similar(TensorAlgebra.unscaled(a), elt, ax)
end
function graded_similar(
        a::ConjGradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    return graded_similar(conjed(a), elt, ax)
end
function graded_similar(
        a::AddGradedArray,
        elt::Type,
        ax::NTuple{N, <:GradedUnitRange}
    ) where {N}
    style = BC.combine_styles(TensorAlgebra.addends(a)...)
    bc = BC.Broadcasted(style, +, TensorAlgebra.addends(a))
    return similar(bc, elt, ax)
end

function copy_lazygraded(a::LazyGradedArray)
    c = graded_similar(a, eltype(a), axes(a))
    for I in graded_eachblockstoredindex(a)
        c[I] = lazyblock(a, I)
    end
    return c
end

function TensorAlgebra.:+ₗ(a::LazyGradedArray, b::LazyGradedArray)
    check_graded_broadcast_axes(a, b)
    return AddGradedArray(a, b)
end
TensorAlgebra.:*ₗ(α::Number, a::GradedArray) = ScaledGradedArray(α, a)
TensorAlgebra.conjed(a::GradedArray) = ConjGradedArray(a)

Base.copy(a::ScaledGradedArray) = copy_lazygraded(a)
Base.copy(a::ConjGradedArray) = copy_lazygraded(a)
Base.copy(a::AddGradedArray) = copy_lazygraded(a)
Base.Array(a::ScaledGradedArray) = Array(copy(a))
Base.Array(a::ConjGradedArray) = Array(copy(a))
Base.Array(a::AddGradedArray) = Array(copy(a))

graded_broadcasted_linear(::typeof(identity), a::GradedArray) = a
graded_broadcasted_linear(::typeof(+), a::GradedArray) = a
graded_broadcasted_linear(f::Base.Fix1{typeof(*), <:Number}, a::GradedArray) = f.x *ₗ a
graded_broadcasted_linear(f::Base.Fix2{typeof(*), <:Number}, a::GradedArray) = a *ₗ f.x
graded_broadcasted_linear(f::Base.Fix2{typeof(/), <:Number}, a::GradedArray) = a /ₗ f.x
function graded_broadcasted_linear(f, args...)
    bc = BC.Broadcasted(f, args)
    TensorAlgebra.is_linear(bc) || graded_broadcast_error(f)
    return TensorAlgebra.to_linear(bc)
end
BC.broadcasted(::GradedStyle, f, args...) = graded_broadcasted_linear(f, args...)

# constructor utilities
# ---------------------
function Base.zeros(elt::Type, axes::NTuple{N, R}) where {N, R <: GradedUnitRange}
    return BlockSparseArrays.blocksparsezeros(elt, axes...)
end

function BlockSparseArrays.blocksparsezeros(
        elt::Type,
        ax1::R,
        axs::R...
    ) where {R <: GradedUnitRange}
    N = length(axs) + 1
    blocktype = SectorArray{elt, N, sector_type(R), Array{elt, N}}
    return BlockSparseArrays.blocksparsezeros(BlockType(blocktype), ax1, axs...)
end

# Flux
# ----
@doc """
    flux(a::AbstractArray)
    flux(a::AbstractArray, I::Block...)

Compute the total flux of an `AbstractArray`, defined as the fusion of all of the incoming charges,
or the flux associated to a provided block.
Whenever the flux cannot be meaningfully computed, for example for non-graded arrays, or empty ones,
this function returns `UndefinedFlux`.
""" flux

struct UndefinedFlux end

flux(::AbstractArray) = UndefinedFlux()

function flux(a::GradedArray{<:Any, N}, I::Vararg{Block{1}, N}) where {N}
    sects = ntuple(N) do d
        return flux(axes(a, d), I[d])
    end
    return ⊗(sects...)
end
flux(a::GradedArray{<:Any, N}, I::Block{N}) where {N} = flux(a, Tuple(I)...)

function flux(a::GradedArray)
    isempty(eachblockstoredindex(a)) && return UndefinedFlux()
    sect = flux(a, first(eachblockstoredindex(a)))
    checkflux(a, sect)
    return sect
end

function checkflux(::AbstractArray, sect)
    return sect == UndefinedFlux() ? nothing : throw(ArgumentError("Inconsistent flux."))
end
function checkflux(a::GradedArray, sect)
    for I in eachblockstoredindex(a)
        flux(a, I) == sect || throw(ArgumentError("Inconsistent flux."))
    end
    return nothing
end

# TODO: Handle this through some kind of trait dispatch, maybe
# a `SymmetryStyle`-like trait to check if the block sparse
# matrix has graded axes.
function Base.axes(a::LinearAlgebra.Adjoint{<:Any, <:GradedArray})
    return dual.(reverse(axes(a')))
end

# show
# ----
# # Copy of `Base.dims2string` defined in `show.jl`.
function dims_to_string(d)
    isempty(d) && return "0-dimensional"
    length(d) == 1 && return "$(d[1])-element"
    return join(map(string, d), '×')
end

# Copy of `BlockArrays.block2string` from `BlockArrays.jl`.
block_to_string(b, s) = string(join(map(string, b), '×'), "-blocked ", dims_to_string(s))

function base_type_and_params(type::Type)
    alias = Base.make_typealias(type)
    base_type, params = if isnothing(alias)
        unspecify_type_parameters(type), type_parameters(type)
    else
        base_type_globalref, params_svec = alias
        base_type_globalref.name, params_svec
    end
    return base_type, params
end

function base_type_and_params(type::Type{<:GradedArray})
    return :GradedArray, type_parameters(type)
end
function base_type_and_params(type::Type{<:GradedVector})
    params = type_parameters(type)
    params′ = [params[1:1]..., params[3:end]...]
    return :GradedVector, params′
end
function base_type_and_params(type::Type{<:GradedMatrix})
    params = type_parameters(type)
    params′ = [params[1:1]..., params[3:end]...]
    return :GradedMatrix, params′
end

# Modified version of `BlockSparseArrays.concretetype_to_string_truncated`.
# This accounts for the fact that the GradedArray alias is not defined in
# BlockSparseArrays so for the sake of printing, Julia doesn't show it as
# an alias: https://github.com/JuliaLang/julia/issues/40448
function concretetype_to_string_truncated(
        type::Type;
        param_truncation_length = typemax(Int)
    )
    isconcretetype(type) || throw(ArgumentError("Type must be concrete."))
    base_type, params = base_type_and_params(type)
    str = string(base_type)
    if isempty(params)
        return str
    end
    str *= '{'
    param_strings = map(params) do param
        param_string = string(param)
        if length(param_string) > param_truncation_length
            return "…"
        end
        return param_string
    end
    str *= join(param_strings, ", ")
    str *= '}'
    return str
end

function Base.summary(io::IO, a::GradedArray)
    print(io, block_to_string(blocksize(a), size(a)))
    print(io, ' ')
    print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length = 40))
    return nothing
end

function Base.showarg(io::IO, a::GradedArray, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length = 40))
    return nothing
end
