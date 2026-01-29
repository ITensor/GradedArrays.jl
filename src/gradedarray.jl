using BlockArrays: AbstractBlockedUnitRange, BlockedOneTo, blockisequal
using BlockSparseArrays:
    BlockSparseArrays, AbstractBlockSparseMatrix, AnyAbstractBlockSparseArray, BlockSparseArray,
    BlockUnitRange, blocktype, eachblockstoredindex, sparsemortar
using TypeParameterAccessors: similartype, unwrap_array_type
using ArrayLayouts: ArrayLayouts

# Axes
# ----
"""
    const GradedUnitRange{I, R1, R2} = BlockUnitRange{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

Type alias for the axis type of graded arrays. This represents the blocked combination of ranges,
where each block is a `SectorUnitRange` with sector labels of type `I` and underlying range types
`R1` and `R2`.

See also [`SectorUnitRange`](@ref) and [`GradedOneTo`](@ref).
"""
const GradedUnitRange{I, R1, R2} = BlockUnitRange{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

const GradedOneTo{I, R1, R2} = BlockOneTo{Int, Vector{SectorUnitRange{I, R1, R2}}, Vector{Int}}

sectors(r::GradedUnitRange) = sector.(eachblockaxis(r))
sector_multiplicities(r::GradedUnitRange) = sector_multiplicity.(eachblockaxis(r))
sector_type(::Type{<:GradedUnitRange{I}}) where {I} = I

# TODO: this should work:
dual(g::GradedUnitRange) = BlockSparseArrays.BlockUnitRange(g.r, dual.(eachblockaxis(g)))
flip(g::GradedUnitRange) = BlockSparseArrays.BlockUnitRange(g.r, flip.(eachblockaxis(g)))
isdual(g::GradedUnitRange) = isdual(first(eachblockaxis(g)))  # crash for empty. Should not be an issue.

flux(a::AbstractBlockedUnitRange, I::Block{1}) = flux(a[I])

function ×(g1::GradedOneTo, g2::GradedOneTo)
    v = vec([a × b for a in eachblockaxis(g1), b in eachblockaxis(g2)])
    return mortar_axis(v)
end

×(g::GradedUnitRange, a::AbstractUnitRange) = ×(g, to_gradedrange(a))
×(a::AbstractUnitRange, g::GradedUnitRange) = ×(to_gradedrange(a), g)
×(g::GradedUnitRange, a::SectorRange) = ×(g, to_gradedrange(a))
×(a::SectorRange, g::GradedUnitRange) = ×(to_gradedrange(a), g)
function ×(g1::GradedUnitRange, g2::GradedUnitRange)
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
    allequal(isdual, geachblockaxis) || throw(ArgumentError("Cannot combine sectors with different arrows"))
    return blockrange(geachblockaxis)
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


# Specific overloads
# ------------------
# convert Array to SectorArray upon insertion
function Base.setindex!(A::GradedArray{T, N}, value::AbstractArray{T, N}, I::Vararg{Block{1}, N}) where {T, N}
    # TODO: refactor into a function
    sectors = ntuple(dim -> kroneckerfactors(axes(A, dim)[I[dim]], 1), N)
    sarray = SectorArray(sectors, value)
    Base.setindex!(A, sarray, I...)
    return A
end
# this is a copy of the BlockSparseArrays implementation to ensure that is more specific
function Base.setindex!(A::GradedArray{T, N}, value::SectorArray{<:Any, N}, I::Vararg{Block{1}, N}) where {T, N}
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

# constructor utilities
# ---------------------
Base.zeros(elt::Type, axes::NTuple{N, R}) where {N, R <: GradedUnitRange} =
    BlockSparseArrays.blocksparsezeros(elt, axes...)

function BlockSparseArrays.blocksparsezeros(elt::Type, ax1::R, axs::R...) where {R <: GradedUnitRange}
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

checkflux(::AbstractArray, sect) =
    sect == UndefinedFlux() ? nothing : throw(ArgumentError("Inconsistent flux."))
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
function concretetype_to_string_truncated(type::Type; param_truncation_length = typemax(Int))
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
