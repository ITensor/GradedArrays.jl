# ===========================================================================
#  Graded-array construction and projection
# ===========================================================================

# Each of `rand`/`randn`/`zeros`/`ones`/`fill` supports three forms:
#     f(axs...) / f((axs...,))         codomain only
#     f((cod...), (dom...))            codomain and domain
#     f(flux, (cod...)[, (dom...)])    appends a multiplicity-1 leg carrying `flux`
# Requiring a leading graded axis keeps these from pirating Base's zero-argument and
# `Integer`/`Dims` methods.

# One method requires a codomain axis and the other a domain axis, so a fully empty `((), ())`
# matches neither.
for f in (:rand, :randn)
    fmap, fbang = Symbol(f, :_map), Symbol(f, :!)
    @eval begin
        function TA.$fmap(
                rng::AbstractRNG, ::Type{T},
                cod::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}},
                dom::Tuple{Vararg{AbstractGradedOneTo}}
            ) where {T}
            return $fbang(rng, GradedArray{T}(undef, cod, dom))
        end
        function TA.$fmap(
                rng::AbstractRNG, ::Type{T},
                cod::Tuple{}, dom::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
            ) where {T}
            return $fbang(rng, GradedArray{T}(undef, cod, dom))
        end
    end
end
for (f, fill_block) in ((:zeros, :(zero!(a))), (:ones, :(fill!(a, one(T)))))
    fmap = Symbol(f, :_map)
    @eval begin
        function TA.$fmap(
                ::Type{T},
                cod::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}},
                dom::Tuple{Vararg{AbstractGradedOneTo}}
            ) where {T}
            a = GradedArray{T}(undef, cod, dom)
            return $fill_block
        end
        function TA.$fmap(
                ::Type{T}, cod::Tuple{},
                dom::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
            ) where {T}
            a = GradedArray{T}(undef, cod, dom)
            return $fill_block
        end
    end
end
function TA.fill_map(
        value, cod::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}},
        dom::Tuple{Vararg{AbstractGradedOneTo}}
    )
    return fill!(GradedArray{typeof(value)}(undef, cod, dom), value)
end
function TA.fill_map(
        value,
        cod::Tuple{},
        dom::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
    )
    return fill!(GradedArray{typeof(value)}(undef, cod, dom), value)
end

# Pairs-vector axes must be keyed by a GradedArrays `Sector`, since a bare `TensorKitSectors.Sector`
# key would be type piracy. Other keys `to_range` accepts are left out to limit the number of
# generated methods.
for axis_type in (
        :AbstractGradedOneTo,
        :(AbstractVector{<:Pair{<:Sector, <:Integer}}),
    )
    axs_type = :(Tuple{$axis_type, Vararg{$axis_type}})
    for f in (:rand, :randn)
        fmap = Symbol(f, :_map)
        @eval begin
            function Base.$f(
                    rng::AbstractRNG,
                    ::Type{T},
                    cod::$axs_type,
                    dom::$axs_type
                ) where {T}
                return TA.$fmap(rng, T, map(TA.to_range, cod), map(TA.to_range, dom))
            end
            function Base.$f(::Type{T}, cod::$axs_type, dom::$axs_type) where {T}
                return $f(Random.default_rng(), T, cod, dom)
            end
            function Base.$f(rng::AbstractRNG, cod::$axs_type, dom::$axs_type)
                return $f(rng, Float64, cod, dom)
            end
            function Base.$f(cod::$axs_type, dom::$axs_type)
                return $f(Random.default_rng(), Float64, cod, dom)
            end
            function Base.$f(
                    rng::AbstractRNG,
                    ::Type{T},
                    cod::$axs_type,
                    ::Tuple{}
                ) where {T}
                return TA.$fmap(rng, T, map(TA.to_range, cod), ())
            end
            function Base.$f(::Type{T}, cod::$axs_type, dom::Tuple{}) where {T}
                return $f(Random.default_rng(), T, cod, dom)
            end
            function Base.$f(rng::AbstractRNG, cod::$axs_type, dom::Tuple{})
                return $f(rng, Float64, cod, dom)
            end
            function Base.$f(cod::$axs_type, dom::Tuple{})
                return $f(Random.default_rng(), Float64, cod, dom)
            end
            function Base.$f(rng::AbstractRNG, ::Type{T}, cod::$axs_type) where {T}
                return $f(rng, T, cod, ())
            end
            function Base.$f(::Type{T}, cod::$axs_type) where {T}
                return $f(T, cod, ())
            end
            function Base.$f(rng::AbstractRNG, cod::$axs_type)
                return $f(rng, cod, ())
            end
            Base.$f(cod::$axs_type) = $f(cod, ())
            function Base.$f(
                    rng::AbstractRNG,
                    ::Type{T},
                    ::Tuple{},
                    dom::$axs_type
                ) where {T}
                return TA.$fmap(rng, T, (), map(TA.to_range, dom))
            end
            function Base.$f(::Type{T}, cod::Tuple{}, dom::$axs_type) where {T}
                return $f(Random.default_rng(), T, cod, dom)
            end
            function Base.$f(rng::AbstractRNG, cod::Tuple{}, dom::$axs_type)
                return $f(rng, Float64, cod, dom)
            end
            function Base.$f(cod::Tuple{}, dom::$axs_type)
                return $f(Random.default_rng(), Float64, cod, dom)
            end
            function Base.$f(
                    rng::AbstractRNG,
                    ::Type{T},
                    ax::$axis_type,
                    axs::$axis_type...
                ) where {T}
                return $f(rng, T, (ax, axs...))
            end
            Base.$f(::Type{T}, ax::$axis_type, axs::$axis_type...) where {T} =
                $f(T, (ax, axs...))
            Base.$f(rng::AbstractRNG, ax::$axis_type, axs::$axis_type...) =
                $f(rng, (ax, axs...))
            Base.$f(ax::$axis_type, axs::$axis_type...) = $f((ax, axs...))
        end
    end
    for f in (:zeros, :ones)
        fmap = Symbol(f, :_map)
        @eval begin
            function Base.$f(::Type{T}, cod::$axs_type, dom::$axs_type) where {T}
                return TA.$fmap(T, map(TA.to_range, cod), map(TA.to_range, dom))
            end
            function Base.$f(cod::$axs_type, dom::$axs_type)
                return $f(Float64, cod, dom)
            end
            function Base.$f(::Type{T}, cod::$axs_type, ::Tuple{}) where {T}
                return TA.$fmap(T, map(TA.to_range, cod), ())
            end
            function Base.$f(cod::$axs_type, dom::Tuple{})
                return $f(Float64, cod, dom)
            end
            function Base.$f(::Type{T}, cod::$axs_type) where {T}
                return $f(T, cod, ())
            end
            Base.$f(cod::$axs_type) = $f(cod, ())
            function Base.$f(::Type{T}, ::Tuple{}, dom::$axs_type) where {T}
                return TA.$fmap(T, (), map(TA.to_range, dom))
            end
            function Base.$f(cod::Tuple{}, dom::$axs_type)
                return $f(Float64, cod, dom)
            end
            Base.$f(::Type{T}, ax::$axis_type, axs::$axis_type...) where {T} =
                $f(T, (ax, axs...))
            Base.$f(ax::$axis_type, axs::$axis_type...) = $f((ax, axs...))
        end
    end
    @eval begin
        function Base.fill(value, cod::$axs_type, dom::$axs_type)
            return TA.fill_map(value, map(TA.to_range, cod), map(TA.to_range, dom))
        end
        function Base.fill(value, cod::$axs_type, ::Tuple{})
            return TA.fill_map(value, map(TA.to_range, cod), ())
        end
        Base.fill(value, cod::$axs_type) = fill(value, cod, ())
        function Base.fill(value, ::Tuple{}, dom::$axs_type)
            return TA.fill_map(value, (), map(TA.to_range, dom))
        end
        Base.fill(value, ax::$axis_type, axs::$axis_type...) = fill(value, (ax, axs...))
    end
end

# `f(flux, (cod...)[, (dom...)])` appends a multiplicity-1 domain leg carrying `flux`. A bare
# `TensorKitSectors.Sector` flux is not type piracy here, since these forms always take a graded
# axis. The flux-only forms below take only a `Sector`, since a bare sector would be piracy.
for axis_type in (
        :AbstractGradedOneTo,
        :(AbstractVector{<:Pair{<:Sector, <:Integer}}),
    )
    axs_type = :(Tuple{$axis_type, Vararg{$axis_type}})
    for flux_type in (:(TKS.Sector), :Sector)
        for f in (:rand, :randn)
            fmap = Symbol(f, :_map)
            @eval begin
                function Base.$f(
                        rng::AbstractRNG,
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type,
                        dom::$axs_type
                    ) where {T}
                    return TA.$fmap(
                        rng,
                        T,
                        map(TA.to_range, cod),
                        (map(TA.to_range, dom)..., to_gradedrange(c))
                    )
                end
                function Base.$f(
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type,
                        dom::$axs_type
                    ) where {T}
                    return $f(Random.default_rng(), T, c, cod, dom)
                end
                function Base.$f(
                        rng::AbstractRNG,
                        c::$flux_type,
                        cod::$axs_type,
                        dom::$axs_type
                    )
                    return $f(rng, Float64, c, cod, dom)
                end
                function Base.$f(c::$flux_type, cod::$axs_type, dom::$axs_type)
                    return $f(Random.default_rng(), Float64, c, cod, dom)
                end
                function Base.$f(
                        rng::AbstractRNG,
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type,
                        ::Tuple{}
                    ) where {T}
                    return TA.$fmap(rng, T, map(TA.to_range, cod), (to_gradedrange(c),))
                end
                function Base.$f(
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type,
                        dom::Tuple{}
                    ) where {T}
                    return $f(Random.default_rng(), T, c, cod, dom)
                end
                function Base.$f(
                        rng::AbstractRNG,
                        c::$flux_type,
                        cod::$axs_type,
                        dom::Tuple{}
                    )
                    return $f(rng, Float64, c, cod, dom)
                end
                function Base.$f(c::$flux_type, cod::$axs_type, dom::Tuple{})
                    return $f(Random.default_rng(), Float64, c, cod, dom)
                end
                function Base.$f(
                        rng::AbstractRNG,
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type
                    ) where {T}
                    return $f(rng, T, c, cod, ())
                end
                function Base.$f(::Type{T}, c::$flux_type, cod::$axs_type) where {T}
                    return $f(T, c, cod, ())
                end
                function Base.$f(rng::AbstractRNG, c::$flux_type, cod::$axs_type)
                    return $f(rng, c, cod, ())
                end
                Base.$f(c::$flux_type, cod::$axs_type) = $f(c, cod, ())
                function Base.$f(
                        rng::AbstractRNG,
                        ::Type{T},
                        c::$flux_type,
                        ::Tuple{},
                        dom::$axs_type
                    ) where {T}
                    return TA.$fmap(
                        rng,
                        T,
                        (),
                        (map(TA.to_range, dom)..., to_gradedrange(c))
                    )
                end
                function Base.$f(
                        ::Type{T},
                        c::$flux_type,
                        cod::Tuple{},
                        dom::$axs_type
                    ) where {T}
                    return $f(Random.default_rng(), T, c, cod, dom)
                end
                function Base.$f(
                        rng::AbstractRNG,
                        c::$flux_type,
                        cod::Tuple{},
                        dom::$axs_type
                    )
                    return $f(rng, Float64, c, cod, dom)
                end
                function Base.$f(c::$flux_type, cod::Tuple{}, dom::$axs_type)
                    return $f(Random.default_rng(), Float64, c, cod, dom)
                end
            end
        end
        for f in (:zeros, :ones)
            fmap = Symbol(f, :_map)
            @eval begin
                function Base.$f(
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type,
                        dom::$axs_type
                    ) where {T}
                    return TA.$fmap(
                        T,
                        map(TA.to_range, cod),
                        (map(TA.to_range, dom)..., to_gradedrange(c))
                    )
                end
                function Base.$f(c::$flux_type, cod::$axs_type, dom::$axs_type)
                    return $f(Float64, c, cod, dom)
                end
                function Base.$f(
                        ::Type{T},
                        c::$flux_type,
                        cod::$axs_type,
                        ::Tuple{}
                    ) where {T}
                    return TA.$fmap(T, map(TA.to_range, cod), (to_gradedrange(c),))
                end
                function Base.$f(c::$flux_type, cod::$axs_type, dom::Tuple{})
                    return $f(Float64, c, cod, dom)
                end
                function Base.$f(::Type{T}, c::$flux_type, cod::$axs_type) where {T}
                    return $f(T, c, cod, ())
                end
                Base.$f(c::$flux_type, cod::$axs_type) = $f(c, cod, ())
                function Base.$f(
                        ::Type{T},
                        c::$flux_type,
                        ::Tuple{},
                        dom::$axs_type
                    ) where {T}
                    return TA.$fmap(T, (), (map(TA.to_range, dom)..., to_gradedrange(c)))
                end
                function Base.$f(c::$flux_type, cod::Tuple{}, dom::$axs_type)
                    return $f(Float64, c, cod, dom)
                end
            end
        end
        @eval begin
            function Base.fill(value, c::$flux_type, cod::$axs_type, dom::$axs_type)
                return TA.fill_map(
                    value,
                    map(TA.to_range, cod),
                    (map(TA.to_range, dom)..., to_gradedrange(c))
                )
            end
            function Base.fill(value, c::$flux_type, cod::$axs_type, ::Tuple{})
                return TA.fill_map(value, map(TA.to_range, cod), (to_gradedrange(c),))
            end
            Base.fill(value, c::$flux_type, cod::$axs_type) = fill(value, c, cod, ())
            function Base.fill(value, c::$flux_type, ::Tuple{}, dom::$axs_type)
                return TA.fill_map(value, (), (map(TA.to_range, dom)..., to_gradedrange(c)))
            end
        end
    end
end
# Flux-only forms, with only the flux leg. `f(flux, ())` and `f(flux)` are shorthands for
# `f(flux, (), ())`.
for f in (:rand, :randn)
    fmap = Symbol(f, :_map)
    @eval begin
        function Base.$f(
                rng::AbstractRNG, ::Type{T}, c::Sector, ::Tuple{}, ::Tuple{}
            ) where {T}
            return TA.$fmap(rng, T, (), (to_gradedrange(c),))
        end
        function Base.$f(::Type{T}, c::Sector, cod::Tuple{}, dom::Tuple{}) where {T}
            return $f(Random.default_rng(), T, c, cod, dom)
        end
        function Base.$f(rng::AbstractRNG, c::Sector, cod::Tuple{}, dom::Tuple{})
            return $f(rng, Float64, c, cod, dom)
        end
        function Base.$f(c::Sector, cod::Tuple{}, dom::Tuple{})
            return $f(Random.default_rng(), Float64, c, cod, dom)
        end
        function Base.$f(
                rng::AbstractRNG,
                ::Type{T},
                c::Sector,
                dom::Tuple{}
            ) where {T}
            return $f(rng, T, c, (), dom)
        end
        function Base.$f(::Type{T}, c::Sector, dom::Tuple{}) where {T}
            return $f(T, c, (), dom)
        end
        function Base.$f(rng::AbstractRNG, c::Sector, dom::Tuple{})
            return $f(rng, c, (), dom)
        end
        Base.$f(c::Sector, dom::Tuple{}) = $f(c, (), dom)
        function Base.$f(rng::AbstractRNG, ::Type{T}, c::Sector) where {T}
            return $f(rng, T, c, ())
        end
        function Base.$f(::Type{T}, c::Sector) where {T}
            return $f(T, c, ())
        end
        Base.$f(rng::AbstractRNG, c::Sector) = $f(rng, c, ())
        Base.$f(c::Sector) = $f(c, ())
    end
end
for f in (:zeros, :ones)
    fmap = Symbol(f, :_map)
    @eval begin
        function Base.$f(::Type{T}, c::Sector, ::Tuple{}, ::Tuple{}) where {T}
            return TA.$fmap(T, (), (to_gradedrange(c),))
        end
        function Base.$f(c::Sector, cod::Tuple{}, dom::Tuple{})
            return $f(Float64, c, cod, dom)
        end
        function Base.$f(::Type{T}, c::Sector, dom::Tuple{}) where {T}
            return $f(T, c, (), dom)
        end
        Base.$f(c::Sector, dom::Tuple{}) = $f(Float64, c, (), dom)
        function Base.$f(::Type{T}, c::Sector) where {T}
            return $f(T, c, ())
        end
        Base.$f(c::Sector) = $f(c, ())
    end
end
@eval begin
    function Base.fill(value, c::Sector, ::Tuple{}, ::Tuple{})
        return TA.fill_map(value, (), (to_gradedrange(c),))
    end
    Base.fill(value, c::Sector, dom::Tuple{}) = fill(value, c, (), dom)
    Base.fill(value, c::Sector) = fill(value, c, ())
end

"""
    zeros([T=Float64,] axs...)
    zeros([T=Float64,] (codomain...)[, (domain...)])
    zeros([T=Float64,] flux, (codomain...)[, (domain...)])

Construct a `GradedArray` over the given graded axes with every symmetry-allowed (zero-flux)
block allocated and filled with zeros. Each axis may be a graded range or a vector of
`sector => multiplicity` pairs. Passing a `(codomain, domain)` split builds a tensor map whose
domain axes are dual. A leading `flux` sector appends a multiplicity-1 leg carrying it, so the
other axes fuse to that total charge.
"""
Base.zeros(::Type{T}, ::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}) where {T}

"""
    ones([T=Float64,] axs...)
    ones([T=Float64,] (codomain...)[, (domain...)])
    ones([T=Float64,] flux, (codomain...)[, (domain...)])

Like [`zeros`](@ref), but filling every symmetry-allowed block with ones.
"""
Base.ones(::Type{T}, ::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}) where {T}

"""
    fill(v, axs...)
    fill(v, (codomain...)[, (domain...)])
    fill(v, flux, (codomain...)[, (domain...)])

Like [`zeros`](@ref), but filling every symmetry-allowed block with `v`
(the element type is taken from `v`).
"""
Base.fill(::Any, ::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}})

# Throw unless `sz1` and `sz2` are equal up to trailing length-1 axes. Vendored from
# TensorAlgebra, where it is not public.
function check_project_size(sz1::Dims, sz2::Dims)
    all(i -> get(sz1, i, 1) == get(sz2, i, 1), 1:max(length(sz1), length(sz2))) || throw(
        DimensionMismatch("sizes $sz1 and $sz2 differ beyond trailing length-1 axes")
    )
    return nothing
end

function TA.unchecked_project(
        raw, codomain_axes::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}},
        domain_axes::Tuple{Vararg{AbstractGradedOneTo}}
    )
    return unchecked_project_graded(raw, codomain_axes, domain_axes)
end
function TA.unchecked_project(
        raw, codomain_axes::Tuple{},
        domain_axes::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
    )
    return unchecked_project_graded(raw, codomain_axes, domain_axes)
end
# Projects through TensorKit. Axes with unsorted or repeated sectors are first block-permuted into
# sorted order, as a TensorKit `GradedSpace` requires.
function unchecked_project_graded(raw, codomain_axes, domain_axes)
    # `TA.unchecked_project` allocates a fresh result, so it can be wrapped without copying.
    all_axes = (codomain_axes..., domain_axes...)
    if ndims(raw) == length(all_axes) && !all(is_fused_sorted, all_axes)
        N = length(all_axes)
        storedlengths = map(g -> Vector(blocklengths(g)), all_axes)
        perms = ntuple(d -> sectorsortperm(all_axes[d]), Val(N))
        sorted = parent(BlockedArray(raw, storedlengths...)[perms...])
        t = TA.unchecked_project(
            sorted,
            map(ElementarySpace ∘ fusesectors, codomain_axes),
            map(ElementarySpace ∘ fusesectors, domain_axes)
        )
        return GradedArray(matricize(to_gradedarray(t)), codomain_axes, domain_axes)
    end
    t = TA.unchecked_project(
        raw, map(ElementarySpace, codomain_axes), map(ElementarySpace, domain_axes)
    )
    return to_gradedarray(t)
end

function TA.infer_aux_space(
        raw, codomain_axes::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}},
        domain_axes::Tuple{Vararg{AbstractGradedOneTo}}
    )
    return infer_aux_space_graded(raw, codomain_axes, domain_axes)
end
function TA.infer_aux_space(
        raw, codomain_axes::Tuple{},
        domain_axes::Tuple{AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}
    )
    return infer_aux_space_graded(raw, codomain_axes, domain_axes)
end

# For abelian sectors each slice gets its own charge, in slice order, for example `[S⁺, Sᶻ, S⁻]`
# gives `[U1(2), U1(0), U1(-2)]`. A non-abelian multiplet spans several slices, so it goes through
# TensorKit.
function infer_aux_space_graded(raw, codomain_axes, domain_axes)
    if TKS.FusionStyle(first((codomain_axes..., domain_axes...))) isa TKS.UniqueFusion
        return infer_aux_space_abelian(raw, codomain_axes, domain_axes)
    end
    aux = TA.infer_aux_space(
        raw, map(ElementarySpace, codomain_axes), map(ElementarySpace, domain_axes)
    )
    return GradedOneTo(aux)
end

# Contiguous equal charges merge into one sector, as TensorKit does, while non-contiguous repeats
# stay separate to preserve the slice order.
function infer_aux_space_abelian(src::AbstractArray, codomain_axes, domain_axes)
    aux_dim = length(codomain_axes) + length(domain_axes) + 1
    qs = map(eachslice(src; dims = aux_dim)) do slice
        return projected_charge(slice, codomain_axes, domain_axes)
    end
    ps = Pair{eltype(qs), Int}[]
    for q in qs
        if isempty(ps) || first(ps[end]) != q
            push!(ps, q => 1)
        else
            ps[end] = q => (last(ps[end]) + 1)
        end
    end
    return gradedrange(ps)
end

# Net charge of a dense operator, read from the block holding its largest-magnitude entry.
function projected_charge(src::AbstractArray, codomain_axes, domain_axes)
    stored = (codomain_axes..., conj.(domain_axes)...)
    src = reshape(src, length.(stored))
    I = Tuple(findmax(abs, src)[2])
    secs = map(stored, I) do ax, i
        return eachstructureaxis(ax)[Int(BlockArrays.findblock(ax, i))]
    end
    return reduce(tensor_product, secs)
end

"""
    getindex(a::AbstractArray, ax1, axs...)

Construct a `GradedArray` by projecting the dense data of `a` onto the symmetry-allowed blocks
of the graded axes `(ax1, axs...)`, which errors if `a` has weight outside the allowed blocks.
`a` is reshaped to `length.((ax1, axs...))` first, so a trailing size-1 bond can be supplied
implicitly. Index with `dual` axes to get dual axes in the result.
"""
function Base.getindex(
        a::AbstractArray,
        ax1::AbstractGradedOneTo,
        axs::AbstractGradedOneTo...
    )
    dest_axes = (ax1, axs...)
    # So that `project` does not derive an extra axis, and a shape mismatch errors.
    check_project_size(size(a), length.(dest_axes))
    return TA.project(reshape(a, length.(dest_axes)), dest_axes)
end
# Disambiguate against `Base.getindex(::Array, ::AbstractUnitRange{<:Integer})`.
function Base.getindex(a::Array, ax1::AbstractGradedOneTo)
    return invoke(
        getindex, Tuple{AbstractArray, AbstractGradedOneTo, Vararg{AbstractGradedOneTo}}, a,
        ax1
    )
end
