# ===========================================================================
#  Graded-array construction and projection surface
# ===========================================================================
# The public `rand`/`randn`/`zeros`/`ones`/`fill` constructors over graded axes, and the projection
# hooks (`unchecked_project`/`infer_aux_space`), all build the concrete graded array (`FusionArray`).
# Kept type-agnostic (dispatched on the axes, not on the array type) so the construction API is
# independent of the concrete array.

# An axis is a `GradedOneTo` or a vector of `sector => multiplicity` pairs (keyed by a
# `SectorRange` or a bare `TensorKitSectors.Sector`), normalized to a `GradedOneTo` by
# `TA.to_range`. Each of `rand`/`randn`/`zeros`/`ones`/`fill` supports three shapes:
#     f(axs...) / f((axs...,))         codomain-only, allocated directly
#     f((cod...), (dom...))            tensor map, `dom` axes stored dual
#     f(flux, (cod...)[, (dom...)])    appends a multiplicity-1 leg carrying `flux`
# A leading graded axis on every tuple/vararg form keeps these from pirating Base's
# zero-argument and `Integer`/`Dims` calls.

# Two anchored `*_map` entries — codomain-led and empty-codomain domain-led — each building the
# `FusionArray` directly, mirroring the `unchecked_project` / `allocate_project` split below. A
# fully empty `((), ())` matches neither.
for f in (:rand, :randn)
    fmap, fbang = Symbol(f, :_map), Symbol(f, :!)
    @eval begin
        function TA.$fmap(
                rng::AbstractRNG, ::Type{T},
                cod::Tuple{GradedOneTo, Vararg{GradedOneTo}}, dom::Tuple{Vararg{GradedOneTo}}
            ) where {T}
            return $fbang(rng, FusionArray{T}(undef, cod, dom))
        end
        function TA.$fmap(
                rng::AbstractRNG, ::Type{T},
                cod::Tuple{}, dom::Tuple{GradedOneTo, Vararg{GradedOneTo}}
            ) where {T}
            return $fbang(rng, FusionArray{T}(undef, cod, dom))
        end
    end
end
for (f, fill_block) in ((:zeros, :(zero!(a))), (:ones, :(fill!(a, one(T)))))
    fmap = Symbol(f, :_map)
    @eval begin
        function TA.$fmap(
                ::Type{T},
                cod::Tuple{GradedOneTo, Vararg{GradedOneTo}}, dom::Tuple{Vararg{GradedOneTo}}
            ) where {T}
            a = FusionArray{T}(undef, cod, dom)
            return $fill_block
        end
        function TA.$fmap(
                ::Type{T}, cod::Tuple{}, dom::Tuple{GradedOneTo, Vararg{GradedOneTo}}
            ) where {T}
            a = FusionArray{T}(undef, cod, dom)
            return $fill_block
        end
    end
end
function TA.fill_map(
        value, cod::Tuple{GradedOneTo, Vararg{GradedOneTo}}, dom::Tuple{Vararg{GradedOneTo}}
    )
    return fill!(FusionArray{typeof(value)}(undef, cod, dom), value)
end
function TA.fill_map(value, cod::Tuple{}, dom::Tuple{GradedOneTo, Vararg{GradedOneTo}})
    return fill!(FusionArray{typeof(value)}(undef, cod, dom), value)
end

# Public `Base` constructors: normalize pairs-vector axes with `to_range` and route to `*_map`.
# Pairs-vector axes are keyed by `SectorRange` (which every GradedArrays sector subtypes); keying
# by a bare `TensorKitSectors.Sector` is not accepted, since overloading `Base` constructors on a
# purely TensorKitSectors signature would be type piracy. Wrap such sectors with `SectorRange`.
for axis_type in (
        :GradedOneTo,
        :(AbstractVector{<:Pair{<:SectorRange, <:Integer}}),
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

# Flux `f(flux, (cod...)[, (dom...)])`: append a multiplicity-1 leg carrying `flux` to the
# dualized domain, so the physical axes fuse to that total charge. The flux may be a `SectorRange`
# or a bare `TensorKitSectors.Sector`: these forms always carry a physical axis (`GradedOneTo` or
# a `SectorRange`-keyed pairs vector), so the signature contains a GradedArrays-owned type and
# accepting a bare sector is not type piracy. The axis-less flux-only forms below stay
# `SectorRange`-only, where a bare-sector method would be piracy.
for axis_type in (
        :GradedOneTo,
        :(AbstractVector{<:Pair{<:SectorRange, <:Integer}}),
    )
    axs_type = :(Tuple{$axis_type, Vararg{$axis_type}})
    for flux_type in (:(TKS.Sector), :SectorRange)
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
# Flux-only forms: no physical axes, just the flux leg. Independent of `axis_type`, so defined
# outside the `axis_type` loop. `f(flux, ())` and `f(flux)` are shorthands for `f(flux, (), ())`,
# mirroring how the codomain-only and empty-domain forms collapse. These dispatch on `SectorRange`
# and so take precedence over `Base.rand`/`zeros`/`fill` on a plain range, returning a graded
# array carrying the flux rather than a plain array over that range.
for f in (:rand, :randn)
    fmap = Symbol(f, :_map)
    @eval begin
        function Base.$f(
                rng::AbstractRNG, ::Type{T}, c::SectorRange, ::Tuple{}, ::Tuple{}
            ) where {T}
            return TA.$fmap(rng, T, (), (to_gradedrange(c),))
        end
        function Base.$f(::Type{T}, c::SectorRange, cod::Tuple{}, dom::Tuple{}) where {T}
            return $f(Random.default_rng(), T, c, cod, dom)
        end
        function Base.$f(rng::AbstractRNG, c::SectorRange, cod::Tuple{}, dom::Tuple{})
            return $f(rng, Float64, c, cod, dom)
        end
        function Base.$f(c::SectorRange, cod::Tuple{}, dom::Tuple{})
            return $f(Random.default_rng(), Float64, c, cod, dom)
        end
        function Base.$f(
                rng::AbstractRNG,
                ::Type{T},
                c::SectorRange,
                dom::Tuple{}
            ) where {T}
            return $f(rng, T, c, (), dom)
        end
        function Base.$f(::Type{T}, c::SectorRange, dom::Tuple{}) where {T}
            return $f(T, c, (), dom)
        end
        function Base.$f(rng::AbstractRNG, c::SectorRange, dom::Tuple{})
            return $f(rng, c, (), dom)
        end
        Base.$f(c::SectorRange, dom::Tuple{}) = $f(c, (), dom)
        function Base.$f(rng::AbstractRNG, ::Type{T}, c::SectorRange) where {T}
            return $f(rng, T, c, ())
        end
        function Base.$f(::Type{T}, c::SectorRange) where {T}
            return $f(T, c, ())
        end
        Base.$f(rng::AbstractRNG, c::SectorRange) = $f(rng, c, ())
        Base.$f(c::SectorRange) = $f(c, ())
    end
end
for f in (:zeros, :ones)
    fmap = Symbol(f, :_map)
    @eval begin
        function Base.$f(::Type{T}, c::SectorRange, ::Tuple{}, ::Tuple{}) where {T}
            return TA.$fmap(T, (), (to_gradedrange(c),))
        end
        function Base.$f(c::SectorRange, cod::Tuple{}, dom::Tuple{})
            return $f(Float64, c, cod, dom)
        end
        function Base.$f(::Type{T}, c::SectorRange, dom::Tuple{}) where {T}
            return $f(T, c, (), dom)
        end
        Base.$f(c::SectorRange, dom::Tuple{}) = $f(Float64, c, (), dom)
        function Base.$f(::Type{T}, c::SectorRange) where {T}
            return $f(T, c, ())
        end
        Base.$f(c::SectorRange) = $f(c, ())
    end
end
@eval begin
    function Base.fill(value, c::SectorRange, ::Tuple{}, ::Tuple{})
        return TA.fill_map(value, (), (to_gradedrange(c),))
    end
    Base.fill(value, c::SectorRange, dom::Tuple{}) = fill(value, c, (), dom)
    Base.fill(value, c::SectorRange) = fill(value, c, ())
end

"""
    zeros([T=Float64,] axs::GradedOneTo...)
    zeros([T=Float64,] (codomain...)[, (domain...)])
    zeros([T=Float64,] flux, (codomain...)[, (domain...)])

Construct a graded array (`FusionArray{T}`) over the given graded axes with every symmetry-allowed
(zero-flux) block allocated and filled with zeros. Each axis may be a `GradedOneTo` or a vector
of `sector => multiplicity` pairs. Passing a `(codomain, domain)` split builds a tensor map,
storing the domain axes dual; a leading `flux` sector appends a multiplicity-1 leg carrying it,
so the physical axes fuse to that total charge.
"""
Base.zeros(::Type{T}, ::Tuple{GradedOneTo, Vararg{GradedOneTo}}) where {T}

"""
    ones([T=Float64,] axs::GradedOneTo...)
    ones([T=Float64,] (codomain...)[, (domain...)])
    ones([T=Float64,] flux, (codomain...)[, (domain...)])

Like [`zeros`](@ref), but filling every symmetry-allowed block with ones.
"""
Base.ones(::Type{T}, ::Tuple{GradedOneTo, Vararg{GradedOneTo}}) where {T}

"""
    fill(v, axs::GradedOneTo...)
    fill(v, (codomain...)[, (domain...)])
    fill(v, flux, (codomain...)[, (domain...)])

Like [`zeros`](@ref), but filling every symmetry-allowed block with `v`
(the element type is taken from `v`).
"""
Base.fill(::Any, ::Tuple{GradedOneTo, Vararg{GradedOneTo}})

# Block-aware diagonal check: block-diagonal (no off-diagonal stored blocks), and each

# Throw unless `sz1` and `sz2` are equal ignoring trailing length-1 axes (an axis beyond one
# size's rank counts as length 1, mirroring `Base.size(A, d)` for `d > ndims(A)`), guarding a
# `reshape` against silently reinterpreting same-length data of a genuinely different shape.
# Vendored from TensorAlgebra rather than reused because it is not part of its public API.
function check_project_size(sz1::Dims, sz2::Dims)
    all(i -> get(sz1, i, 1) == get(sz2, i, 1), 1:max(length(sz1), length(sz2))) || throw(
        DimensionMismatch("sizes $sz1 and $sz2 differ beyond trailing length-1 axes")
    )
    return nothing
end

function TA.unchecked_project(
        raw, codomain_axes::Tuple{GradedOneTo, Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    return unchecked_project_graded(raw, codomain_axes, domain_axes)
end
function TA.unchecked_project(
        raw, codomain_axes::Tuple{}, domain_axes::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    )
    return unchecked_project_graded(raw, codomain_axes, domain_axes)
end
# Project a dense array into the symmetry-allowed subspace as a `FusionArray`, delegating the
# projection to TensorKit over the equivalent `ElementarySpace`s and wrapping the resulting
# `TensorMap`. Unfused / unsorted external axes are first reordered per leg into sorted order (a
# whole-block permutation, so equal sectors become contiguous and the array type is preserved) to
# match the fused-sorted `GradedSpace` TensorKit needs, then wrapped carrying the original axes.
function unchecked_project_graded(raw, codomain_axes, domain_axes)
    all_axes = (codomain_axes..., domain_axes...)
    if ndims(raw) == length(all_axes) && !all(is_fused_sorted, all_axes)
        N = length(all_axes)
        storedlengths = map(g -> Vector(blocklengths(g)), all_axes)
        perms = ntuple(d -> sectorsortperm(all_axes[d]), Val(N))
        sorted = parent(BlockedArray(raw, storedlengths...)[perms...])
        t = TA.unchecked_project(
            sorted,
            map(ElementarySpace ∘ sectormergesort, codomain_axes),
            map(ElementarySpace ∘ sectormergesort, domain_axes)
        )
        return FusionArray(matricize(FusionArray(t)), codomain_axes, domain_axes)
    end
    t = TA.unchecked_project(
        raw, map(ElementarySpace, codomain_axes), map(ElementarySpace, domain_axes)
    )
    return FusionArray(t)
end

# `infer_aux_space` is the only projection hook a graded backend adds beyond `similar_map`:
# `project_aux` derives the auxiliary axis through it, then projects into the full axes. Plain
# `project` needs no graded `allocate_project` override, since the generic strict allocation routes
# through the graded `similar_map`. Both the codomain-led and the (empty-codomain) domain-led cases
# route to `infer_aux_space_graded`.
function TA.infer_aux_space(
        raw, codomain_axes::Tuple{GradedOneTo, Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    return infer_aux_space_graded(raw, codomain_axes, domain_axes)
end
function TA.infer_aux_space(
        raw, codomain_axes::Tuple{}, domain_axes::Tuple{GradedOneTo, Vararg{GradedOneTo}}
    )
    return infer_aux_space_graded(raw, codomain_axes, domain_axes)
end

# Abelian sectors derive one charge per slice, preserving slice order (a direct-sum aux, e.g.
# `[S⁺, Sᶻ, S⁻]` gives `[U1(2), U1(0), U1(-2)]`). A non-abelian multiplet is a single irrep
# spanning the whole slice axis, which the per-slice reading cannot express, so it is derived
# through the `TensorMap` path over the equivalent `ElementarySpace`s and converted back.
function infer_aux_space_graded(raw, codomain_axes, domain_axes)
    if TKS.FusionStyle(first((codomain_axes..., domain_axes...))) isa TKS.UniqueFusion
        return infer_aux_space_abelian(raw, codomain_axes, domain_axes)
    end
    aux = TA.infer_aux_space(
        raw, map(ElementarySpace, codomain_axes), map(ElementarySpace, domain_axes)
    )
    return GradedOneTo(aux)
end

# Abelian sectors are one-dimensional, so each length-1 slice along the aux axis carries a single
# charge (`projected_charge`); contiguous equal charges merge into one sector of that multiplicity
# (matching the `TensorMap` backend), while non-contiguous repeats stay separate to preserve slice
# order.
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

# Net charge of a dense operator, read from its dominant-magnitude entry: find the block
# holding that entry over the stored axes (domain dualized to match `zeros_map`/`similar_map`)
# and fuse that block's per-axis sectors, each with its axis's arrow applied (the same fusion
# `allowedblocks` is built on, so the charge lines up with which blocks `project` keeps). This is
# the abelian per-slice primitive `infer_aux_space_abelian` builds the aux space from; the
# non-abelian derivation lives in the `TensorMap` backend.
function projected_charge(src::AbstractArray, codomain_axes, domain_axes)
    stored = (codomain_axes..., conj.(domain_axes)...)
    src = reshape(src, length.(stored))
    I = Tuple(findmax(abs, src)[2])
    secs = map(stored, I) do ax, i
        return eachsectoraxis(ax)[Int(BlockArrays.findblock(ax, i))]
    end
    return reduce(tensor_product, secs)
end

"""
    getindex(a::AbstractArray, ax1::GradedOneTo, axs::GradedOneTo...)

Construct a graded array (`FusionArray`) by projecting the dense data of `a` onto the
symmetry-allowed blocks of the graded axes `(ax1, axs...)`, via
`TA.project` (which errors if `a` has weight outside
the allowed blocks). `a` is reshaped to `length.((ax1, axs...))` first, so a
trailing size-1 bond can be supplied implicitly. Each axis carries its own arrow,
so index with `dual`/`conj` axes to set duality.
"""
function Base.getindex(a::AbstractArray, ax1::GradedOneTo, axs::GradedOneTo...)
    dest_axes = (ax1, axs...)
    # Match `a` to the requested axes up to trailing length-1 axes: indexing selects exactly these
    # axes, so the surplus-axis derivation branch of `project` must not trigger, and a genuine
    # shape mismatch errors rather than reinterpreting the data.
    check_project_size(size(a), length.(dest_axes))
    return TA.project(reshape(a, length.(dest_axes)), dest_axes)
end
# Disambiguate the single-axis case for a concrete `Array`: `Base.getindex(::Array,
# ::AbstractUnitRange{<:Integer})` and the projection method above are otherwise equally
# specific, so `dense[graded_axis]` (e.g. building a one-leg graded tensor) is ambiguous.
function Base.getindex(a::Array, ax1::GradedOneTo)
    return invoke(getindex, Tuple{AbstractArray, GradedOneTo, Vararg{GradedOneTo}}, a, ax1)
end
