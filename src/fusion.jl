struct SectorFusion <: FusionStyle end

# Fusion style for the right factor of a fermionic contraction: matricize as `SectorFusion`
# after twisting the contracted legs (see `contraction_twist!`). A no-op twist for bosonic
# sectors, so it matricizes identically to `SectorFusion` there.
struct TwistedSectorFusion <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:AbstractSectorDelta}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbstractSectorArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:AbstractGradedArray}) = SectorFusion()
TensorAlgebra.FusionStyle(::Type{<:SectorOneTo}) = SectorFusion()

# ========================  trivial_gradedrange  ========================

function trivial_gradedrange(t::Tuple{Vararg{GradedOneTo}})
    return tensor_product(trivial.(t)...)
end
function trivial_gradedrange(::Type{S}) where {S <: SectorRange}
    return gradedrange([trivial(S) => 1])
end

# ========================  unmerged_matricize_axes  ========================

# Fuse a bipartitioned tuple of graded axes into the unmerged 2D row/column axes: one
# block per source-block combination, before `sectormergesort` merges same-sector blocks
# into the final matricized axes. The codomain group fuses as-is; the domain group is
# `flip`ed (same sectors and sizes, opposite arrow) so the matrix reads as a
# `codomain ← domain` map and the matmul pairs contracted legs correctly.
function unmerged_matricize_axes(
        S::Type{<:SectorRange},
        axes_codomain::Tuple{Vararg{GradedOneTo}}, axes_domain::Tuple{Vararg{GradedOneTo}}
    )
    # The trivial-sector init seeds each `reduce`, so a group with no axes (a rank-0
    # codomain or domain, as in a full contraction to a scalar) fuses to the trivial
    # sector. `S` supplies that sector when no axis is present to carry it.
    init = trivial_gradedrange(S)
    ax_codomain = reduce(unmerged_tensor_product, axes_codomain; init)
    ax_domain = flip(reduce(unmerged_tensor_product, axes_domain; init))
    return ax_codomain, ax_domain
end

# ========================  AbelianSectorDelta matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianSectorDelta, ndims_codomain::Val{Ncodomain}
    ) where {Ncodomain}
    ax_codomain = first(bipartition(axes(a), ndims_codomain))
    ax_codomain =
        isempty(ax_codomain) ? trivial(sectortype(a)) : tensor_product(ax_codomain...)
    return SectorIdentity{eltype(a)}(ax_codomain)
end

# ========================  AbelianSectorArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianSectorArray, ndims_codomain::Val{K}
    ) where {K}
    asectors_reshaped = matricize(sector(a), Val(K))
    adata_reshaped = matricize(data(a), Val(K))
    return sector_kron(asectors_reshaped, adata_reshaped)
end

# ========================  SectorFusion AbelianGradedArray matricize  ========================

function TensorAlgebra.matricize(
        ::SectorFusion, a::AbelianGradedArray{T, <:Any, N}, ndims_codomain::Val{K}
    ) where {T, N, K}
    # Gather the stored blocks straight into the sector-merged `FusedGradedMatrix`, rather
    # than materializing the unmerged 2D block array and sector-merging it as a second pass.
    S = sectortype(a)
    unfused_row, unfused_col =
        unmerged_matricize_axes(S, bipartition(axes(a), ndims_codomain)...)
    merged_row = sectormergesort(unfused_row)
    merged_col = sectormergesort(unfused_col)
    # Where each unmerged block lands inside its merged block: `Block(j)[subrange]`.
    row_dest = invblockmergeperm(unfused_row, sectorsortperm(unfused_row), merged_row)
    col_dest = invblockmergeperm(unfused_col, sectorsortperm(unfused_col), merged_col)

    codomain = Dictionary{S, Int}(eachsectoraxis(merged_row), datalengths(merged_row))
    domain = Dictionary{S, Int}(dual.(eachsectoraxis(merged_col)), datalengths(merged_col))
    m = FusedGradedMatrix{T}(undef, codomain, domain)
    # `undef` leaves the per-sector blocks uninitialized; unmerged blocks with no stored
    # source must read back as zero, so zero every block before scattering.
    foreach(b -> fill!(b, zero(T)), m.blocks)

    cod_lin = LinearIndices(Tuple(blocklength.(axes(a)[1:K])))
    dom_lin = LinearIndices(Tuple(blocklength.(axes(a)[(K + 1):N])))
    merged_row_sectors = eachsectoraxis(merged_row)
    for bI_src in eachblockstoredindex(a)
        src = Tuple(bI_src)
        row_fine = cod_lin[ntuple(i -> Int(src[i]), Val(K))...]
        col_fine = dom_lin[ntuple(i -> Int(src[K + i]), Val(N - K))...]
        row_bir = row_dest[row_fine]
        col_bir = col_dest[col_fine]
        s = merged_row_sectors[Int(Block(row_bir))]
        block_2d = matricize(a[bI_src], ndims_codomain)
        m.blocks[s][only(row_bir.indices), only(col_bir.indices)] = data(block_2d)
    end
    return m
end

# ========================  AbelianGradedArray unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::AbstractSectorDelta,
        codomain_axes::Tuple{Vararg{SectorRange}},
        domain_axes::Tuple{Vararg{SectorRange}}
    )
    return AbelianSectorDelta{eltype(m)}((codomain_axes..., domain_axes...))
end

# Unmatricize a 2D sector array back to an N-D AbelianSectorArray. The
# codomain/domain axes must be SectorOneTo (carrying multiplicity info).
# Works for both AbelianSectorMatrix and SectorMatrix.
function TensorAlgebra.unmatricize(
        ::SectorFusion, m::AbstractSectorArray{<:Any, <:Any, 2},
        codomain_axes::Tuple{Vararg{SectorOneTo}},
        domain_axes::Tuple{Vararg{SectorOneTo}}
    )
    msectors = unmatricize(
        sector(m),
        sector.(codomain_axes),
        sector.(domain_axes)
    )
    mdata = unmatricize(
        data(m),
        data.(codomain_axes),
        data.(domain_axes)
    )
    return AbelianSectorArray(msectors, mdata)
end

# ========================  SectorFusion FusedGradedMatrix unmatricize  ========================

function TensorAlgebra.unmatricize(
        ::SectorFusion, m::FusedGradedMatrix,
        codomain_axes::Tuple{Vararg{GradedOneTo}},
        domain_axes::Tuple{Vararg{GradedOneTo}}
    )
    if isempty(codomain_axes) && isempty(domain_axes)
        # Scalar (rank-0) result: only the trivial-sector block contributes, and
        # `cod ∩ dom = {trivial}` for a fused scalar means `m.blocks` is at most
        # one 1×1 entry. Return a 0-D `Array` matching the eltype.
        a = fill(zero(eltype(m)))
        triv = trivial(sectortype(m))
        haskey(m.blocks, triv) && (a[] = m.blocks[triv][1, 1])
        return a
    end
    # Scatter each merged sector block of `m` straight into the N-D destination blocks,
    # the inverse of the `matricize` gather: each destination block reads a `[rows, cols]`
    # subrange of its coupled sector's matrix and reshapes it into the N-D block shape.
    K = length(codomain_axes)
    N = K + length(domain_axes)
    dest_axes = (codomain_axes..., domain_axes...)
    unfused_row, unfused_col =
        unmerged_matricize_axes(sectortype(m), codomain_axes, domain_axes)
    merged_row = sectormergesort(unfused_row)
    merged_col = sectormergesort(unfused_col)
    row_dest = invblockmergeperm(unfused_row, sectorsortperm(unfused_row), merged_row)
    col_dest = invblockmergeperm(unfused_col, sectorsortperm(unfused_col), merged_col)
    merged_row_sectors = eachsectoraxis(merged_row)

    # Not every allocated block gets written below, so we zero first.
    a = zero!(similar(m, dest_axes))
    cod_lin = LinearIndices(Tuple(map(blocklength, codomain_axes)))
    dom_lin = LinearIndices(Tuple(map(blocklength, domain_axes)))
    for bI in eachblockstoredindex(a)
        dest_bk = Int.(Tuple(bI))
        row_fine = cod_lin[ntuple(i -> dest_bk[i], Val(K))...]
        col_fine = dom_lin[ntuple(i -> dest_bk[K + i], Val(N - K))...]
        row_bir = row_dest[row_fine]
        col_bir = col_dest[col_fine]
        s = merged_row_sectors[Int(Block(row_bir))]
        haskey(m.blocks, s) || continue
        sub = m.blocks[s][only(row_bir.indices), only(col_bir.indices)]
        dest_sects = ntuple(d -> eachsectoraxis(dest_axes[d])[dest_bk[d]], Val(N))
        dest_dims = ntuple(d -> blocklengths(dest_axes[d])[dest_bk[d]], Val(N))
        a[bI] = AbelianSectorArray(dest_sects, reshape(sub, dest_dims))
    end
    return a
end

# ========================  Allowed block keys  ========================

function allowedblocks(axs::NTuple{N, GradedOneTo}) where {N}
    N == 0 && return Block{0, Int}[Block()]
    @assert SymmetryStyle(sectortype(eltype(axs))) === AbelianStyle()
    unfused = reduce(axs; init = trivial_gradedrange(axs)) do ax1, ax2
        return unmerged_tensor_product(ax1, ax2)
    end
    cart = CartesianIndices(Tuple(blocklength.(axs)))
    return Block.(Tuple.(cart[findall(istrivial, sectors(unfused))]))
end
