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
        ::SectorFusion, a::AbelianGradedArray{<:Any, <:Any, N}, ::Val{K}
    ) where {N, K}
    return TensorAlgebra.matricizeopperm(
        SectorFusion(), identity, a, ntuple(identity, Val(K)),
        ntuple(i -> K + i, Val(N - K))
    )
end

# The block-level piece of `matricizeopperm`: write one stored block into its region of the
# coupled-sector matrix, folding the permute, reshape to 2D, `op`, and fermion sign into a single
# strided in-place write with no intermediate. `perm` reorders the block's legs into the
# destination group order and `phase` is its ±1 fermion sign. The block must be strided (an
# `AbelianGradedArray` stores dense blocks); `op` and `permutedims` ride on the `StridedView`
# lazily.
function matricizeopperm_block!(dst, op, src, phase, perm::NTuple{N, Int}) where {N}
    isstrided(src) ||
        throw(ArgumentError("non-strided blocks are not supported in matricize"))
    grouped = reshape(StridedView(dst), ntuple(i -> size(src, perm[i]), Val(N)))
    grouped .= phase .* op(permutedims(StridedView(src), perm))
    return dst
end

# The block-level piece of `unmatricizeperm!` and the inverse of `matricizeopperm_block!`: write
# one region of the coupled-sector matrix into its destination N-D block, folding the reshape,
# permute back to destination order, and fermion sign into a single in-place write. When the
# block needs neither a reshape nor a permute (a rank-2 factor whose axes already match), a plain
# type-preserving broadcast handles it, which keeps a non-strided factorization block (a
# `Diagonal` `S`) on its own array type. Any other non-strided region would need a permute or
# reshape, which is unsupported.
function unmatricizeperm_block!(dst, src, phase, perm::NTuple{N, Int}) where {N}
    if perm == ntuple(identity, Val(N)) && size(src) == size(dst)
        dst .= phase .* src
        return dst
    end
    isstrided(src) || throw(
        ArgumentError(
            "non-strided blocks needing a permute or reshape are not supported in unmatricize"
        )
    )
    grouped_dims = ntuple(i -> size(dst, invperm(perm)[i]), Val(N))
    grouped = reshape(StridedView(src), grouped_dims)
    StridedView(dst) .= phase .* permutedims(grouped, perm)
    return dst
end

# Build the sector-merged `FusedGradedMatrix` for the bipartition `(perm_codomain, perm_domain)`.
# `op` transforms the fused axes (`conj` dualizes them), and each stored block is scattered
# straight into its coupled-sector matrix slice, carrying `op` and the block's fermion sign.
function TensorAlgebra.matricizeopperm(
        ::SectorFusion, op, a::AbelianGradedArray{T, <:Any, N},
        perm_codomain::Tuple{Vararg{Int}}, perm_domain::Tuple{Vararg{Int}}
    ) where {T, N}
    K = length(perm_codomain)
    K + length(perm_domain) == N || throw(ArgumentError("Invalid bipermutation"))
    S = sectortype(a)
    codomain_axes = ntuple(i -> op(axes(a)[perm_codomain[i]]), Val(K))
    domain_axes = ntuple(i -> op(axes(a)[perm_domain[i]]), Val(N - K))
    unfused_row, unfused_col = unmerged_matricize_axes(S, codomain_axes, domain_axes)
    merged_row = sectormergesort(unfused_row)
    merged_col = sectormergesort(unfused_col)
    # Where each unmerged block lands inside its merged block: `Block(j)[subrange]`.
    row_dest = invblockmergeperm(unfused_row, sectorsortperm(unfused_row), merged_row)
    col_dest = invblockmergeperm(unfused_col, sectorsortperm(unfused_col), merged_col)

    codomain = Dictionary{S, Int}(eachsectoraxis(merged_row), datalengths(merged_row))
    domain = Dictionary{S, Int}(dual.(eachsectoraxis(merged_col)), datalengths(merged_col))
    m = FusedGradedMatrix{T}(undef, codomain, domain)
    # Allowed blocks with no stored source must read back as zero.
    foreach(b -> fill!(b, zero(T)), m.blocks)

    cod_lin = LinearIndices(Tuple(blocklength.(codomain_axes)))
    dom_lin = LinearIndices(Tuple(blocklength.(domain_axes)))
    merged_row_sectors = eachsectoraxis(merged_row)
    perm = (perm_codomain..., perm_domain...)
    for bI_src in eachblockstoredindex(a)
        src = Tuple(bI_src)
        row_fine = cod_lin[ntuple(i -> Int(src[perm_codomain[i]]), Val(K))...]
        col_fine = dom_lin[ntuple(i -> Int(src[perm_domain[i]]), Val(N - K))...]
        row_bir = row_dest[row_fine]
        col_bir = col_dest[col_fine]
        s = merged_row_sectors[Int(Block(row_bir))]
        blk = view(a, bI_src)
        phase =
            fermion_permutation_phase(op, sector(blk), invperm(perm)) *
            fermion_bend_phase(sector(blk), perm_domain)
        matricizeopperm_block!(
            view(m.blocks[s], only(row_bir.indices), only(col_bir.indices)),
            op, data(blk), phase, perm
        )
    end
    return m
end

# ========================  AbelianGradedArray unmatricize  ========================

# `unmatricize` receives the domain axes codomain-facing (un-dualized); a graded array stores
# them dualized, so `conj` re-dualizes them before they are placed.
function TensorAlgebra.unmatricize(
        ::SectorFusion, m::AbstractSectorDelta,
        codomain_axes::Tuple{Vararg{SectorRange}},
        domain_axes::Tuple{Vararg{SectorRange}}
    )
    return AbelianSectorDelta{eltype(m)}((codomain_axes..., conj.(domain_axes)...))
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
    K = length(codomain_axes)
    N = K + length(domain_axes)
    # The domain axes arrive codomain-facing (un-dualized); dualize them into the stored
    # convention with `conj` as they are flattened into the destination axes.
    a = similar(m, (codomain_axes..., conj.(domain_axes)...))
    return TensorAlgebra.unmatricizeperm!(
        SectorFusion(), a, m, ntuple(identity, Val(K)), ntuple(i -> K + i, Val(N - K))
    )
end

# Scatter `m`'s merged sector blocks straight into the N-D destination blocks of `a_dest`,
# folding the inverse permutation into the scatter: the inverse of the `matricizeopperm`
# gather. `(invperm_codomain, invperm_domain)` is the same bipermutation `matricize` used, so
# the matrix's codomain/domain axes sit at those positions of `axes(a_dest)`. Each destination
# block reads the `[rows, cols]` subrange of its coupled sector's matrix, reshapes it to the
# codomain/domain-order block shape, and permutes back to destination order, carrying the
# block's fermion sign.
function TensorAlgebra.unmatricizeperm!(
        ::SectorFusion, a_dest::AbelianGradedArray{<:Any, <:Any, N}, m::FusedGradedMatrix,
        invperm_codomain::Tuple{Vararg{Int}}, invperm_domain::Tuple{Vararg{Int}}
    ) where {N}
    K = length(invperm_codomain)
    K + length(invperm_domain) == N || throw(ArgumentError("Invalid bipermutation"))
    S = sectortype(m)
    codomain_axes = ntuple(i -> axes(a_dest)[invperm_codomain[i]], Val(K))
    domain_axes = ntuple(i -> axes(a_dest)[invperm_domain[i]], Val(N - K))
    unfused_row, unfused_col = unmerged_matricize_axes(S, codomain_axes, domain_axes)
    merged_row = sectormergesort(unfused_row)
    merged_col = sectormergesort(unfused_col)
    row_dest = invblockmergeperm(unfused_row, sectorsortperm(unfused_row), merged_row)
    col_dest = invblockmergeperm(unfused_col, sectorsortperm(unfused_col), merged_col)
    merged_row_sectors = eachsectoraxis(merged_row)

    # Legs land in codomain/domain order; `perm_dest` puts them back to destination order.
    perm_dest = invperm((invperm_codomain..., invperm_domain...))
    # Not every allocated block gets written below, so we zero first.
    zero!(a_dest)
    cod_lin = LinearIndices(Tuple(map(blocklength, codomain_axes)))
    dom_lin = LinearIndices(Tuple(map(blocklength, domain_axes)))
    for bI in eachblockstoredindex(a_dest)
        dest_bk = Int.(Tuple(bI))
        row_fine = cod_lin[ntuple(i -> dest_bk[invperm_codomain[i]], Val(K))...]
        col_fine = dom_lin[ntuple(i -> dest_bk[invperm_domain[i]], Val(N - K))...]
        row_bir = row_dest[row_fine]
        col_bir = col_dest[col_fine]
        s = merged_row_sectors[Int(Block(row_bir))]
        haskey(m.blocks, s) || continue
        slice = view(m.blocks[s], only(row_bir.indices), only(col_bir.indices))
        cd_leg = ntuple(d -> d <= K ? invperm_codomain[d] : invperm_domain[d - K], Val(N))
        cd_sects =
            ntuple(d -> eachsectoraxis(axes(a_dest)[cd_leg[d]])[dest_bk[cd_leg[d]]], Val(N))
        # The block's fermion sign takes `S` from the input: `cd_sects` is empty for a rank-0
        # destination (a full contraction to a scalar) and so carries no `S`.
        cd_delta = AbelianSectorDelta{eltype(slice), S, N}(cd_sects)
        phase =
            fermion_permutation_phase(identity, cd_delta, invperm(perm_dest)) *
            fermion_bend_phase(cd_delta, ntuple(i -> K + i, Val(N - K)))
        unmatricizeperm_block!(data(view(a_dest, bI)), slice, phase, perm_dest)
    end
    return a_dest
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
