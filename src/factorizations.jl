using BlockArrays: blocks
using BlockSparseArrays:
    BlockSparseArrays,
    eachblockaxis,
    mortar_axis,
    infimum,
    output_type,
    BlockType,
    blockstoredlength,
    BlockPermutedDiagonalAlgorithm,
    BlockDiagonalAlgorithm,
    BlockDiagonalTruncationStrategy
using LinearAlgebra: Diagonal
using MatrixAlgebraKit:
    MatrixAlgebraKit,
    lq_compact!,
    lq_full!,
    qr_compact!,
    qr_full!,
    svd_compact!,
    svd_full!,
    svd_trunc!,
    left_polar!,
    right_polar!,
    TruncatedAlgorithm,
    PolarViaSVD
using TensorAlgebra: TensorAlgebra

# flux but assume zero if it cannot be obtained
_safe_flux(A) = blockstoredlength(A) > 0 ? flux(A) : trivial(sector_type(axes(A, 1)))

function unfluxify(A, charge; side::Symbol = :domain)
    side === :domain || side === :codomain || throw(ArgumentError("invalid side $(side)"))

    istrivial(charge) && return TensorAlgebra.matricize(A, (1,), (2,))

    if side === :domain
        A′ = similar(A, (axes(A, 1), axes(A, 2), to_gradedrange(dual(charge))))
        for I in eachblockstoredindex(A)
            bI = A[I]
            A′[I, Block(1)] = reshape(bI, size(bI)..., 1)
        end

        return TensorAlgebra.matricize(A′, (1,), (2, 3))
    else
        A′ = similar(A, (to_gradedrange(dual(charge)), axes(A, 1), axes(A, 2)))
        for I in eachblockstoredindex(A)
            bI = A[I]
            A′[Block(1), I] = reshape(bI, 1, size(bI)...)
        end
        return TensorAlgebra.matricize(A′, (1, 2), (3,))
    end
end

function fluxify(A, Aaxes, charge; side::Symbol = :domain)
    side === :domain || side === :codomain || throw(ArgumentError("invalid side $(side)"))

    istrivial(charge) && return TensorAlgebra.unmatricize(A, (Aaxes[1],), (Aaxes[2],))

    if side === :domain
        A′ = TensorAlgebra.unmatricize(A, (Aaxes[1],), (Aaxes[2], to_gradedrange(charge)))
    else
        A′ = TensorAlgebra.unmatricize(A, (to_gradedrange(charge), Aaxes[1]), (Aaxes[2],))
    end

    A″ = similar(A′, Aaxes)
    for I in eachblockstoredindex(A′)
        bI = A′[I]
        I′ = side === :domain ? (Tuple(I)[1], Tuple(I)[2]) : (Tuple(I)[2], Tuple(I)[3])
        A″[I′...] = dropdims(bI; dims = (side === :domain ? 3 : 1))
    end
    return A″
end

function BlockSparseArrays.blockdiagonalize(A::GradedMatrix)
    istrivial(_safe_flux(A)) || throw(ArgumentError("input should have trivial flux"))
    ax1, ax2 = axes(A)
    s1 = sectors(ax1)
    s2 = sectors(ax2)
    @assert allunique(s1) && allunique(s2) "input should have merged axes for sectors"
    allsectors1 = sort!(union(s1, s2))
    allsectors2 = isdual(ax1) == isdual(ax2) ? dual.(allsectors1) : allsectors1

    p1 = indexin(allsectors1, s1)
    ax1′ = gradedrange(
        map(allsectors1, p1) do s, i
            return s => isnothing(i) ? 0 : length(ax1[Block(i)])
        end;
        isdual = isdual(ax1),
    )

    p2 = indexin(allsectors2, s2)
    ax2′ = gradedrange(
        map(allsectors2, p2) do s, i
            return s => isnothing(i) ? 0 : length(ax2[Block(i)])
        end;
        isdual = isdual(ax2),
    )

    Ad = similar(A, ax1′, ax2′)

    p_rows = indexin(s1, allsectors1)
    p_cols = indexin(s2, allsectors2)
    for bI in eachblockstoredindex(A)
        block = A[bI]
        bId = Block(getindex.((p_rows, p_cols), Int.(Tuple(bI))))
        Ad[bId] = block
    end

    invp_rows = Block.(p_rows)
    invp_cols = Block.(p_cols)
    return Ad, (invp_rows, invp_cols)
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(svd_compact!), A::GradedMatrix, alg::BlockDiagonalAlgorithm
    )
    brows = eachblockaxis(axes(A, 1))
    bcols = eachblockaxis(axes(A, 2))

    # Note: this is a small hack that uses the non-symmetric infimum(a, b) ≠ infimum(b, a)
    # where the sector is obtained from the first range, while the second sector is ignored
    # also using the property that zip stops as soon as one of the iterators is exhausted
    s_axes1 = map(splat(infimum), zip(brows, bcols))
    s_axis1 = mortar_axis(s_axes1)
    s_axes2 = map(splat(infimum), zip(bcols, brows))
    s_axis2 = mortar_axis(s_axes2)

    S_axes = (
        isdual(axes(A, 1)) ? dual(s_axis1) : s_axis1,
        isdual(axes(A, 2)) ? dual(s_axis2) : s_axis2,
    )

    BU, BS, BVᴴ = fieldtypes(output_type(svd_compact!, blocktype(A)))
    U = similar(A, BlockType(BU), (axes(A, 1), dual(S_axes[1])))
    S = similar(A, BlockType(BS), (S_axes[1], dual(S_axes[2])))
    Vᴴ = similar(A, BlockType(BVᴴ), (S_axes[2], axes(A, 2)))

    return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(svd_full!), A::GradedMatrix, alg::BlockDiagonalAlgorithm
    )
    BU, BS, BVᴴ = fieldtypes(output_type(svd_full!, blocktype(A)))
    U = similar(A, BlockType(BU), (axes(A, 1), dual(axes(A, 1))))
    S = similar(A, BlockType(BS), axes(A))
    Vᴴ = similar(A, BlockType(BVᴴ), (dual(axes(A, 2)), axes(A, 2)))

    return U, S, Vᴴ
end

for f! in (:svd_compact!, :svd_full!)
    @eval function MatrixAlgebraKit.$f!(
            A::GradedMatrix, USVᴴ, alg::BlockPermutedDiagonalAlgorithm
        )
        MatrixAlgebraKit.check_input($f!, A, USVᴴ, alg)

        charge = _safe_flux(A)
        axA = axes(A)
        A = unfluxify(A, charge; side = :domain)

        Ad, (invrowperm, invcolperm) = BlockSparseArrays.blockdiagonalize(A)
        Ud, S, Vᴴd = $f!(Ad, BlockDiagonalAlgorithm(alg))
        U = BlockSparseArrays.transform_rows(Ud, invrowperm)
        Vᴴ = BlockSparseArrays.transform_cols(Vᴴd, invcolperm)

        Vᴴ = fluxify(Vᴴ, (axes(Vᴴ, 1), axA[2]), charge; side = :domain)
        Vᴴ = unfluxify(Vᴴ, charge; side = :codomain)
        S = fluxify(S, (axes(S, 1), dual(axes(Vᴴ, 1))), charge; side = :domain)

        nonzero_blocks_U = Block.(findall(!isempty, eachblockaxis(axes(U, 2))))
        nonzero_blocks_Vᴴ = Block.(findall(!isempty, eachblockaxis(axes(Vᴴ, 1))))
        return U[:, nonzero_blocks_U],
            S[nonzero_blocks_U, nonzero_blocks_Vᴴ],
            Vᴴ[nonzero_blocks_Vᴴ, :]
    end
end

function MatrixAlgebraKit.svd_trunc!(
        A::GradedMatrix, USVᴴ, alg::TruncatedAlgorithm{<:BlockPermutedDiagonalAlgorithm}
    )
    charge = _safe_flux(A)
    axA = axes(A)
    A = unfluxify(A, charge; side = :domain)

    Ad, (invrowperm, invcolperm) = BlockSparseArrays.blockdiagonalize(A)
    blockalg = BlockDiagonalAlgorithm(alg.alg)
    blockstrategy = BlockDiagonalTruncationStrategy(alg.trunc)
    Ud, S, Vᴴd = svd_trunc!(Ad, TruncatedAlgorithm(blockalg, blockstrategy))
    U = BlockSparseArrays.transform_rows(Ud, invrowperm)
    Vᴴ = BlockSparseArrays.transform_cols(Vᴴd, invcolperm)

    Vᴴ = fluxify(Vᴴ, (axes(Vᴴ, 1), axA[2]), charge; side = :domain)
    Vᴴ = unfluxify(Vᴴ, charge; side = :codomain)
    S = unfluxify(S, dual(charge); side = :domain)

    nonzero_blocks_U = Block.(findall(!isempty, eachblockaxis(axes(U, 2))))
    nonzero_blocks_Vᴴ = Block.(findall(!isempty, eachblockaxis(axes(Vᴴ, 1))))
    return U[:, nonzero_blocks_U],
        S[nonzero_blocks_U, nonzero_blocks_Vᴴ],
        Vᴴ[nonzero_blocks_Vᴴ, :]
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(qr_compact!), A::GradedMatrix, alg::BlockDiagonalAlgorithm
    )
    brows = eachblockaxis(axes(A, 1))
    bcols = eachblockaxis(axes(A, 2))
    # using the property that zip stops as soon as one of the iterators is exhausted
    r_axes = map(splat(infimum), zip(brows, bcols))
    r_axis = mortar_axis(r_axes)

    BQ, BR = fieldtypes(output_type(qr_compact!, blocktype(A)))
    Q = similar(A, BlockType(BQ), (axes(A, 1), dual(r_axis)))
    R = similar(A, BlockType(BR), (r_axis, axes(A, 2)))

    return Q, R
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(qr_full!), A::GradedMatrix, alg::BlockDiagonalAlgorithm
    )
    BQ, BR = fieldtypes(output_type(qr_full!, blocktype(A)))
    Q = similar(A, BlockType(BQ), (axes(A, 1), dual(axes(A, 1))))
    R = similar(A, BlockType(BR), (axes(A, 1), axes(A, 2)))
    return Q, R
end

for f! in (:qr_compact!, :qr_full!, :left_polar!)
    @eval function MatrixAlgebraKit.$f!(
            A::GradedMatrix, QR, alg::BlockPermutedDiagonalAlgorithm
        )
        MatrixAlgebraKit.check_input($f!, A, QR, alg)

        axA = axes(A)
        charge = _safe_flux(A)
        A = unfluxify(A, charge; side = :domain)

        Ad, (invrowperm, invcolperm) = BlockSparseArrays.blockdiagonalize(A)
        Qd, Rd = $f!(Ad, BlockDiagonalAlgorithm(alg))
        Q = BlockSparseArrays.transform_rows(Qd, invrowperm)
        R = BlockSparseArrays.transform_cols(Rd, invcolperm)

        R = fluxify(R, (axes(R, 1), axA[2]), charge; side = :domain)

        nonzero_blocks = Block.(findall(!isempty, eachblockaxis(axes(R, 1))))
        return Q[:, nonzero_blocks], R[nonzero_blocks, :]
    end
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(lq_compact!), A::GradedMatrix, alg::BlockDiagonalAlgorithm
    )
    brows = eachblockaxis(axes(A, 1))
    bcols = eachblockaxis(axes(A, 2))
    # using the property that zip stops as soon as one of the iterators is exhausted
    l_axes = map(splat(infimum), zip(bcols, brows))
    l_axis = mortar_axis(l_axes)

    BL, BQ = fieldtypes(output_type(lq_compact!, blocktype(A)))
    L = similar(A, BlockType(BL), (axes(A, 1), l_axis))
    Q = similar(A, BlockType(BQ), (dual(l_axis), axes(A, 2)))

    return L, Q
end

function MatrixAlgebraKit.initialize_output(
        ::typeof(lq_full!), A::GradedMatrix, alg::BlockDiagonalAlgorithm
    )
    BL, BQ = fieldtypes(output_type(lq_full!, blocktype(A)))
    L = similar(A, BlockType(BL), (axes(A, 1), axes(A, 2)))
    Q = similar(A, BlockType(BQ), (dual(axes(A, 2)), axes(A, 2)))
    return L, Q
end

for f! in (:lq_compact!, :lq_full!, :right_polar!)
    @eval function MatrixAlgebraKit.$f!(
            A::GradedMatrix, LQ, alg::BlockPermutedDiagonalAlgorithm
        )
        MatrixAlgebraKit.check_input($f!, A, LQ, alg)

        charge = _safe_flux(A)
        axA = axes(A)
        A = unfluxify(A, charge; side = :codomain)

        Ad, (invrowperm, invcolperm) = BlockSparseArrays.blockdiagonalize(A)
        Ld, Qd = $f!(Ad, BlockDiagonalAlgorithm(alg))
        L = BlockSparseArrays.transform_rows(Ld, invrowperm)
        Q = BlockSparseArrays.transform_cols(Qd, invcolperm)

        L = fluxify(L, (axA[1], axes(L, 2)), charge; side = :codomain)

        # avoid length zero blockaxis
        nonzero_blocks = Block.(findall(!isempty, eachblockaxis(axes(L, 2))))
        return L[:, nonzero_blocks], Q[nonzero_blocks, :]
    end
end

# Fix for polar decompositions not following standard codepath
for f! in (:left_polar!, :right_polar!)
    @eval function MatrixAlgebraKit.$f!(A::GradedMatrix, alg::PolarViaSVD)
        return $f!(A, MatrixAlgebraKit.initialize_output($f!, A, alg), alg)
    end
end

function MatrixAlgebraKit.left_polar!(
        A::GradedMatrix, WP, alg::PolarViaSVD{<:BlockPermutedDiagonalAlgorithm}
    )
    MatrixAlgebraKit.check_input(left_polar!, A, WP, alg)

    charge = _safe_flux(A)
    axA = axes(A)
    A = unfluxify(A, charge; side = :domain)

    Ad, (invrowperm, invcolperm) = BlockSparseArrays.blockdiagonalize(A)
    Ud, S, Vᴴd = svd_compact!(Ad, BlockDiagonalAlgorithm(alg.svdalg))
    U = BlockSparseArrays.transform_rows(Ud, invrowperm)
    Vᴴ = BlockSparseArrays.transform_cols(Vᴴd, invcolperm)

    W = U * Vᴴ
    P = Vᴴ' * S * Vᴴ
    P = fluxify(P, (dual(axes(W, 2)), axA[2]), charge; side = :domain)
    return W, P
end

function MatrixAlgebraKit.right_polar!(A::GradedMatrix, PWᴴ, alg::PolarViaSVD)
    MatrixAlgebraKit.check_input(right_polar!, A, PWᴴ, alg)

    charge = _safe_flux(A)
    axA = axes(A)
    A = unfluxify(A, charge; side = :codomain)

    Ad, (invrowperm, invcolperm) = BlockSparseArrays.blockdiagonalize(A)
    Ud, S, Vᴴd = svd_compact!(Ad, BlockDiagonalAlgorithm(alg.svdalg))
    U = BlockSparseArrays.transform_rows(Ud, invrowperm)
    Vᴴ = BlockSparseArrays.transform_cols(Vᴴd, invcolperm)

    Wᴴ = U * Vᴴ
    P = U * S * U'
    P = fluxify(P, (axA[1], dual(axes(Wᴴ, 1))), charge; side = :domain)
    return P, Wᴴ
end
