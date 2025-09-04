using BlockArrays: blocks
using BlockSparseArrays:
  BlockSparseArrays,
  eachblockaxis,
  mortar_axis,
  infimum,
  output_type,
  BlockType,
  BlockDiagonalAlgorithm
using LinearAlgebra: Diagonal
using MatrixAlgebraKit:
  MatrixAlgebraKit,
  lq_compact!,
  lq_full!,
  qr_compact!,
  qr_full!,
  svd_compact!,
  svd_full!,
  svd_trunc!

function BlockSparseArrays.blockdiagonalize(A::GradedMatrix)
  ax1, ax2 = axes(A)
  s1 = sectors(ax1)
  s2 = sectors(ax2)
  @assert allunique(s1) && allunique(s2) "TBA"
  allsectors1 = sort!(union(s1, fusion_rule.(s2, Ref(flux(A)))))
  allsectors2 = fusion_rule.(allsectors1, Ref(dual(flux(A))))
  allsectors2 = isdual(ax1) == isdual(ax2) ? dual.(allsectors2) : allsectors2

  p1 = indexin(allsectors1, s1)
  ax1′ = gradedrange(
    map(allsectors1, p1) do s, i
      return s => isnothing(i) ? 0 : length(ax1[Block(i)].full_range)
    end;
    isdual=isdual(ax1),
  )

  p2 = indexin(allsectors2, s2)
  ax2′ = gradedrange(
    map(allsectors2, p2) do s, i
      return s => isnothing(i) ? 0 : length(ax2[Block(i)].full_range)
    end;
    isdual=isdual(ax2),
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
