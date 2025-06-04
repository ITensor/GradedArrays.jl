using BlockArrays: blocks
using BlockSparseArrays:
  BlockSparseArrays,
  BlockSparseMatrix,
  BlockPermutedDiagonalAlgorithm,
  BlockPermutedDiagonalTruncationStrategy,
  diagview,
  eachblockaxis,
  mortar_axis
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

function BlockSparseArrays.similar_output(
  ::typeof(svd_compact!), A::GradedMatrix, S_axes, alg::BlockPermutedDiagonalAlgorithm
)
  u_axis, v_axis = S_axes
  U = similar(A, axes(A, 1), dual(u_axis))
  T = real(eltype(A))
  S = BlockSparseMatrix{T,Diagonal{T,Vector{T}}}(undef, (u_axis, v_axis))
  Vt = similar(A, dual(v_axis), axes(A, 2))
  return U, S, Vt
end

function BlockSparseArrays.similar_output(
  ::typeof(svd_full!), A::GradedMatrix, S_axes, alg::BlockPermutedDiagonalAlgorithm
)
  u_axis, s_axis = S_axes
  U = similar(A, axes(A, 1), dual(u_axis))
  T = real(eltype(A))
  S = similar(A, T, S_axes)
  Vt = similar(A, dual(S_axes[2]), axes(A, 2))
  return U, S, Vt
end

const TGradedUSVᴴ = Tuple{<:GradedMatrix,<:GradedMatrix,<:GradedMatrix}

function BlockSparseArrays.similar_output(
  ::typeof(qr_compact!), A::GradedMatrix, R_axis, alg::BlockPermutedDiagonalAlgorithm
)
  Q = similar(A, axes(A, 1), dual(R_axis))
  R = similar(A, R_axis, axes(A, 2))
  return Q, R
end

function BlockSparseArrays.similar_output(
  ::typeof(qr_full!), A::GradedMatrix, R_axis, alg::BlockPermutedDiagonalAlgorithm
)
  Q = similar(A, axes(A, 1), dual(R_axis))
  R = similar(A, R_axis, axes(A, 2))
  return Q, R
end

function BlockSparseArrays.similar_output(
  ::typeof(lq_compact!), A::GradedMatrix, L_axis, alg::BlockPermutedDiagonalAlgorithm
)
  L = similar(A, axes(A, 1), L_axis)
  Q = similar(A, dual(L_axis), axes(A, 2))
  return L, Q
end

function BlockSparseArrays.similar_output(
  ::typeof(lq_full!), A::GradedMatrix, L_axis, alg::BlockPermutedDiagonalAlgorithm
)
  L = similar(A, axes(A, 1), L_axis)
  Q = similar(A, dual(L_axis), axes(A, 2))
  return L, Q
end
