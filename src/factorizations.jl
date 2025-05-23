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
using MatrixAlgebraKit: MatrixAlgebraKit, svd_compact!, svd_full!, svd_trunc!

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

function BlockSparseArrays.similar_truncate(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::TGradedUSVᴴ,
  strategy::BlockPermutedDiagonalTruncationStrategy,
  indexmask=MatrixAlgebraKit.findtruncated(diagview(S), strategy),
)
  u_axis, v_axis = axes(S)
  counter = Base.Fix1(count, Base.Fix1(getindex, indexmask))
  s_lengths = map(counter, blocks(u_axis))
  u_sectors = sectors(u_axis) .=> s_lengths
  v_sectors = sectors(v_axis) .=> s_lengths
  u_sectors_filtered = filter(>(0) ∘ last, u_sectors)
  v_sectors_filtered = filter(>(0) ∘ last, v_sectors)
  u_axis′ = gradedrange(u_sectors_filtered)
  u_axis = isdual(u_axis) ? dual(u_axis′) : u_axis′
  v_axis′ = gradedrange(v_sectors_filtered)
  v_axis = isdual(v_axis) ? dual(v_axis′) : v_axis′
  Ũ = similar(U, axes(U, 1), dual(u_axis))
  S̃ = similar(S, u_axis, v_axis)
  Ṽᴴ = similar(Vᴴ, dual(v_axis), axes(Vᴴ, 2))
  return Ũ, S̃, Ṽᴴ
end
