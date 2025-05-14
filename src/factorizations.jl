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
  ::typeof(svd_compact!),
  A::GradedMatrix,
  s_axis::AbstractUnitRange,
  alg::BlockPermutedDiagonalAlgorithm,
)
  u_axis = s_axis
  flx = flux(A)
  axs = eachblockaxis(s_axis)
  # TODO: Use `gradedrange` constructor.
  v_axis = mortar_axis(
    map(axs) do ax
      return sectorrange(dual(sector(ax)) ⊗ flx, ungrade(ax))
    end,
  )
  U = similar(A, axes(A, 1), dual(u_axis))
  T = real(eltype(A))
  S = BlockSparseMatrix{T,Diagonal{T,Vector{T}}}(undef, (u_axis, dual(v_axis)))
  Vt = similar(A, v_axis, axes(A, 2))
  return U, S, Vt
end

function BlockSparseArrays.similar_output(
  ::typeof(svd_full!),
  A::GradedMatrix,
  s_axis::AbstractUnitRange,
  alg::BlockPermutedDiagonalAlgorithm,
)
  U = similar(A, axes(A, 1), dual(s_axis))
  T = real(eltype(A))
  S = similar(A, T, (s_axis, axes(A, 2)))
  Vt = similar(A, dual(axes(A, 2)), axes(A, 2))
  return U, S, Vt
end

const TGradedUSVᴴ = Tuple{<:GradedMatrix,<:GradedMatrix,<:GradedMatrix}

function BlockSparseArrays.similar_truncate(
  ::typeof(svd_trunc!),
  (U, S, Vᴴ)::TGradedUSVᴴ,
  strategy::BlockPermutedDiagonalTruncationStrategy,
  indexmask=MatrixAlgebraKit.findtruncated(diagview(S), strategy),
)
  ax = axes(S, 1)
  counter = Base.Fix1(count, Base.Fix1(getindex, indexmask))
  s_lengths = filter!(>(0), map(counter, blocks(ax)))
  s_axis = gradedrange(sectors(ax) .=> s_lengths)
  u_axis = s_axis
  flx = flux(S)
  axs = eachblockaxis(s_axis)
  # TODO: Use `gradedrange` constructor.
  v_axis = mortar_axis(
    map(axs) do ax
      return sectorrange(dual(sector(ax)) ⊗ flx, ungrade(ax))
    end,
  )
  Ũ = similar(U, axes(U, 1), dual(u_axis))
  S̃ = similar(S, u_axis, dual(v_axis))
  Ṽᴴ = similar(Vᴴ, v_axis, axes(Vᴴ, 2))
  return Ũ, S̃, Ṽᴴ
end
