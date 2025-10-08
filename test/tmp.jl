using GradedArrays
using GradedArrays: flux, GradedMatrix, trivial
using BlockArrays, BlockSparseArrays
using BlockSparseArrays: blockdiagonalize, transform_cols, transform_rows, isblockdiagonal
using MatrixAlgebraKit

r1 = gradedrange([U1(0) => 1, U1(1) => 1])
a = zeros(r1, dual(r1))
a[Block(1, 2)] = randn(eltype(a), blocksizes(a)[1, 2])
# a[Block(2, 2)] = randn(eltype(a), blocksizes(a)[2, 2])

a
Q, R = qr_compact(a)
L, Q = lq_compact(a)

b = zeros(r1, r1)
b[Block(1, 1)] = randn(eltype(b), blocksizes(b)[2, 2])
flux(b)
b

r1

s1 = sectors(r1)
s2 = sectors(r1)

@assert allunique(s1) && allunique(s2) "TBA"

allsectors = sort!(union(s1, s2))

ax1 = r1
ax2 = dual(r1)

p1 = indexin(allsectors, s1)
ax1′ = gradedrange(
  map(allsectors, p1) do s, i
    return s => isnothing(i) ? 0 : length(ax1[Block(i)].full_range)
  end,
)

p2 = indexin(allsectors, s2)
ax2′ = gradedrange(
  map(allsectors, p2) do s, i
    return s => isnothing(i) ? 0 : length(ax2[Block(i)].full_range)
  end;
  isdual=true,
)

r1 = gradedrange([U1(x) => 1 for x in -1:2])
r2 = gradedrange([U1(x) => 1 for x in [1, -1, 0]]; isdual=true)
a = zeros(r1, r2)

for I in eachindex(IndexCartesian(), blocks(a))
  if flux(a, Block(I[1]), Block(I[2])) == U1(0)
    a[Block(I[1]), Block(I[2])] = randn(eltype(a), blocksizes(a)[I])
  end
end
flux(a)
GradedArrays.checkflux(a, U1(0))
a
ad, (invp_rows, invp_cols) = blockdiagonalize(a)

BlockSparseArrays.transform_cols(BlockSparseArrays.transform_rows(ad, invp_rows), invp_cols)
a

a[Block(2, 2)] = randn(eltype(a), blocksizes(a)[2, 2])

r1 = gradedrange([U1(1) => 1])
r2 = gradedrange([U1(-1) => 1])
r3 = gradedrange([U1(1) => 1]; isdual=true)
r4 = gradedrange([U1(-1) => 1]; isdual=true)

a1 = zeros(r1, r1)
a1[Block(1, 1)] = rand(1, 1)
flux(a1)

a2 = zeros(r1, r2)
a2[Block(1, 1)] = rand(1, 1)
flux(a2)

a3 = zeros(r1, r3)
a3[Block(1, 1)] = rand(1, 1)
flux(a3)

a4 = zeros(r1, r4)
a4[Block(1, 1)] = rand(1, 1)
flux(a4)

a2d, = blockdiagonalize(a2)
flux(a2d)
eachblockstoredindex(a2d)
u, s, v = svd_compact(a2d)

a3d, = blockdiagonalize(a3)
flux(a3d)
eachblockstoredindex(a3d)
axes(a3d)
axes(a3)
u, s, v = svd_compact(a3d)

a5 = zeros(r3, r1)
a5[Block(1, 1)] = rand(1, 1)
flux(a5)
a5d, = blockdiagonalize(a5)
flux(a5d)
eachblockstoredindex(a5d)
axes(a5d)
axes(a5)
u, s, v = svd_compact(a5d)

b = zeros(gradedrange([U1(1) => 1]), dual(gradedrange([U1(0) => 1])))
b[Block(1, 1)] = rand(1, 1)
flux(b)

bd, = blockdiagonalize(b)
flux(bd)
eachblockstoredindex(bd)
axes(bd)

eachblockstoredindex(b)
u, s, v = svd_compact(b)

flux(u)
flux(s)
flux(v)

c = zeros(
  gradedrange([U1(0) => 1, U1(1) => 1]), dual(gradedrange([U1(1) => 1, U1(2) => 1]))
)
c[1, 1] = 1
flux(c)
u, s, v = svd_compact(c)
flux(u)
flux(s)
flux(v)

using MatrixAlgebraKit: left_polar

q, r = left_polar(c)
flux(q)
flux(r)

elt = Float64
r1 = gradedrange([U1(0) => 2, U1(1) => 3])
r2 = gradedrange([U1(0) => 3, U1(1) => 4])
a = zeros(elt, r1, dual(r2))
a[Block(1, 2)] = randn(elt, blocksizes(a)[1, 2])

u, s, v = svd_compact(a)

axes(u)
u
axes(v)
v
axes(v)
flux(u), flux(v), flux(u * v)
