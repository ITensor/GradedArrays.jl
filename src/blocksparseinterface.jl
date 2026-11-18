# Block-sparse interface functions owned by GradedArrays, so it does not depend on
# BlockSparseArrays. `isblockdiag` follows `LinearAlgebra.isdiag` rather than BlockSparseArrays'
# `isblockdiagonal`.

function eachblockstoredindex end
function eachblockaxis end
function mortar_axis end
function blocktype end
function isblockdiag end

# Not `storedlength(blocks(a))`, which would materialize a view of every block.
blockstoredlength(a) = length(eachblockstoredindex(a))
