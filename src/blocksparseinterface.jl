# Block-sparse interface functions owned by GradedArrays.
#
# GradedArrays implements a block-sparse interface on its own graded array and axis types.
# These names are duplicated with BlockSparseArrays by design: GradedArrays owns them here so
# it does not depend on BlockSparseArrays. They are internal (not exported); downstream reaches
# them by qualified import, e.g. `using GradedArrays: eachblockstoredindex`.

function eachblockstoredindex end
function eachblockaxis end
function mortar_axis end
function blocktype end
function isblockdiagonal end

# The number of stored (symmetry-allowed) blocks. Counted from the stored block indices directly
# rather than via `storedlength(blocks(a))`, whose generic `length(storedvalues(...))` would
# materialize a view of every block (disabled by the block-indexing guard on a `GradedArray`).
blockstoredlength(a) = length(eachblockstoredindex(a))
