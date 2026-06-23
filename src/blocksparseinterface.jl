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

# `blockstoredlength` is only called (never extended on graded types), so vendor the generic
# definition that BlockSparseArrays provides, built on the SparseArraysBase block storage.
blockstoredlength(a) = SparseArraysBase.storedlength(blocks(a))
