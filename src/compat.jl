# The two-argument `allequal(f, itr)` only entered `Base` in Julia 1.11. GradedArrays supports the
# 1.10 LTS, so provide our own version rather than pull in Compat. Drop this once the supported Julia
# floor is 1.11+ and call `allequal` directly.
allequal_compat(itr) = allequal(itr)
if VERSION < v"1.11.0-DEV.1562"
    allequal_compat(f, itr) = allequal(f(x) for x in itr)
else
    allequal_compat(f, itr) = allequal(f, itr)
end
