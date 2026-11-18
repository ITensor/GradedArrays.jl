# For iterative solvers such as `KrylovKit.linsolve`. The generic `AbstractArray` methods broadcast
# a function over the elements, which the graded broadcast styles reject.
for AT in (:GradedArray, :AbstractFusedGradedArray, :AbstractSectorArray)
    @eval begin
        function VI.zerovector(a::$AT, ::Type{S}) where {S <: Number}
            return VI.zerovector!(similar(a, S))
        end
        VI.zerovector!(a::$AT) = zero!(a)
        VI.zerovector!!(a::$AT) = VI.zerovector!(a)

        # `Base.promote_op` since `VectorInterface.promote_scale`/`promote_add` are internal.
        function VI.scale(a::$AT, α::Number)
            T = Base.promote_op(VI.scale, VI.scalartype(a), typeof(α))
            return VI.scale!(similar(a, T), a, α)
        end
        VI.scale!(a::$AT, α::Number) = TensorAlgebra.scale!(a, α)
        function VI.scale!(b::$AT, a::$AT, α::Number)
            return TensorAlgebra.scaleadd!(b, a, α, false)
        end
        function VI.scale!!(a::$AT, α::Number)
            T = Base.promote_op(VI.scale, VI.scalartype(a), typeof(α))
            T <: VI.scalartype(a) || return VI.scale(a, α)
            return VI.scale!(a, α)
        end
        function VI.scale!!(b::$AT, a::$AT, α::Number)
            T = Base.promote_op(VI.scale, VI.scalartype(a), typeof(α))
            T <: VI.scalartype(b) || return VI.scale(a, α)
            return VI.scale!(b, a, α)
        end

        function VI.add(a::$AT, b::$AT, α::Number, β::Number)
            T = Base.promote_op(
                VI.add,
                VI.scalartype(a),
                VI.scalartype(b),
                typeof(α),
                typeof(β)
            )
            return VI.add!(VI.scale!(similar(a, T), a, β), b, α, true)
        end
        function VI.add!(a::$AT, b::$AT, α::Number, β::Number)
            return TensorAlgebra.scaleadd!(a, b, α, β)
        end
        function VI.add!!(a::$AT, b::$AT, α::Number, β::Number)
            T = Base.promote_op(
                VI.add,
                VI.scalartype(a),
                VI.scalartype(b),
                typeof(α),
                typeof(β)
            )
            T <: VI.scalartype(a) || return VI.add(a, b, α, β)
            return VI.add!(a, b, α, β)
        end

        VI.inner(a::$AT, b::$AT) = dot(a, b)
    end
end
