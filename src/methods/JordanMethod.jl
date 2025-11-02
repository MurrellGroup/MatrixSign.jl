@doc raw"""
    JordanMethod <: QuinticNewtonSchulzMethod

Uses a 5-step quintic Newton-Schulz iteration to compute a coarse
approximation of the sign of a matrix, with singular values roughly
ending up in the interval ``[0.7, 1.2]``.

Reference: [Muon: An optimizer for hidden layers in neural
networks](https://kellerjordan.github.io/posts/muon/)
"""
abstract type JordanMethod <: QuinticNewtonSchulzMethod end

const COEFFICIENTS_JORDAN = (3.4445, -4.7750, 2.0315)

function msign(X::AbstractArray{T}, ::Type{JordanMethod}) where T
    return msign(
        X, QuinticNewtonSchulzMethod;
        coefficients = ntuple(Returns(T.(COEFFICIENTS_JORDAN)), Val(5))
    )
end

function msign!!(X::AbstractArray{T}, ::Type{JordanMethod}) where T
    return msign!!(
        X, QuinticNewtonSchulzMethod;
        coefficients = ntuple(Returns(T.(COEFFICIENTS_JORDAN)), Val(5))
    )
end
