@doc raw"""
    JordanMethod

Uses a 5-step quintic Newton-Schulz iteration to compute a coarse
approximation of the sign of a matrix, with singular values roughly
ending up in the interval ``[0.7, 1.2]``.

Reference: [Muon: An optimizer for hidden layers in neural
networks](https://kellerjordan.github.io/posts/muon/)
"""
abstract type JordanMethod <: QuinticNewtonSchulzMethod end

const COEFFICIENTS_JORDAN = (3.4445f0, -4.7750f0, 2.0315f0)

@constprop :aggressive function msign(
    X::AbstractArray, ::Type{JordanMethod};
    steps=5, kws...
)
    coefficients = ntuple(Returns(COEFFICIENTS_JORDAN), Val(steps))
    return newtonschulz5(normalize(X), coefficients; kws...)
end

@constprop :aggressive function msign!(
    X::AbstractArray, ::Type{JordanMethod};
    steps=5, kws...
)
    coefficients = ntuple(Returns(COEFFICIENTS_JORDAN), Val(steps))
    return newtonschulz5!(normalize!(X), coefficients; kws...)
end

function msign!(X::AbstractArray{Float16}, ::Type{JordanMethod}; kws...)
    XB = BFloat16.(X)
    msign!(XB, JordanMethod; kws...)
    X .= XB
    return X
end

function msign(X::AbstractArray{Float16}, ::Type{JordanMethod}; kws...)
    XB = BFloat16.(X)
    msign!(XB, JordanMethod; kws...)
    return Float16.(XB)
end
