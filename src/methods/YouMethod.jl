@doc raw"""
    YouMethod <: QuinticNewtonSchulzMethod

A 6-step quintic Newton-Schulz iteration with different coefficients
tuned for each step, resulting in higher accuracy.

Reference: [Squeezing 1-2% Efficiency Gains Out of Muon by Optimizing
the Newton-Schulz Coefficients](https://leloykun.github.io/ponder/muon-opt-coeffs/)
"""
abstract type YouMethod <: QuinticNewtonSchulzMethod end

const COEFFICIENTS_YOU = (
    (3955/1024, -8306/1024, 5008/1024),
    (3735/1024, -6681/1024, 3463/1024),
    (4019/1024, -6385/1024, 2906/1024),
    (2677/1024, -3029/1024, 1162/1024),
    (2172/1024, -1833/1024,  682/1024),
    (3799/1024, -6499/1024, 3211/1024),
)

function msign(X::AbstractArray{T}, ::Type{YouMethod}) where T
    return msign(
        X, QuinticNewtonSchulzMethod;
        coefficients = map(x -> T.(x), COEFFICIENTS_YOU)
    )
end

function msign!!(X::AbstractArray{T}, ::Type{YouMethod}) where T
    return msign!!(
        X, QuinticNewtonSchulzMethod;
        coefficients = map(x -> T.(x), COEFFICIENTS_YOU)
    )
end
