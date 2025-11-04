"""
    MatrixSignMethod

Abstract type for matrix sign methods.

Subtypes of this type are expected to implement the following methods:

- `msign(X::AbstractArray, ::Type{M}; kws...)`
- `msign!(X::AbstractArray, ::Type{M}; kws...)`
"""
abstract type MatrixSignMethod end

include("methods/SVDMethod.jl")

@doc raw"""
    QuinticNewtonSchulzMethod <: MatrixSignMethod

Takes advantage of the following equivalence to change the singular values of a matrix,
where ``p`` is an odd quintic polynomial of the form ``p(x) = aX + bX(X^TX) + cX(X^TX)^2``:

```math
\begin{aligned}
p(U\Sigma V^T) &= a(U\Sigma V^T) + b(U\Sigma V^T)(U\Sigma V^T)^T(U\Sigma V^T)
                                 + c(U\Sigma V^T)((U\Sigma V^T)^T(U\Sigma V^T))^2 \\
               &= a(U\Sigma V^T) + b(U\Sigma V^T)(V\Sigma U^T)(U\Sigma V^T)
                                 + c(U\Sigma V^T)((V\Sigma U^T)(U\Sigma V^T))^2 \\
               &= aU\Sigma V^T + bU\Sigma^3 V^T + cU\Sigma^5 V^T \\
               &= U(a\Sigma + b\Sigma^3 + c\Sigma^5)V^T \\
               &= Up(\Sigma)V^T
\end{aligned}
```

And given ``\Sigma`` is a diagonal matrix of singular values,
applying the polynomial to the matrix is equivalent to applying it to each scalar singular value,
so we can approximate the sign of a matrix by approximating the scalar sign function.

Depending on the coefficients and number of steps, we can trade off accuracy for speed.
See [`JordanMethod`](@ref), [`YouMethod`](@ref), and [`PolarExpress`](@ref).
"""
abstract type QuinticNewtonSchulzMethod <: MatrixSignMethod end

norm2(X::AbstractArray) = sum(abs2, X; dims=(1,2))

function normalize(X::AbstractArray{T}; safety_factor=1, eps=1f-7) where T
    return @. X / (√($norm2(X)) * $T(safety_factor) + $T(eps))
end

function normalize!(X::AbstractArray{T}; safety_factor=1, eps=1f-7) where T
    @. X = X / (√($norm2(X)) * $T(safety_factor) + $T(eps))
    return X
end

include("methods/JordanMethod.jl")
include("methods/PolarExpress.jl")

msign(X::AbstractArray; kws...) = msign(X, PolarExpress; kws...)
msign!(X::AbstractArray; kws...) = msign!(X, PolarExpress; kws...)
