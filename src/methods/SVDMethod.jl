@doc raw"""
    SVDMethod <: MatrixSignMethod

Uses a costly singular value decomposition to compute the sign of a matrix.

Let ``\bold{M} = U\Sigma V^T`` be the singular value decomposition of ``\bold{M}``.
Then the sign, or more generally polar factor, of ``\bold{M}`` is given by ``UV^T``.
"""
abstract type SVDMethod <: MatrixSignMethod end

"""
    msign(X::AbstractMatrix, ::Type{SVDMethod})

Return the sign of `X` using the SVD method.
This is the most accurate method for computing the matrix sign,
albeit while typically being an order of magnitude slower than
e.g. [`PolarExpress`](@ref).
"""
function msign(X::AbstractMatrix, ::Type{SVDMethod})
    (; U, Vt) = svd(X)
    return U * Vt
end

function msign!(X::AbstractMatrix, ::Type{SVDMethod})
    (; U, Vt) = svd(X)
    return @mul! X = U * Vt
end
