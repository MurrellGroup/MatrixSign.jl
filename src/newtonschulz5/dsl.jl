ᵀ(X::Union{Number,AbstractArray{<:Any,0}}) = X
ᵀ(X::AbstractVecOrMat) = transpose(X)
ᵀ(X::AbstractArray{<:Any,3}) = batched_transpose(X)
Base.:*(X, ::typeof(ᵀ)) = ᵀ(X)

#=
ᴴ(X::Union{Number,AbstractArray{<:Any,0},AbstractVecOrMat}) = adjoint(X)
ᴴ(X::AbstractArray{<:Any,3}) = batched_adjoint(X)
Base.:*(X, ::typeof(ᴴ)) = ᴴ(X)

⁻¹(X::Union{Number,AbstractMatrix}) = inv(X)
Base.:*(X, ::typeof(⁻¹)) = ⁻¹(X)
=#

function _mul!(
    C::AbstractMatrix,
    A::AbstractVecOrMat,
    B::AbstractVecOrMat,
    args...
)
    mul!(C, A, B, args...)
    return C
end

function _mul!(
    C::AbstractArray,
    A::AbstractArray,
    B::AbstractArray,
    args...
)
    batched_mul!(C, A, B, args...)
    return C
end

macro mul!(ex)
    return :($_mul!($(esc.(_get_mul_args(ex))...)))
end

function _get_mul_args(ex)
    if @capture(ex, left_ + right_)
        b, C = _get_left_args(left)
        a, A, B = _get_right_args(right)
        C, A, B, a, b
    elseif @capture(ex, C_ = A_ * B_) || @capture(ex, C_ = (A_)^2)
        C, A, B
    else
        error("Expected left and right sides to be separated by + or =")
    end
end

function _get_left_args(ex)
    if @capture(ex, b_ * C_) || @capture(ex, b_(C_))
        b, C
    elseif @capture(ex, C_)
        1, C
    else
        error("Expected left side to be of the form b * C or C")
    end
end

function _get_right_args(ex)
    if @capture(ex, a_ * A_^2) || @capture(ex, a_(A_^2))
        a, A, A
    elseif @capture(ex, A_^2)
        1, A, A
    elseif @capture(ex, a_ * A_ * B_) || @capture(ex, a_(A_ * B_)) || @capture(ex, a_Number * (A_ * B_))
        a, A, B
    elseif @capture(ex, A_ * B_)
        1, A, B
    else
        error("Expected right side to be of the form a * A * B or A * B")
    end
end
