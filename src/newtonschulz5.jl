is_tall(X) = size(X, 1) > size(X, 2)
similar_square(X::AbstractMatrix, n = minimum(size(X))) = similar(X, n, n)

# equivalent to `aX + bXX'X + cXX'XX'X`
# expects Y == X && !(Y === X)
function newtonschulz5_step!(
    (; A, B, Y),
    X, (; a, b, c)
)
    if is_tall(X)
        XᵀX = mul!(A, X', X)
        mul!(B .= XᵀX, XᵀX, XᵀX, c, b) # B ← bXᵀX + cXᵀXXᵀX
        mul!(Y, X, B, 1, a)            # Y ← aX + XB = aX + bXXᵀX + cXXᵀXXᵀX
    else
        XXᵀ = mul!(A, X, X')
        mul!(B .= XXᵀ, XXᵀ, XXᵀ, c, b) # B ← bXXᵀ + cXXᵀXXᵀ
        mul!(Y, B, X, 1, a)            # Y ← aX + BX = aX + bXXᵀX + cXXᵀXXᵀX
    end
    return Y
end

function newtonschulz5!(X::AbstractMatrix, coefficients)
    A, B, Y = similar_square(X), similar_square(X), similar(X)
    for (a, b, c) in coefficients
        copyto!(Y, X)
        newtonschulz5_step!((; A, B, Y=X), Y, (; a, b, c))
    end
    return X
end

newtonschulz5(X::AbstractMatrix, coefficients) = newtonschulz5!(copy(X), coefficients)
newtonschulz5(X::AbstractArray, coef) = reshape(newtonschulz5(reshape(X, size(X, 1), :), coef), size(X))
newtonschulz5!(X::AbstractArray, coef) = reshape(newtonschulz5!(reshape(X, size(X, 1), :), coef), size(X))
newtonschulz5!!(X::AbstractArray, coef) = maywrite(X) ? newtonschulz5!(X, coef) : newtonschulz5(X, coef)

function vjp_newtonschulz5_step!(
    (; A, B, C), X̄::AbstractMatrix,
    Ȳ::AbstractMatrix, X::AbstractMatrix, (; a, b, c)
)
    if is_tall(X)
        XᵀX = mul!(A, X', X)
        mul!(B .= XᵀX, XᵀX, XᵀX, c, b)  # B ← bXᵀX + cXᵀXXᵀX
        mul!(X̄ .= Ȳ, Ȳ, B, 1, a)        # X̄ ← aȲ + bȲXᵀX + cȲXᵀXXᵀX
        XᵀȲ = mul!(B, X', Ȳ)
        mul!(C .= XᵀȲ, XᵀȲ, XᵀX, c, b)  # C ← bXᵀȲ + cXᵀȲXᵀX
        mul!(C,        XᵀX, XᵀȲ, c, 1)  # C ← C + cXᵀXXᵀȲ = bXᵀȲ + cXᵀȲXᵀX + cXᵀXXᵀȲ
        mul!(X̄, X, B .= C .+ C', 1, 1)  # X̄ ← X̄ + X(C + Cᵀ)
                                        #   = X̄ + X(bXᵀȲ + cXᵀȲXᵀX + cXᵀXXᵀȲ + bȲᵀX + cXᵀXȲᵀX + cȲᵀXXᵀX)
                                        #   = aȲ + bȲXᵀX + cȲXᵀXXᵀX + bXXᵀȲ + cXXᵀȲXᵀX + cXXᵀXXᵀȲ + bXȲᵀX + cXXᵀXȲᵀX + cXȲᵀXXᵀX
    else                                #   = aȲ + b(ȲXᵀX + XȲᵀX + XXᵀȲ) + c(ȲXᵀXXᵀX + XȲᵀXXᵀX + XXᵀȲXᵀX + XXᵀXȲᵀX + XXᵀXXᵀȲ)
        XXᵀ = mul!(A, X, X')
        mul!(B .= XXᵀ, XXᵀ, XXᵀ, c, b)  # B ← bXXᵀ + cXXᵀXXᵀ
        mul!(X̄ .= Ȳ, B, Ȳ, 1, a)        # X̄ ← aȲ + bXXᵀȲ + cXXᵀXXᵀȲ
        ȲXᵀ = mul!(B, Ȳ, X')
        mul!(C .= ȲXᵀ, ȲXᵀ, XXᵀ, c, b)  # C ← bȲXᵀ + cȲXᵀXXᵀ
        mul!(C,        XXᵀ, ȲXᵀ, c, 1)  # C ← C + cXXᵀȲXᵀ = bȲXᵀ + cȲXᵀXXᵀ + cXXᵀȲXᵀ
        mul!(X̄, B .= C .+ C', X, 1, 1)  # X̄ ← X̄ + (C + Cᵀ)X
                                        #   = X̄ + (bȲXᵀ + cȲXᵀXXᵀ + cXXᵀȲXᵀ + bXȲᵀ + cXXᵀXȲᵀ + cXȲᵀXXᵀ)X
                                        #   = aȲ + bXXᵀȲ + cXXᵀXXᵀȲ + bȲXᵀX + cȲXᵀXXᵀX + cXXᵀȲXᵀX + bXȲᵀX + cXXᵀXȲᵀX + cXȲᵀXXᵀX
    end                                 #   = aȲ + b(ȲXᵀX + XȲᵀX + XXᵀȲ) + c(ȲXᵀXXᵀX + XȲᵀXXᵀX + XXᵀȲXᵀX + XXᵀXȲᵀX + XXᵀXXᵀȲ)
    return X̄
end

function ChainRulesCore.rrule(::typeof(newtonschulz5), X::AbstractMatrix, coefficients)
    checkpoints = [X]
    A = similar_square(X)
    B = similar_square(X)
    local Y
    for (a, b, c) in coefficients
        X′ = last(checkpoints)
        Y = copy(X′)
        newtonschulz5_step!((; A, B, Y), X′, (; a, b, c))
        push!(checkpoints, Y)
    end

    function pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        X̄ = similar(X)
        C = similar_square(X)
        for (k, (a, b, c)) in Iterators.reverse(enumerate(coefficients))
            vjp_newtonschulz5_step!(
                (; A, B, C), X̄,
                Ȳ, checkpoints[k], (; a, b, c))
            if k > 1
                Ȳ, X̄ = X̄, Ȳ
            end
        end
        return NoTangent(), X̄, NoTangent()
    end

    return Y, pullback
end
