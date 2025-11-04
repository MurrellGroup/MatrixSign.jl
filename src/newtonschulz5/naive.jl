# equivalent to `aX + bXX'X + cXX'XX'X`
# expects Y == X && !(Y === X)
function naive_newtonschulz5_step!(
    (; A, B, Y),
    X, (; a, b, c)
)
    if is_tall(X)
        XᵀX = @mul! A = (X)ᵀ * X
        B .= XᵀX
        @mul! b(B) + c(XᵀX^2)        # B ← bXᵀX + cXᵀXXᵀX
        @mul! a(Y) + X * B           # Y ← aX + XB = aX + bXXᵀX + cXXᵀXXᵀX
    else
        XXᵀ = @mul! A = X * (X)ᵀ
        @mul! b(B .= XXᵀ) + c(XXᵀ^2) # B ← bXXᵀ + cXXᵀXXᵀ
        @mul! a(Y) + B * X           # Y ← aX + BX = aX + bXXᵀX + cXXᵀXXᵀX
    end
    return Y
end

function naive_newtonschulz5!(X::AbstractArray, coefficients)
    A, B, Y = similar_square(X), similar_square(X), similar(X)
    for (a, b, c) in coefficients
        copyto!(Y, X)
        naive_newtonschulz5_step!((; A, B, Y=X), Y, (; a, b, c))
    end
    return X
end
