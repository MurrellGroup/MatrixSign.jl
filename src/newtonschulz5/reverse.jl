function vjp_naive_newtonschulz5!(
    (; A, B, C), X̄::AbstractArray,
    Ȳ::AbstractArray, X::AbstractArray, (; a, b, c)
)
    if is_tall(X)
        XᵀX = @mul! A = (X)ᵀ * X
        @mul! b(B .= XᵀX) + c(XᵀX^2)     # B ← bXᵀX + cXᵀXXᵀX
        @mul! a(X̄ .= Ȳ) + Ȳ * B          # X̄ ← aȲ + bȲXᵀX + cȲXᵀXXᵀX
        XᵀȲ = @mul! B = (X)ᵀ * Ȳ
        @mul! b(C .= XᵀȲ) + c(XᵀȲ * XᵀX) # C ← bXᵀȲ + cXᵀȲXᵀX
        @mul! C + c(XᵀX * XᵀȲ)           # C ← C + cXᵀXXᵀȲ = bXᵀȲ + cXᵀȲXᵀX + cXᵀXXᵀȲ
        @mul! X̄ + X * (B .= C .+ (C)ᵀ)   # X̄ ← X̄ + X(C + Cᵀ)
                                         #   = X̄ + X(bXᵀȲ + cXᵀȲXᵀX + cXᵀXXᵀȲ + bȲᵀX + cXᵀXȲᵀX + cȲᵀXXᵀX)
                                         #   = aȲ + bȲXᵀX + cȲXᵀXXᵀX + bXXᵀȲ + cXXᵀȲXᵀX + cXXᵀXXᵀȲ + bXȲᵀX + cXXᵀXȲᵀX + cXȲᵀXXᵀX
    else                                 #   = aȲ + b(ȲXᵀX + XȲᵀX + XXᵀȲ) + c(ȲXᵀXXᵀX + XȲᵀXXᵀX + XXᵀȲXᵀX + XXᵀXȲᵀX + XXᵀXXᵀȲ)
        XXᵀ = @mul! A = X * (X)ᵀ
        @mul! b(B .= XXᵀ) + c(XXᵀ^2)     # B ← bXXᵀ + cXXᵀXXᵀ
        @mul! a(X̄ .= Ȳ) + B * Ȳ          # X̄ ← aȲ + bXXᵀȲ + cXXᵀXXᵀȲ
        ȲXᵀ = @mul! B = Ȳ * (X)ᵀ
        @mul! b(C .= ȲXᵀ) + c(ȲXᵀ * XXᵀ) # C ← bȲXᵀ + cȲXᵀXXᵀ
        @mul! C + c(XXᵀ * ȲXᵀ)           # C ← C + cXXᵀȲXᵀ = bȲXᵀ + cȲXᵀXXᵀ + cXXᵀȲXᵀ
        @mul! X̄ + (B .= C .+ (C)ᵀ) * X   # X̄ ← X̄ + (C + Cᵀ)X
                                         #   = X̄ + (bȲXᵀ + cȲXᵀXXᵀ + cXXᵀȲXᵀ + bXȲᵀ + cXXᵀXȲᵀ + cXȲᵀXXᵀ)X
                                         #   = aȲ + bXXᵀȲ + cXXᵀXXᵀȲ + bȲXᵀX + cȲXᵀXXᵀX + cXXᵀȲXᵀX + bXȲᵀX + cXXᵀXȲᵀX + cXȲᵀXXᵀX
    end                                  #   = aȲ + b(ȲXᵀX + XȲᵀX + XXᵀȲ) + c(ȲXᵀXXᵀX + XȲᵀXXᵀX + XXᵀȲXᵀX + XXᵀXȲᵀX + XXᵀXXᵀȲ)
    return X̄
end

function ChainRulesCore.rrule(::typeof(newtonschulz5), X::AbstractArray, coefficients)
    checkpoints = [X]
    A = similar_square(X)
    B = similar_square(X)
    local Y
    for (a, b, c) in coefficients
        X′ = last(checkpoints)
        Y = copy(X′)
        naive_newtonschulz5_step!((; A, B, Y), X′, (; a, b, c))
        push!(checkpoints, Y)
    end

    function pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        X̄ = similar(X)
        C = similar_square(X)
        for (k, (a, b, c)) in Iterators.reverse(enumerate(coefficients))
            vjp_naive_newtonschulz5_step!(
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
