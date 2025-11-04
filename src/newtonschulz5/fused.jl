# h(Y) = aI + bY + cY²
# p(X) = aX + bXXᵀX + cXXᵀXXᵀX
#      = X * h(XᵀX)
#      = h(XXᵀ) * X
function h!(
    Y′::AbstractArray, Y::AbstractArray,
    (a, b, c)
)
    Y′ .= Diagonal(Fill(a, size(Y, 1))) .+ b .* Y
    @mul! Y′ + c(Y * Y)
    return Y′
end

# (pₖ ∘ … ∘ p₁)(X)
function fused_newtonschulz5_steps!(
    (; A, B, C, D, X′),
    X::AbstractArray, coefficients;
    A_scaling=false
)
    @assert allequal(size, (A, B, C, D))
    @assert size(X′) == size(X)
    Y = if is_tall(X)
        @mul! A_scaling * A + (X)ᵀ * X
    else
        @mul! A_scaling * A + X * (X)ᵀ
    end
    Qₜ = h!(B, Y, coefficients[1])
    K = length(coefficients)
    for t in 2:K
        Rₜ = @mul! D = (@mul! C = (Qₜ)ᵀ * Y) * Qₜ
        hₜRₜ = h!(C, Rₜ, coefficients[t])
        Qₜ = @mul! B = (D .= Qₜ) * hₜRₜ
    end
    if is_tall(X)
        @mul! X′ = X * Qₜ
    else
        @mul! X′ = Qₜ * X
    end
    return X′
end

function fused_newtonschulz5!(
    X::AbstractArray, coefficients;
    interval = 3
)
    A, B, C, D = ntuple(_ -> similar_square(X), 4)
    X′ = similar(X)
    for (i, chunk) in enumerate(Iterators.partition(coefficients, interval))
        copyto!(X′, X)
        A_scaling = if i == 1
            A .= Eye{Bool}(size(A, 1))
            1f-3 # for numerical stability: uniform input values, 1024x1024, BFloat16, interval=3
        else
            false
        end
        fused_newtonschulz5_steps!((; A, B, C, D, X′=X), X′, chunk; A_scaling)
    end
    return X′
end
