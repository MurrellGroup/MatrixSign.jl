include("dsl.jl")
include("naive.jl")
include("fused.jl")

is_tall(X) = size(X, 1) > size(X, 2)
aspect_ratio(X) = is_tall(X) ? size(X, 1) / size(X, 2) : size(X, 2) / size(X, 1)
similar_square(X::AbstractArray, n = min(size(X, 1), size(X, 2))) = similar(X, n, n, size(X)[3:end]...)

function newtonschulz5!(X::AbstractArray, coefficients; fused=1)
    X′ = reshape(X, size(X, 1), size(X, 2), :)
    if aspect_ratio(X′) < 2.5 || fused < 2
        naive_newtonschulz5!(X′, coefficients)
    else
        fused_newtonschulz5!(X′, coefficients; interval=fused)
    end
    return X
end

function newtonschulz5(X::AbstractArray, args...; kws...)
    newtonschulz5!(copy(X), args...; kws...)
end
