@doc raw"""
    PolarExpress <: QuinticNewtonSchulzMethod

Uses a variable-step quintic Newton-Schulz iteration to achieve
accuracy close to or equal to [`SVDMethod`](@ref) given enough steps,
depending on the numerical precision.

Reference: [The Polar Express: Optimal Matrix Sign Methods and Their
Application to the Muon Algorithm](https://arxiv.org/abs/2505.16932)
"""
abstract type PolarExpress <: QuinticNewtonSchulzMethod end

const COEFFICIENTS_POLAR_EXPRESS = (
    (8.28721201814563,   -23.595886519098837, 17.300387312530933),
    (4.107059111542203,  -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949,  0.5518191394370137),
    (3.3184196573706015, -2.488488024314874,  0.51004894012372),
    (2.300652019954817,  -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398,  -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)

function get_coefficients(::Type{PolarExpress}, ::Val{F}, ::Type{T}, ::Val{S}) where {F,T,S}
    coefficients = map(x -> T.(x), COEFFICIENTS_POLAR_EXPRESS)
    N = length(coefficients)
    scaled = map(enumerate(coefficients)) do (i, (a, b, c))
        f = i < N ? T(F) : one(T)
        (a / f, b / f^3, c / f^5)
    end
    base = ntuple(i -> scaled[i], Val(min(S, N)))
    padding = ntuple(i -> last(scaled), Val(max(0, S - N)))
    return (base..., padding...)
end

Base.@constprop :aggressive function msign(
    X::AbstractArray{T}, ::Type{PolarExpress};
    steps::Int=8, batched::Bool=false
) where T
    coefficients = @ignore_derivatives get_coefficients(PolarExpress, Val(1.01), T, Val(steps))
    return msign(
        X, QuinticNewtonSchulzMethod;
        coefficients, batched, safety_factor = T(1.01)
    )
end

Base.@constprop :aggressive function msign!!(
    X::AbstractArray{T}, ::Type{PolarExpress};
    steps::Int=8, batched::Bool=false
) where T
    coefficients = get_coefficients(PolarExpress, Val(1.01), T, Val(steps))
    return msign!!(
        X, QuinticNewtonSchulzMethod;
        coefficients, batched, safety_factor = T(1.01)
    )
end
