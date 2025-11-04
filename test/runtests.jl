using MatrixSign
using Test
using LinearAlgebra
using Statistics

using Pkg
Pkg.activate(temp=true)

# ENV["MATRIXSIGN_TEST_CUDA"] = "true"
const MATRIXSIGN_TEST_CUDA = get(ENV, "MATRIXSIGN_TEST_CUDA", "false") == "true"

const TEST_TYPES, INITS = if MATRIXSIGN_TEST_CUDA
    Pkg.add("CUDA")
    using CUDA, BFloat16s
    zip([BFloat16, Float32, Float64], [7, 8, 12], [1.01, 1.001, 1.00001]),
    [CUDA.rand, CUDA.randn]
else
    zip([Float32, Float64], [8, 12], [1.001, 1.00001]),
    [rand, randn]
end

@testset "MatrixSign.jl" begin

    @testset for (T, steps, rtol) in TEST_TYPES,
                    s in [8, 32, 128, 512],
                    r in [1, 2, 4, 1//2, 1//4],
                    b in [(), 1, 2],
                    fused in [1, 2, 3, 4],
                    init in INITS
        X = init(T, s, Int(s * r), b...)
        O = @view msign(X, PolarExpress; fused, steps)[:, :, end]
        @test opnorm(O) ≈ 1 rtol=rtol
        @test mean(svd(O).S) ≈ 1 rtol=rtol
    end

    # TODO: test newtonschulz5 pullback

end
