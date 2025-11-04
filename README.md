# MatrixSign

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/MatrixSign.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/MatrixSign.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/MatrixSign.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/MatrixSign.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/MatrixSign.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/MatrixSign.jl)

MatrixSign.jl is a Julia package for computing the matrix sign, or more generally, polar factor of a matrix. Three methods are provided:

- **Polar Express**[^1]: an optimized variable-step Newton-Schulz polynomial iteration with optimal coefficients to compute the polar factor more accurately.
- **Keller Jordan's 5-step**[^2]: a 5-step quintic Newton-Schulz polynomial iteration to compute a noisy approximation of the polar factor.
- **SVD**: uses SVD to compute the exact polar factor using $UV^T$.

## Features

- Newton-Schulz-based methods rely entirely on in-place generic matrix multiplication to minimize memory usage, and to be very fast on GPUs.
- Differentiable with custom chain rules.
- Fixed memory w.r.t. steps (except for checkpoints in chain rule).
- Fused polynomial iteration to optimize performance for rectangular matrices (See Algorithm 4 in Polar Express[^1]).
- Batched matrix sign for inputs with more than 2 dimensions.

## Usage

The main interface is the `msign` function, dispatching to the `PolarExpress` method by default:

```julia
julia> using MatrixSign

julia> X = randn(Float64, 1024, 1024);

julia> msign(X, steps=16) ≈ msign(X, SVDMethod)
true

julia> msign(Float32.(X), steps=14) ≈ msign(Float32.(X), SVDMethod)
true
```

For Newton-Schulz-based methods, `Float16` matrices would underflow, so they are converted to `BFloat16` by default, which may only perform well on supported hardware.

```julia
julia> using CUDA, BFloat16s, PrettyChairmarks

julia> X = CUDA.randn(BFloat16, 1024, 4096); # 1x4 aspect ratio

julia> @b CUDA.@sync msign(X, SVDMethod)
67.891 ms (713 allocs: 32.013 MiB, 1.19% gc time)

julia> CUDA.@allocated msign(X, SVDMethod)
71307272

julia> @b CUDA.@sync msign(X, PolarExpress, steps=8)
979.261 μs (2205 allocs: 59.531 KiB)

julia> @b CUDA.@sync msign(X, PolarExpress, steps=8, fused=3)
863.068 μs (2571 allocs: 77.516 KiB)

julia> @b CUDA.@sync msign(X, PolarExpress, steps=6)
771.161 μs (1759 allocs: 47.031 KiB)

julia> @b CUDA.@sync msign(X, PolarExpress, steps=6, fused=3)
684.425 μs (2024 allocs: 61.078 KiB)

julia> CUDA.@allocated msign(X, PolarExpress, steps=8)
29360890

julia> CUDA.@allocated msign(X, PolarExpress, steps=8, fused=3)
33555234
```

## Recommendations

- Use `PolarExpress` for most cases.
  - For many use-cases, particularly with `randn`-like inputs, the majority of singular values get close to 1 in just ~5-7 steps. A few extra steps may sometimes be necessary to bring up the smallest singular values, but this is not always important.
  - Take advantage of the `fused` keyword to optimize performance for rectangular matrices with aspect ratios great than `2.5`.
  - For smaller data types like `BFloat16`, `fused` beyond `3` can lead to error accumulation.
- Use `JordanMethod` for parity with the original Muon[^2].
- Use `SVDMethod` for robust, but slow comparison.

## Limitations

- Complex matrices are not supported.

## References

[^1] [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](https://arxiv.org/pdf/2505.16932)

[^2] [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
