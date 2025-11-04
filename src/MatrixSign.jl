module MatrixSign

using Base: @constprop
using BFloat16s: BFloat16
using ChainRulesCore: ChainRulesCore, rrule, @ignore_derivatives
using FillArrays: Fill, Eye
using LinearAlgebra
using MacroTools: @capture
using NNlib: batched_mul!, batched_transpose

export msign, msign!
export PolarExpress, JordanMethod, SVDMethod

include("newtonschulz5/newtonschulz5.jl")
include("msign.jl")

end
