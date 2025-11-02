module MatrixSign

using ChainRulesCore
using LinearAlgebra

export msign, msign!!
export SVDMethod, JordanMethod, YouMethod, PolarExpress

maywrite(::DenseArray) = true
maywrite(_) = false

include("newtonschulz5.jl")
include("msign.jl")

end
