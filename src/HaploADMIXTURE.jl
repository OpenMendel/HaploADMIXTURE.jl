module HaploADMIXTURE
using SnpArrays
using Base.Threads
import LinearAlgebra
import LinearAlgebra: svd
using OpenADMIXTURE
include("structs.jl")
include("utils.jl")
include("loops.jl")
include("algorithms_inner.jl")
include("algorithms_outer.jl")
end
