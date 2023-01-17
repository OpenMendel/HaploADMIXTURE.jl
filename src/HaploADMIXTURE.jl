module HaploADMIXTURE
using SnpArrays
using Base.Threads
import LinearAlgebra
import LinearAlgebra: svd
using OpenADMIXTURE
using Requires, Adapt
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        using .CUDA
        CUDA.allowscalar(true)
        include("cuda/structs.jl")
        include("cuda/kernels.jl")
        include("cuda/runners.jl")
    end
end
include("structs.jl")
include("utils.jl")
include("loops.jl")
include("algorithms_inner.jl")
include("algorithms_outer.jl")
end
