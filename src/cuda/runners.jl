using .CUDA
function loglikelihood(g::CuArray{UInt8, 2}, q::CuArray{T, 2}, p::CuArray{T, 2}, I, J) where T
    out = CUDA.zeros(Float64)
    kernel = @cuda launch=false loglikelihood_kernel(out, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    CUDA.@sync kernel(out, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))
    out[]
end

function loglikelihood(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}, q::CuArray{T, 2}, p::CuArray{T, 2}) where T
    I, J, K = d.I, d.J, d.K
    return loglikelihood(g_cu, q, p, I, J)
end

function em!(q_next, p_next, g::CuArray{UInt8, 2}, 
    q::CuArray{T, 2}, p::CuArray{T, 2}, I, J) where T
    fill!(q_next, zero(eltype(q_next)))
    fill!(p_next, zero(eltype(q_next)))
    kernel = @cuda launch=false em_kernel!(q_next, p_next, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(q_next, p_next, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(cld(I, threads_1d), cld(I, threads_1d)))
    q_next, p_next
end

function em!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = d.I, d.J, d.K
    em!(d.q_next, d.p_next, g_cu, d.q, d.p, I, J)
    d.q_next, d.p_next
end

function update_q_cuda!(XtX, Xtz, g, q, p, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_q_kernel!(XtX, Xtz, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_q_cuda!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = d.I, d.J, d.K
    update_q_cuda!(d.XtX_q, d.Xtz_q, g_cu, d.q, d.p, I, J)
end

function update_p_cuda!(XtX, Xtz, g, q, p, I, J)
    fill!(XtX, zero(eltype(XtX)))
    fill!(Xtz, zero(eltype(Xtz)))
    kernel = @cuda launch=false update_p_kernel!(XtX, Xtz, g, q, p, 1:I, 1:J)
    config = launch_configuration(kernel.fun)
    threads = config.threads
    threads_1d = Int(floor(sqrt(threads)))
    # println("threads: $threads")
    CUDA.@sync kernel(XtX, Xtz, g, q, p, 1:I, 1:J; threads=(threads_1d, threads_1d), 
        blocks=(threads_1d, threads_1d))#cld(I, threads_1d), cld(I, threads_1d)))
    nothing
end

function update_p_cuda!(d::CuAdmixData{T}, g_cu::CuMatrix{UInt8}) where T
    I, J, K = d.I, d.J, d.K
    update_p_cuda!(d.XtX_p, d.Xtz_p, g_cu, d.q, d.p, I, J)  
end
