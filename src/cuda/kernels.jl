using .CUDA
macro qp_local_gpu()
    quote
        qp00 = zero(T2)
        qp01 = zero(T2)
        qp10 = zero(T2)
        qp11 = zero(T2)
        for k in 1:K
            qp00 += q[k, i] * p[k, 4(j-1) + 1]
            qp01 += q[k, i] * p[k, 4(j-1) + 2]
            qp10 += q[k, i] * p[k, 4(j-1) + 3]
            qp11 += q[k, i] * p[k, 4j]
        end
    end |> esc
end
@inline function loglikelihood_kernel(out, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    acc = zero(Float64)
    T2 = eltype(out)
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            @qp_local_gpu
            gmat = g
            @gcoefs_snparray
            @coefs_likelihood
            @inbounds acc += log(a1 + a2) + log(b1 + b2)
        end
    end
    CUDA.@atomic out[] += acc
    return nothing
end

@inline function em_kernel!(q_next, p_next, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    T2 = T
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            @qp_local_gpu
            gmat = g
            @gcoefs_snparray
            @coefs
            for k in 1:K
                CUDA.@atomic q_next[k, i] += c00 * q[k, i] * p[k, 4(j-1)+1] / qp00
                CUDA.@atomic q_next[k, i] += c01 * q[k, i] * p[k, 4(j-1)+2] / qp01
                CUDA.@atomic q_next[k, i] += c10 * q[k, i] * p[k, 4(j-1)+3] / qp10
                CUDA.@atomic q_next[k, i] += c11 * q[k, i] * p[k, 4j      ] / qp11
                CUDA.@atomic p_next[k, 4(j-1)+1] += c00 * q[k, i] * p[k, 4(j-1)+1] / qp00
                CUDA.@atomic p_next[k, 4(j-1)+2] += c01 * q[k, i] * p[k, 4(j-1)+2] / qp01
                CUDA.@atomic p_next[k, 4(j-1)+3] += c10 * q[k, i] * p[k, 4(j-1)+3] / qp10
                CUDA.@atomic p_next[k, 4j      ] += c11 * q[k, i] * p[k, 4j      ] / qp11
            end
        end
    end
    return nothing
end

@inline function em_p_kernel!(p_next, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    T2 = T
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            @qp_local_gpu
            gmat = g
            @gcoefs_snparray
            @coefs
            for k in 1:K
                CUDA.@atomic p_next[k, 4(j-1)+1] += c00 * q[k, i] * p[k, 4(j-1)+1] / qp00
                CUDA.@atomic p_next[k, 4(j-1)+2] += c01 * q[k, i] * p[k, 4(j-1)+2] / qp01
                CUDA.@atomic p_next[k, 4(j-1)+3] += c10 * q[k, i] * p[k, 4(j-1)+3] / qp10
                CUDA.@atomic p_next[k, 4j      ] += c11 * q[k, i] * p[k, 4j      ] / qp11
            end
        end
    end
    return nothing
end

@inline function update_q_kernel!(XtX, Xtz, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    T2 = T
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            @qp_local_gpu
            gmat = g
            @gcoefs_snparray
            @coefs
            for k in 1:K
                CUDA.@atomic Xtz[k, i] += c00 * p[k, 4(j-1)+1] / qp00
                CUDA.@atomic Xtz[k, i] += c01 * p[k, 4(j-1)+2] / qp01
                CUDA.@atomic Xtz[k, i] += c10 * p[k, 4(j-1)+3] / qp10
                CUDA.@atomic Xtz[k, i] += c11 * p[k, 4j      ] / qp11
                for k2 in 1:K
                    CUDA.@atomic XtX[k2, k, i] += c00 * p[k, 4(j-1)+1] * p[k2, 4(j-1)+1] / qp00^2
                    CUDA.@atomic XtX[k2, k, i] += c01 * p[k, 4(j-1)+2] * p[k2, 4(j-1)+2] / qp01^2
                    CUDA.@atomic XtX[k2, k, i] += c10 * p[k, 4(j-1)+3] * p[k2, 4(j-1)+3] / qp10^2
                    CUDA.@atomic XtX[k2, k, i] += c11 * p[k, 4j      ] * p[k2, 4j      ] / qp11^2
                end
            end
        end
    end

    return nothing
end

@inline function update_p_kernel!(XtX, Xtz, g::AbstractArray{UInt8, 2}, 
    q::AbstractArray{T, 2}, p::AbstractArray{T, 2}, irange, jrange, joffset=0) where T
    K = size(q, 1)
    firsti, firstj = first(irange), first(jrange)
    lasti, lastj = last(irange), last(jrange)
    xindex = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    xstride = blockDim().x * gridDim().x
    yindex = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    ystride = blockDim().y * gridDim().y   
    oneT = one(T)
    twoT = 2one(T)
    T2 = T
    @inbounds for j = (firstj + yindex - 1):ystride:lastj
        for i = (firsti + xindex - 1):xstride:lasti
            qp_local = zero(eltype(XtX))
            @qp_local_gpu
            gmat = g
            @gcoefs_snparray
            @coefs
            for k in 1:K
                CUDA.@atomic Xtz[k, 4(j-1)+1] += c00 * q[k, i] / qp00
                CUDA.@atomic Xtz[k, 4(j-1)+2] += c01 * q[k, i] / qp01
                CUDA.@atomic Xtz[k, 4(j-1)+3] += c10 * q[k, i] / qp10
                CUDA.@atomic Xtz[k, 4j      ] += c11 * q[k, i] / qp11
                for k2 in 1:K
                    CUDA.@atomic XtX[k2, k, 4(j-1)+1] += c00 * q[k, i] * q[k2, i] / qp00^2
                    CUDA.@atomic XtX[k2, k, 4(j-1)+2] += c01 * q[k, i] * q[k2, i] / qp01^2
                    CUDA.@atomic XtX[k2, k, 4(j-1)+3] += c10 * q[k, i] * q[k2, i] / qp10^2
                    CUDA.@atomic XtX[k2, k, 4j      ] += c11 * q[k, i] * q[k2, i] / qp11^2
                end
            end
        end
    end

    return nothing
end
