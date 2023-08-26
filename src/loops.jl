const g_map_Float64 = Float64[0.0, 3.0, 1.0, 2.0]
const g_map_Float32 = Float32[0.0, 3.0, 1.0, 2.0]
"""
    qp_block!(qp_small00, qp_q, p, irange, jrange, K)
Compute a block of the matrix Q x P.

# Input
- `qp_small`: the output.
- `q`: Q matrix.
- `p`: P matrix.
- `irange`: i index range over which this function is applied
- `jrange`: j index range over which this function is applied
- `K`: number of clusters
"""
@inline function qp_block!(qp_small00::AbstractArray{T}, 
    qp_small01::AbstractArray{T},
    qp_small10::AbstractArray{T},
    qp_small11::AbstractArray{T},
    q, p, irange, jrange, K) where T
    tid = threadid()
    fill!(view(qp_small00, :, :, tid), zero(T))
    fill!(view(qp_small01, :, :, tid), zero(T))
    fill!(view(qp_small10, :, :, tid), zero(T))
    fill!(view(qp_small11, :, :, tid), zero(T))
    firsti, firstj = first(irange), first(jrange)
    @inbounds for j in jrange
        for i in irange
            for k in 1:K
                qp_small00[i-firsti+1, j-firstj+1, tid] += q[k, i] * p[k, 4(j-1) + 1]
                qp_small01[i-firsti+1, j-firstj+1, tid] += q[k, i] * p[k, 4(j-1) + 2]
                qp_small10[i-firsti+1, j-firstj+1, tid] += q[k, i] * p[k, 4(j-1) + 3]
                qp_small11[i-firsti+1, j-firstj+1, tid] += q[k, i] * p[k, 4j]
            end
        end
    end  
end

macro gcoefs_numeric()
    quote
        g1 = g[i, 2(j-1)+1]
        g2 = g[i, 2j]
        g11 = T(g1 == 1)
        g12 = T(g1 == 2)
        g10 = T(g1 == 0)
        g1M = T(g1 == 3)
        g1B = T(g1 != 3)
        g21 = T(g2 == 1)
        g22 = T(g2 == 2)
        g20 = T(g2 == 0)
        g2M = T(g2 == 3)
        g2B = T(g2 != 3)
    end |> esc
end

macro gcoefs_snparray()
    quote
        blk1 = gmat[(i - 1) >> 2 + 1, 2j-1]
        blk2 = gmat[(i - 1) >> 2 + 1, 2j]
        re = (i - 1) % 4
        blk1_shifted  = blk1 >> (re << 1)
        blk2_shifted  = blk2 >> (re << 1)
        g1_pre = blk1_shifted & 0x03
        g2_pre = blk2_shifted & 0x03 # 0b00: a1-a1, 0b01: missing, 0b10: a1-a2, 0b11: a2-a2
        g1 = (g1_pre == 0) ? 0.0 : 
            (g1_pre == 1) ? 3.0 : 
            (g1_pre == 2) ? 1.0 :
            2.0
        g2 = (g2_pre == 0) ? 0.0 : 
            (g2_pre == 1) ? 3.0 : 
            (g2_pre == 2) ? 1.0 :
            2.0
        g11 = T(g1 == 1)
        g12 = T(g1 == 2)
        g10 = T(g1 == 0)
        g1M = T(g1 == 3)
        g1B = T(g1 != 3)
        g21 = T(g2 == 1)
        g22 = T(g2 == 2)
        g20 = T(g2 == 0)
        g2M = T(g2 == 3)
        g2B = T(g2 != 3)
    end |> esc
end

macro coefs_likelihood()
    quote
        a1 = g1M * g2M + g11 * g21 * qp00 * qp11 + 
            ((g1M + g10) * (g20 + g21) + g2M * (g10 + g11) + g11 * g20) * qp00 +
            ((g1M + g10 + g11) * g22) * qp01 +
            (g12 * (g2M + g20 + g21)) * qp10 +
            (g12 * g22) * qp11

        a2 = g11 * g21 * qp01 * qp10 + g1M * (g20 + g21) * qp10 + 
            g2M * (g10 + g11) * qp01 + (g1M * g22 + g12 * g2M) * qp11

        b1 = g1M * g2M + g11 * g21 + 
            (g10 * g20 + g1M * g20 + g10 * g2M) * qp00 +
            (g1M + g10) * (g21 + g22) * qp01 +
            (g11 + g12) * (g2M + g20) * qp10 +
            (g11 * g22 + g12 * (g21 + g22)) * qp11
        b2 = (g1M * g21 + g1M * g22 + g11 * g2M + g12 * g2M) * qp11 + 
            g1M * g20 * qp10 + g10 * g2M * qp01
    end |> esc
end

macro qp_local_cpu()
    quote
        qp00 = qp_small00[i, j]
        qp01 = qp_small01[i, j]
        qp10 = qp_small10[i, j]
        qp11 = qp_small11[i, j]
    end |> esc
end
macro coefs()
    quote
        a0001 = qp00 / (qp00 + qp01) 
        a0010 = qp00 / (qp00 + qp10) 
        a0100 = qp01 / (qp01 + qp00)
        a0111 = qp01 / (qp01 + qp11)
        a1000 = qp10 / (qp10 + qp00)
        a1011 = qp10 / (qp10 + qp11)
        a1101 = qp11 / (qp11 + qp01)
        a1110 = qp11 / (qp11 + qp10)
        ahet0011 = qp00 * qp11 / (qp00 * qp11 + qp01 * qp10)
        ahet0110 = qp01 * qp10 / (qp00 * qp11 + qp01 * qp10)
        
        c00 = zero(T)
        c00 += g11 * g21 * ahet0011
        c00 += g1M * g2B * T(2 - g2) * a0010 
        c00 += g1B * g2M * T(2 - g1) * a0001
        c00 += g10 * g2B * T(2 - g2) 
        c00 += g11 * g20
        
        c01 = zero(T)
        c01 += g11 * g21 * ahet0110
        c01 += g1M * g2B * g2 * a0111
        c01 += g1B * g2M * T(2 - g1) * a0100
        c01 += g10 * g2B * g2
        c01 += g11 * g22

        c10 = zero(T)
        c10 += g11 * g21 * ahet0110
        c10 += g1M * g2B * T(2 - g2) * a1000
        c10 += g1B * g2M * g1 * a1011
        c10 += g12 * g2B * T(2 - g2)
        c10 += g11 * g20
        
        c11 = zero(T)
        c11 += g11 * g21 * ahet0011
        c11 += g1M * g2B * g2 * a1101
        c11 += g1B * g2M * g1 * a1110
        c11 += g12 * g2B * g2
        c11 += g11 * g22
    end |> esc
end

function loglikelihood_full_loop(g::AbstractArray{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    r = zero(T)
    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    @inbounds for j in jrange
        for i in irange      
            qp00 = qp_small00[i, j]
            qp01 = qp_small01[i, j]
            qp10 = qp_small10[i, j]
            qp11 = qp_small11[i, j]
            g1 = g[i, 2(j-1)+1]
            g2 = g[i, 2j] 
            # typeof(g) <: SnpLinAlg ? begin 
            #     gmat = g.s.data
            #     @gcoefs_snparray
            # end : begin 
            #     @gcoefs_numeric
            # end
            r_prev = r
            if g1 == 0 && g2 == 0
                r += 2log(qp00)
            elseif g1 == 0 && g2 == 2
                r += 2log(qp01)
            elseif g1 == 2 && g2 == 0
                r += 2log(qp10)
            elseif g1 == 2 && g2 == 2
                r += 2log(qp11)
            elseif g1 == 3 && g2 == 0
                r += 2log(qp00 + qp10)
            elseif g1 == 3 && g2 == 2
                r += 2log(qp01 + qp11)
            elseif g1 == 0 && g2 == 3
                r += 2log(qp00 + qp01)
            elseif g1 == 2 && g2 == 3
                r += 2log(qp10 + qp11)
            elseif g1 == 3 && g2 == 1
                r += log(qp00 + qp10) + log(qp01 + qp11)
            elseif g1 == 1 && g2 == 3
                r += log(qp00 + qp01) + log(qp10 + qp11)
            elseif g1 == 0 && g2 == 1
                r += log(qp00) + log(qp01)
            elseif g1 == 2 && g2 == 1
                r += log(qp10) + log(qp11)
            elseif g1 == 1 && g2 == 0
                r += log(qp00) + log(qp10)
            elseif g1 == 1 && g2 == 2
                r += log(qp01) + log(qp11)
            elseif g1 == 1 && g2 == 1
                r += log(qp00 * qp11 + qp01 * qp10)
            end
        end
    end
    return r
end

function loglikelihood_full_loop2(g::AbstractArray{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    r = zero(T)
    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    @inbounds for j in jrange
        for i in irange      
            @gcoefs_numeric
            @qp_local_cpu
            @coefs_likelihood
            r += log(a1 + a2) + log(b1 + b2)
        end
    end
    return r
end

function loglikelihood_full_loop2(g::SnpLinAlg{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    r = zero(T)
    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    gmat = g.s.data
    g_map = T == Float64 ? g_map_Float64 : g_map_Float32
    @inbounds for j in jrange
        for i in irange 
            @gcoefs_snparray
            @qp_local_cpu
            @coefs_likelihood
            r += log(a1 + a2) + log(b1 + b2)
        end
    end
    return r
end

function loglikelihood_loop(g::AbstractArray{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, irange, jrange, K) where T
    oneT = one(T)
    twoT = 2one(T)
    firsti, firstj = first(irange), first(jrange)
    r00 = zero(Float64)
    r01 = zero(Float64)
    r10 = zero(Float64)
    r11 = zero(Float64)

    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    @inbounds for j in jrange
        for i in irange     
            typeof(g) <: SnpLinAlg ? begin 
                gmat = g.s.data
                @gcoefs_snparray
            end : begin 
                @gcoefs_numeric
            end
            @qp_local_cpu
            @coefs   
            r00 += c00 * log(qp00)
            r01 += c01 * log(qp01)
            r10 += c10 * log(qp10)
            r11 += c11 * log(qp11)   
        end
    end
    r00 + r01 + r10 + r11
end

function em_loop!(q_next, p_next, g::AbstractArray{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, 
    irange, jrange, K; fix_p=false, fix_q=false) where T
    firsti, firstj = first(irange), first(jrange)
    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    @inbounds for j in jrange
        for i in irange
            typeof(g) <: SnpLinAlg ? begin 
                gmat = g.s.data
                @gcoefs_snparray
            end : begin 
                @gcoefs_numeric
            end
            @qp_local_cpu
            @coefs
            for k in 1:K
                if !fix_q
                    q_next[k, i] += c00 * q[k, i] * p[k, 4(j-1)+1] / qp00
                    q_next[k, i] += c01 * q[k, i] * p[k, 4(j-1)+2] / qp01
                    q_next[k, i] += c10 * q[k, i] * p[k, 4(j-1)+3] / qp10
                    q_next[k, i] += c11 * q[k, i] * p[k, 4j      ] / qp11
                end
                if !fix_p
                    p_next[k, 4(j-1)+1] += c00 * q[k, i] * p[k, 4(j-1)+1] / qp00
                    p_next[k, 4(j-1)+2] += c01 * q[k, i] * p[k, 4(j-1)+2] / qp01
                    p_next[k, 4(j-1)+3] += c10 * q[k, i] * p[k, 4(j-1)+3] / qp10
                    p_next[k, 4j      ] += c11 * q[k, i] * p[k, 4j      ] / qp11
                end
            end
        end
    end
end

function update_q_loop!(XtX, Xtz, g::AbstractArray{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, 
    irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    # if !(q_ === q && p_ === p)
    #     qp_block!(qp_small00_, qp_small01_, qp_small10_, qp_small11_, q_, p_, irange, jrange, K)
    # end
    @inbounds for j in jrange
        for i in irange
            typeof(g) <: SnpLinAlg ? begin 
                gmat = g.s.data
                @gcoefs_snparray
            end : begin 
                @gcoefs_numeric
            end
            @qp_local_cpu
            @coefs
            for k in 1:K
                Xtz[k, i] += c00 * p[k, 4(j-1)+1] / qp00
                Xtz[k, i] += c01 * p[k, 4(j-1)+2] / qp01
                Xtz[k, i] += c10 * p[k, 4(j-1)+3] / qp10
                Xtz[k, i] += c11 * p[k, 4j      ] / qp11
                for k2 in 1:K
                    XtX[k2, k, i] += c00 * p[k, 4(j-1)+1] * p[k2, 4(j-1)+1] / qp00^2
                    XtX[k2, k, i] += c01 * p[k, 4(j-1)+2] * p[k2, 4(j-1)+2] / qp01^2
                    XtX[k2, k, i] += c10 * p[k, 4(j-1)+3] * p[k2, 4(j-1)+3] / qp10^2
                    XtX[k2, k, i] += c11 * p[k, 4j      ] * p[k2, 4j      ] / qp11^2
                end
            end

        end
    end
end

function update_p_loop!(XtX, Xtz, g::AbstractArray{T}, q, p, qp_small00, qp_small01, qp_small10, qp_small11, 
    irange, jrange, K) where T
    firsti, firstj = first(irange), first(jrange)
    qp_block!(qp_small00, qp_small01, qp_small10, qp_small11, q, p, irange, jrange, K)
    # if !(q_ === q && p_ === p)
    #     qp_block!(qp_small00_, qp_small01_, qp_small10_, qp_small11_, q_, p_, irange, jrange, K)
    # end
    @inbounds for j in jrange
        for i in irange
            typeof(g) <: SnpLinAlg ? begin 
                gmat = g.s.data 
                @gcoefs_snparray
            end : begin 
                @gcoefs_numeric
            end
            @qp_local_cpu
            @coefs
            for k in 1:K
                Xtz[k, 4(j-1)+1] += c00 * q[k, i] / qp00
                Xtz[k, 4(j-1)+2] += c01 * q[k, i] / qp01
                Xtz[k, 4(j-1)+3] += c10 * q[k, i] / qp10
                Xtz[k, 4j      ] += c11 * q[k, i] / qp11
                for k2 in 1:K
                    XtX[k2, k, 4(j-1)+1] += c00 * q[k, i] * q[k2, i] / qp00^2
                    XtX[k2, k, 4(j-1)+2] += c01 * q[k, i] * q[k2, i] / qp01^2
                    XtX[k2, k, 4(j-1)+3] += c10 * q[k, i] * q[k2, i] / qp10^2
                    XtX[k2, k, 4j      ] += c11 * q[k, i] * q[k2, i] / qp11^2
                end
            end
        end
    end
end
