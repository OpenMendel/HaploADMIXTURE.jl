# function loglikelihood(d, g::AbstractArray{T}, q, p) where T
#     I, J, K = d.I, d.J, d.K
#     r = loglikelihood_loop(g, q, p, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11, 1:I, 1:J, K)
#     # r = tiler_scalar(loglikelihood_loop, typeof(qp_small), zero(T), (g, q, p, qp_small), 1:I, 1:J, K)
#     r
# end

import OpenADMIXTURE: tiler_scalar, threader_scalar, threader!

function loglikelihood_full(d::AdmixData2{T, T2}, g::AbstractArray{T2}, q, p) where {T, T2}
    I, J, K = d.I, d.J, d.K
    # r = loglikelihood_full_loop(g, q, p, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11, 1:I, 1:J, K)
    tid = Threads.threadid()

    r = threader_scalar(loglikelihood_full_loop, typeof(qp_small00), zero(T), (g, q, p, 
        d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11), 1:I, 1:J, K)
    r
end

function loglikelihood_full2(d::AdmixData2{T, T2}, g::AbstractArray{T2}, q, p) where {T, T2}
    I, J, K = d.I, d.J, d.K
    tid = Threads.threadid()
    r = threader_scalar(loglikelihood_full_loop2, typeof(d.qp_small00[1]), zero(T), (g, q, p, 
        d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11), 1:I, 1:J, K)
    # r = loglikelihood_full_loop2(g, q, p, qp_small00, qp_small01, qp_small10, qp_small11, 1:I, 1:J, K)
    r 
end

function em!(d::AdmixData2{T, T2}, g::AbstractArray{T2}; d_cu=nothing, g_cu=nothing,
    fix_q=false, fix_p=false) where {T, T2}
    I, J, K = d.I, d.J, d.K
    # ll_prev = loglikelihood_full(d, g, d.q, d.p)
    if d_cu !== nothing
        d.q_T2 .= d.q
        d.p_T2 .= d.p
        OpenADMIXTURE.copyto_sync!([d_cu.q, d_cu.p], [d.q_T2, d.p_T2])
        if !fix_q
            em!(d_cu, g_cu)
        else
            em_p!(d_cu, g_cu)
        end
        OpenADMIXTURE.copyto_sync!([d.q_T2, d.p_T2], [d_cu.q_next, d_cu.p_next])
        if !fix_q
            d.q_next .= d.q_T2
        else 
            d.q_next .= d.q
        end
        if !fix_p
            d.p_next .= d.p_T2
        else
            d.p_next = d.p
        end
    else
        fill!(d.q_next, zero(T))
        fill!(d.p_next, zero(T))
        if !fix_q
            threader!(em_loop!, eltype(d.q_next), (d.q_next, d.p_next, g, d.q, d.p, 
                d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11), 1:I, 1:J, K, true; maxL=64)
            # em_loop!(d.q_next, d.p_next, g, d.q_T2, d.p_T2, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11, 1:I, 1:J, K)
        else
            threader!(em_p_loop!, eltype(d.p_next), (d.p_next, g, d.q, d.p, 
                d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11), 1:I, 1:J, K, true; maxL=64)
            # em_p_loop!(d.p_next, g, d.q_T2, d.p_T2, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11, 1:I, 1:J, K)
        end
    end
    d.q_next ./= 2(J .- d.doublemissing)
    @inbounds for j in 1:J
        for k in 1:K
            s = d.p_next[k, 4(j-1)+1] + d.p_next[k, 4(j-1)+2] + d.p_next[k, 4(j-1)+3] + d.p_next[k, 4(j-1)+4]
            for l in 1:4
                d.p_next[k, 4(j-1)+l] /= s
            end
        end
    end
    OpenADMIXTURE.project_q!(d.q_next, d.idxv[1])
    project_p!(d.p_next, d.idx4v[1], K)
    if d_cu !== nothing
        d.q_T2 .= d.q_next
        d.p_T2 .= d.p_next
        OpenADMIXTURE.copyto_sync!([d_cu.q_next, d_cu.p_next], [d.q_T2, d.p_T2])  
    end
    # ll_new = loglikelihood_full(d, g, d.q_next, d.p_next)#; q_=d.q, p_=d.p)
    # @info "em_update: ll_new=$ll_new, ll_prev=$ll_prev $(ll_new > ll_prev)"
end

"""
    update_q!(d, g, update2=false; d_cu=nothing, g_cu=nothing)
Update Q using sequential quadratic programming.

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `update2`: if running the second update for quasi-Newton step
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of 
"""
function update_q!(d::AdmixData2{T, T2}, g::AbstractArray{T2}, update2=false;
    d_cu=nothing, g_cu=nothing, penalty=false, rho=1e7, 
    guarantee_increase=false) where {T, T2}
    I, J, K = d.I, d.J, d.K
    qdiff, XtX, Xtz, XtX_T2, Xtz_T2, qp_small00, qp_small01, qp_small10, qp_small11 = d.q_tmp, d.XtX_q, d.Xtz_q, d.XtX_q_T2, d.Xtz_q_T2, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11
    q_T2, p_T2 = d.q_T2, d.p_T2
    q_next = update2 ? d.q_next2 : d.q_next
    q      = update2 ? d.q_next  : d.q
    p      = update2 ? d.p_next  : d.p
    qv     = update2 ? d.q_nextv : d.qv
    qdiffv = d.q_tmpv
    XtXv   = d.XtX_qv
    Xtzv   = d.Xtz_qv

    # qf!(qp, q, f)
    # ll_prev = loglikelihood_full(d, g, q, p) # loglikelihood(g, qf)
    # println(d.ll_prev)
    @time if d_cu === nothing # CPU operation
        fill!(XtX_T2, zero(T2))
        fill!(Xtz_T2, zero(T2))
        # update_q_loop!(XtX_T2, Xtz_T2, g, q, p, qp_small00, qp_small01, qp_small10, qp_small11, 1:I, 1:J, K)
        threader!(update_q_loop!, 
            eltype(XtX), (XtX_T2, Xtz_T2, g, q, p, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11), 
            1:I, 1:J, K, true; maxL=16)
    else # GPU operation
        q_T2 .= q
        p_T2 .= p
        OpenADMIXTURE.copyto_sync!([d_cu.q, d_cu.p], [q_T2, p_T2])
        update_q_cuda!(d_cu, g_cu)
        OpenADMIXTURE.copyto_sync!([XtX_T2, Xtz_T2], [d_cu.XtX_q, d_cu.Xtz_q])
    end

    XtX .= XtX_T2
    Xtz .= Xtz_T2

    @assert !any(isnan.(XtX))
    @assert !any(isnan.(Xtz))
    # Solve the quadratic programs
    @time begin
        rho = convert(T, rho)
        Xtz .*= -1 
        pmin = zeros(T, K) 
        pmax = ones(T, K) 

        @batch threadlocal=QPThreadLocal{T}(K) for i in 1:I
            # even the views are allocating something, so we use preallocated views.
            XtX_ = XtXv[i]
            Xtz_ = Xtzv[i]
            q_ = qv[i]
            qdiff_ = qdiffv[i]

            # t = threadid()
            tableau_k1 = threadlocal.tableau_k1
            tableau_k2 = threadlocal.tableau_k2
            tmp_k = threadlocal.tmp_k
            tmp_k1 = threadlocal.tmp_k1
            tmp_k1_ = threadlocal.tmp_k1_
            tmp_k2 = threadlocal.tmp_k2
            tmp_k2_ = threadlocal.tmp_k2_
            swept = threadlocal.swept

            if penalty
                XtX .+= rho
                OpenADMIXTURE.create_tableau!(tableau_k1, XtX_, Xtz_, q_, d.v_kk, tmp_k, false)
                OpenADMIXTURE.quadratic_program!(qdiff_, tableau_k1, q_, pmin, pmax, K, 0, 
                    tmp_k1, tmp_k1_, swept)                
            else
                OpenADMIXTURE.create_tableau!(tableau_k2, XtX_, Xtz_, q_, d.v_kk, tmp_k, true)
                OpenADMIXTURE.quadratic_program!(qdiff_, tableau_k2, q_, pmin, pmax, K, 1, 
                    tmp_k2, tmp_k2_, swept)
            end
        end
        factor = one(T)
        if guarantee_increase
            while True
                @inbounds for i in 1:I
                    for k in 1:K
                        q_next[k, i] = q[k, i] + factor * qdiff[k, i]
                    end
                end
                OpenADMIXTURE.project_q!(q_next, d.idxv[1])
                ll_new = if d_cu !== nothing
                    q_T2 .= q_next
                    p_T2 .= p
                    OpenADMIXTURE.copyto_sync!([d_cu.p, d_cu.q], [d.p_T2, d.q_T2])
                    loglikelihood(d_cu, g_cu, d_cu.q, d_cu.p)
                else
                    loglikelihood_full2(d, g, d.q_next2, d.p_next2)
                end
                if ll_new > d.ll_tmp 
                    d.ll_tmp = ll_new
                    break
                end
                @info "Step size halved in updated_q. Current step size: $factor"
                factor /= 2
            end
        else
            @inbounds for i in 1:I
                for k in 1:K
                    q_next[k, i] = q[k, i] + qdiff[k, i]
                end
            end
        end
    end
    # ll_new = loglikelihood_full(d, g, q_next, p)#; q_=q, p_=p)
    # @info "q_update: ll_new=$ll_new, ll_prev=$ll_prev $(ll_new > ll_prev)"
end

"""
    update_p!(d, g, update2=false; d_cu=nothing, g_cu=nothing)
Update Q using sequential quadratic programming.

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `update2`: if running the second update for quasi-Newton step
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of 
"""
function update_p!(d::AdmixData2{T,T2}, g::AbstractArray{T2}, update2=false;
    d_cu=nothing, g_cu=nothing, penalty=false, rho=1e7) where {T,T2}
    I, J, K = d.I, d.J, d.K
    pdiff, XtX, Xtz, XtX_T2, Xtz_T2, qp_small00, qp_small01, qp_small10, qp_small11 = d.p_tmp, d.XtX_p, d.Xtz_p, d.XtX_p_T2, d.Xtz_p_T2, d.qp_small00, d.qp_small01, d.qp_small10, d.qp_small11
    q_T2, p_T2 = d.q_T2, d.p_T2
    p_next = update2 ? d.p_next2 : d.p_next
    q      = update2 ? d.q_next2 : d.q_next
    p      = update2 ? d.p_next  : d.p
    pv     = update2 ? d.p_nextv : d.pv
    pdiffv = d.p_tmpv
    XtXv   = d.XtX_pv
    Xtzv   = d.Xtz_pv

    # qf!(qp, q, f)
    # ll_prev = loglikelihood_full(d, g, q, p) # loglikelihood(g, qf)
    # println(d.ll_prev)
    @time if d_cu === nothing # CPU operation
        fill!(XtX_T2, zero(T2))
        fill!(Xtz_T2, zero(T2))
        # update_p_loop!(XtX_T2, Xtz_T2, g, q, p, qp_small00, qp_small01, qp_small10, qp_small11, 1:I, 1:J, K)
        threader!(update_p_loop!, 
            eltype(XtX), (XtX_T2, Xtz_T2, g, q, p, qp_small00, qp_small01, qp_small10, qp_small11), 1:I, 1:J, K, false; maxL=16)
    else # GPU operation
        q_T2 .= q
        p_T2 .= p
        OpenADMIXTURE.copyto_sync!([d_cu.q, d_cu.p], [q_T2, p_T2])
        update_p_cuda!(d_cu, g_cu)
        OpenADMIXTURE.copyto_sync!([XtX_T2, Xtz_T2], [d_cu.XtX_p, d_cu.Xtz_p])
    end

    XtX .= XtX_T2
    Xtz .= Xtz_T2

    # Solve the quadratic programs
    @time begin
        rho = convert(T, rho)
        Xtz .*= -1 
        pmin = zeros(T, 4K) 
        pmax = ones(T, 4K)
        @batch threadlocal=QPThreadLocal{T}(K) for j in 1:J
            # even the views are allocating something, so we use preallocated views.
            XtX_ = XtXv[j]
            Xtz_ = Xtzv[j]
            p_ = pv[j]
            pdiff_ = pdiffv[j]

            # t = threadid()
            tmp_XtX_full = threadlocal.tmp_XtX_p

            # copy block-diagonal hessians. 
            fill!(tmp_XtX_full, zero(T))
            for l in 1:4
                for k in 1:K
                    for k2 in 1:K
                        tmp_XtX_full[(l-1) * K + k2, (l-1) * K + k] = 
                            XtX_[(k-1) * K + k2, l]
                    end
                end
            end

            tableau_5k1 = threadlocal.tableau_5k1
            tableau_4k1 = threadlocal.tableau_4k1
            tmp_k = threadlocal.tmp_k
            tmp_4k1 = threadlocal.tmp_4k1
            tmp_4k1_ = threadlocal.tmp_4k1_
            tmp_5k1 = threadlocal.tmp_5k1
            tmp_5k1_ = threadlocal.tmp_5k1_
            swept = threadlocal.swept_4k
            
            # verbose = false
            if penalty
                for l2 in 1:4
                    for l in 1:4
                        for k in 1:K
                            tmp_XtX_full[(l-1) * K + k, (l2-1) * K + k] += rho
                        end
                    end
                end
                OpenADMIXTURE.create_tableau!(tableau_4k1, tmp_XtX_full, Xtz_, p_, d.v_4k4k, tmp_k, false)
                OpenADMIXTURE.quadratic_program!(pdiff_, tableau_4k1, p_, pmin, pmax, 4K, 0,
                    tmp_4k1, tmp_4k1_, swept)
            else
                OpenADMIXTURE.create_tableau!(tableau_5k1, tmp_XtX_full, Xtz_, p_, d.v_4k4k, tmp_k, false, true; 
                    tmp_4k_k=threadlocal.tmp_4k_k, tmp_4k_k_2=threadlocal.tmp_4k_k_2)#, verbose=verbose)
                OpenADMIXTURE.quadratic_program!(pdiff_, tableau_5k1, p_, pmin, pmax, 4K, K, 
                    tmp_5k1, tmp_5k1_, swept)#; verbose=verbose)
            end
        end
        factor = one(T)
        if guarantee_increase
            while True
                @inbounds for j in 1:4J
                    for k in 1:K
                        p_next[k, j] = p[k, j] + factor * pdiff[k, j]
                    end
                end
                project_p!(p_next, d.idx4v[1], K)
                ll_new = if d_cu !== nothing
                    q_T2 .= q
                    p_T2 .= p_next
                    OpenADMIXTURE.copyto_sync!([d_cu.p, d_cu.q], [d.p_T2, d.q_T2])
                    loglikelihood(d_cu, g_cu, d_cu.q, d_cu.p)
                else
                    loglikelihood_full2(d, g, d.q_next2, d.p_next2)
                end
                if ll_new > d.ll_tmp 
                    d.ll_tmp = ll_new
                    break
                end
                @info "Step size halved in updated_p. Current step size: $factor"
                factor /= 2
            end
        else
            @inbounds for j in 1:4J
                for k in 1:K
                    p_next[k, j] = p[k, j] + pdiff[k, j]
                end
            end
            project_p!(p_next, d.idx4v[1], K)
        end
    end
    # ll_new = loglikelihood_full(d, g, q, p_next)#; q_=q, p_=p)
    # @info "p_update: ll_new=$ll_new, ll_prev=$ll_prev $(ll_new > ll_prev)"
end
