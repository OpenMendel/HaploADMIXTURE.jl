function init_em!(d::AdmixData2{T}, g::AbstractArray{T}, iter::Integer) where T
    count_double_missing!(d.doublemissing, g)
    for i in 1:iter
        em!(d, g)
        d.p .= d.p_next
        d.q .= d.q_next
        d.ll_new = loglikelihood_full2(d, g, d.q, d.p)
        # ll_test = loglikelihood_full2(d, g, d.q, d.p)
        @info "EM iter $i, ll: $(d.ll_new)"
    end
end

"""
    admixture_qn!(d, g, iter=1000, rtol=1e-7; d_cu=nothing, g_cu=nothing, 
        mode=:ZAL, iter_count_offset=0)
Initialize P and Q with the FRAPPE EM algorithm

# Input
- `d`: an `AdmixData`.
- `g`: a genotype matrix.
- `iter`: number of iterations.
- `rtol`: convergence tolerance in terms of relative change of loglikelihood.
- `d_cu`: a `CuAdmixData` if using GPU, `nothing` otherwise.
- `g_cu`: a `CuMatrix{UInt8}` corresponding to the data part of genotype matrix
- `mode`: `:ZAL` for Zhou-Alexander-Lange acceleration (2009), `:LBQN` for Agarwal-Xu (2020). 
"""
function admixture_qn!(d::AdmixData2{T}, g::AbstractArray{T}, iter::Int=1000, 
    rtol= 1e-7; d_cu=nothing, g_cu=nothing, mode=:ZAL, iter_count_offset=0) where T
    # qf!(d.qf, d.q, d.f)
    # ll_prev = loglikelihood(g, d.q, d.f, d.qp_small, d.K, d.skipmissing)
    # d.ll_new = ll_prev
    
    if isnan(d.ll_new)
        if d_cu !== nothing
            # copyto_sync!([d_cu.p, d_cu.q], [d.p, d.q])
            # d.ll_new = loglikelihood(d_cu, g_cu)
        else
            d.ll_new = loglikelihood_full2(d, g, d.q, d.p)
        end
    end

    println("initial ll: ", d.ll_new)
    for i in (iter_count_offset + 1):iter
        @time begin
            # qf!(d.qf, d.q, d.f)
            # ll_prev = loglikelihood(g, d.qf)
            d.ll_prev = d.ll_new
            update_q!(d, g)#; d_cu=d_cu, g_cu=g_cu)
            update_p!(d, g)#; d_cu=d_cu, g_cu=g_cu)
            update_q!(d, g, true)#; d_cu=d_cu, g_cu=g_cu)
            update_p!(d, g, true)#; d_cu=d_cu, g_cu=g_cu)

            # qf!(d.qf, d.q_next2, d.f_next2)
            ll_basic = if d_cu !== nothing
                #copyto_sync!([d_cu.p, d_cu.q], [d.p_next2, d.q_next2])
                #loglikelihood(d_cu, g_cu)
            else
                loglikelihood_full2(d, g, d.q_next2, d.p_next2)
            end
            
            if mode == :ZAL
                OpenADMIXTURE.update_UV!(d.U, d.V, d.x_flat, d.x_next_flat, d.x_next2_flat, i, d.Q)
                U_part = i < d.Q ? view(d.U, :, 1:i) : view(d.U, :, :)
                V_part = i < d.Q ? view(d.V, :, 1:i) : view(d.V, :, :)

                OpenADMIXTURE.update_QN!(d.x_tmp_flat, d.x_next_flat, d.x_flat, U_part, V_part)
            elseif mode == :LBQN
                OpenADMIXTURE.update_UV_LBQN!(d.U, d.V, d.x_flat, d.x_next_flat, d.x_next2_flat, i, d.Q)
                U_part = i < d.Q ? view(d.U, :, 1:i) : view(d.U, :, :)
                V_part = i < d.Q ? view(d.V, :, 1:i) : view(d.V, :, :)
                OpenADMIXTURE.update_QN_LBQN!(d.x_tmp_flat, d.x_flat, d.x_qq, d.x_rr, U_part, V_part)
            else 
                @assert false "Invalid mode"
            end

            project_p!(d.p_tmp, d.idx4v[1], d.K)
            OpenADMIXTURE.project_q!(d.q_tmp, d.idxv[1])
            # qf!(d.qf, d.q_tmp, d.f_tmp)
            ll_qn = if d_cu !== nothing # GPU mode
                # copyto_sync!([d_cu.p, d_cu.q], [d.p_tmp, d.q_tmp])
                # loglikelihood(d_cu, g_cu)
            else # CPU mode
                loglikelihood_full2(d, g, d.q_tmp, d.p_tmp)
            end
            if d.ll_prev < ll_qn
                d.x .= d.x_tmp
                d.ll_new = ll_qn
            else
                d.x .= d.x_next2
                d.ll_new = ll_basic
            end
            println(ll_basic)
            println(ll_qn)
            reldiff = abs((d.ll_new - d.ll_prev) / d.ll_prev)
            @info "Iteration $i: ll=$(d.ll_new), reldiff = $reldiff, ll_basic=$ll_basic, ll_qn=$ll_qn"
            if reldiff < rtol
                break
            end
        end
        println()
        println()
    end
end
