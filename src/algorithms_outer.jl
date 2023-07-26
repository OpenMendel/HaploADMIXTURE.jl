function init_em!(d::AdmixData2{T, T2}, g::AbstractArray{T2}, iter::Integer; 
    d_cu=nothing, g_cu=nothing,
    fix_q=false, fix_p=false) where {T, T2}
    for i in 1:iter
        em!(d, g; d_cu=d_cu, g_cu=g_cu, fix_q=fix_q, fix_p=fix_p)
        d.p .= d.p_next
        d.q .= d.q_next
        if d_cu !== nothing
            d_cu.p .= d_cu.p_next
            d_cu.q .= d_cu.q_next
            d.ll_new = loglikelihood(d_cu, g_cu, d_cu.q, d_cu.p)
        else
            d.ll_new = loglikelihood_full2(d, g, d.q, d.p)
        end
        project_p!(d.p, d.idx4v[1], d.K)
        OpenADMIXTURE.project_q!(d.q, d.idxv[1])
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
function admixture_qn!(d::AdmixData2{T, T2}, g::AbstractArray{T2}, iter::Int=1000, 
    rtol= 1e-7; d_cu=nothing, g_cu=nothing, mode=:ZAL, iter_count_offset=0, penalty=nothing, 
    fix_q=false, fix_p=false, em_p=false, em_p_iter=10) where {T, T2}
    # qf!(d.qf, d.q, d.f)
    # ll_prev = loglikelihood(g, d.q, d.f, d.qp_small, d.K, d.skipmissing)
    # d.ll_new = ll_prev
    
    if isnan(d.ll_new)
        if d_cu !== nothing
            d.q_T2 .= d.q
            d.p_T2 .= d.p
            OpenADMIXTURE.copyto_sync!([d_cu.q, d_cu.p], [d.q_T2, d.p_T2])
            d.ll_new = loglikelihood(d_cu, g_cu, d_cu.q, d_cu.p)
        else
            d.ll_new = loglikelihood_full2(d, g, d.q, d.p)
        end
    end

    println("initial ll: ", d.ll_new)
    if penalty === nothing
        p_penalty = true
        q_penalty = false
    else
        p_penalty = penalty
        q_penalty = penalty
    end
    for i in (iter_count_offset + 1):iter
        @time begin
            # qf!(d.qf, d.q, d.f)
            # ll_prev = loglikelihood(g, d.qf)
            d.ll_prev = d.ll_new
            if !fix_q
                update_q!(d, g; d_cu=d_cu, g_cu=g_cu, penalty=q_penalty)
            else
                d.q_next .= d.q
            end
            if !fix_p
                if !em_p
                    update_p!(d, g; d_cu=d_cu, g_cu=g_cu, penalty=p_penalty)
                else
                    for ii in 1:em_p_iter
                        em!(d, g; d_cu=d_cu, g_cu=g_cu, fix_q=true, fix_p=false)
                        d.p_next .= d.p
                        em!(d, g; d_cu=d_cu, g_cu=g_cu, fix_q=true, fix_p=false)
                        d.p_tmp .= d.p
                        OpenADMIXTURE.update_UV!(d.U_p, d.V_p, d.p_flat, d.p_next_flat, d.p_tmp_flat, ii, d.Q)
                        U_part = ii < d.Q ? view(d.U_p, :, 1:i) : view(d.U_p, :, :)
                        V_part = ii < d.Q ? view(d.V_p, :, 1:i) : view(d.V_p, :, :)
                        OpenADMIXTURE.update_QN!(d.p_tmp_flat, d.p_next_flat, d.p_flat, U_part, V_part)
                    end
                end
            else
                d.p_next .= d.p
            end
            if !fix_q
                update_q!(d, g, true; d_cu=d_cu, g_cu=g_cu, penalty=q_penalty)
            else
                d.q_next2 .= d.q
            end
            if !fix_p
                if !em_p
                    update_p!(d, g, true; d_cu=d_cu, g_cu=g_cu, penalty=p_penalty)
                else
                    for ii in 1:em_p_iter
                        em!(d, g; d_cu=d_cu, g_cu=g_cu, fix_q=true, fix_p=false)
                        d.p_next2 .= d.p
                        em!(d, g; d_cu=d_cu, g_cu=g_cu, fix_q=true, fix_p=false)
                        d.p_tmp .= d.p
                        OpenADMIXTURE.update_UV!(d.U_p, d.V_p, d.p_flat, d.p_next2_flat, d.p_tmp_flat, ii, d.Q)
                        U_part = ii < d.Q ? view(d.U_p, :, 1:i) : view(d.U_p, :, :)
                        V_part = ii < d.Q ? view(d.V_p, :, 1:i) : view(d.V_p, :, :)
                        OpenADMIXTURE.update_QN!(d.p_tmp_flat, d.p_next2_flat, d.p_flat, U_part, V_part)
                    end
                end
            else
                d.p_next2 .= d.p
            end

            # qf!(d.qf, d.q_next2, d.f_next2)
            ll_basic = if d_cu !== nothing
                d.q_T2 .= d.q_next2
                d.p_T2 .= d.p_next2
                OpenADMIXTURE.copyto_sync!([d_cu.p, d_cu.q], [d.p_T2, d.q_T2])
                loglikelihood(d_cu, g_cu, d_cu.q, d_cu.p)
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
                d.q_T2 .= d.q_tmp
                d.p_T2 .= d.p_tmp
                OpenADMIXTURE.copyto_sync!([d_cu.q, d_cu.p], [d.q_T2, d.p_T2])
                loglikelihood(d_cu, g_cu, d_cu.q, d_cu.p)
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
            reldiff = abs((d.ll_new - d.ll_prev) / d.ll_prev)
            @info "Iteration $i: ll=$(d.ll_new), reldiff = $reldiff, ll_basic=$ll_basic, ll_qn=$ll_qn"
            if reldiff < rtol
                break
            end
            flush(stdout)
            flush(stderr)
        end
        println()
        println()
    end
end
