struct CuAdmixData{T}
    # g::CuArray{UInt8, 2}
    I::Int
    J::Int
    K::Int
    q::CuArray{T, 2}
    q_next::CuArray{T, 2}
    p::CuArray{T, 2}
    p_next::CuArray{T, 2}
    XtX_q::CuArray{T, 3}
    Xtz_q::CuArray{T, 2}
    XtX_p::CuArray{T, 3}
    Xtz_p::CuArray{T, 2}
    doublemissing::CuArray{T, 2}
end
function CuAdmixData(d::AdmixData2{T, T2}, g::SnpLinAlg{T2}, width=d.J) where {T, T2}
    I, J, K = d.I, d.J, d.K
    Ibytes = (I + 3) ÷ 4
    q = CuArray{T2, 2}(undef, K, I)
    q_next = similar(q)
    p = CuArray{T2, 2}(undef, K, 4J)
    p_next = similar(p)
    doublemissing = CuArray{T2, 2}(undef, 1, I)
    OpenADMIXTURE.copyto_sync!([q, q_next, p, p_next], 
        [d.q_T2, d.q_T2, d.p_T2, d.p_T2])
    OpenADMIXTURE.copyto_sync!([doublemissing], [convert(Array{T2}, d.doublemissing)])
    XtX_q = CuArray{T2, 3}(undef, K, K, I)
    Xtz_q = CuArray{T2, 2}(undef, K, I)
    XtX_p = CuArray{T2, 3}(undef, K, K, 4J)
    Xtz_p = CuArray{T2, 2}(undef, K, 4J)
    CuAdmixData{T2}(I, J, K, q, q_next, p, p_next, XtX_q, Xtz_q, XtX_p, Xtz_p, doublemissing)
end

function _cu_admixture_base(d::AdmixData2{T, T2}, g_la::SnpLinAlg{T2}, I::Int, J::Int) where {T, T2}
    d_cu = CuAdmixData(d, g_la)
    Ibytes = (I + 3) ÷ 4
    g_cu = CuArray{UInt8, 2}(undef, Ibytes, 2J)
    if 2J == size(g_la, 2)
        unsafe_copyto!(pointer(g_cu), pointer(g_la.s.data), Ibytes * 2J)
    else
        copyto!(g_cu, @view(g_la.s.data[:, 1:2J]))
    end
    d_cu, g_cu
end
