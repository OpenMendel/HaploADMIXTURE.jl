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
function CuAdmixData(d::AdmixData2{T}, g::SnpLinAlg{T}, width=d.J) where T
    I, J, K = d.I, d.J, d.K
    Ibytes = (I + 3) ÷ 4
    q = CuArray{T, 2}(undef, K, I)
    q_next = similar(q)
    p = CuArray{T, 2}(undef, K, 4J)
    p_next = similar(p)
    doublemissing = CuArray{T, 2}(undef, 1, I)
    OpenADMIXTURE.copyto_sync!([q, q_next, p, p_next], 
        [d.q, d.q_next, d.p, d.p_next])
    OpenADMIXTURE.copyto_sync!([doublemissing], [d.doublemissing])
    XtX_q = CuArray{T, 3}(undef, K, K, I)
    Xtz_q = CuArray{T, 2}(undef, K, I)
    XtX_p = CuArray{T, 3}(undef, K, K, 4J)
    Xtz_p = CuArray{T, 2}(undef, K, 4J)
    CuAdmixData{T}(I, J, K, q, q_next, p, p_next, XtX_q, Xtz_q, XtX_p, Xtz_p, doublemissing)
end

function _cu_admixture_base(d::AdmixData2, g_la::SnpLinAlg, I::Int, J::Int)
    d_cu = CuAdmixData(d, g_la)
    Ibytes = (I + 3) ÷ 4
    g_cu = CuArray{UInt8, 2}(undef, Ibytes, 2J)
    copyto!(g_cu, @view(g_la.s.data[:, 1:2J]))
    d_cu, g_cu
end
