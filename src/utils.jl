function count_double_missing!(doublemissing, g)
    I = size(g, 1)
    J = convert(Int, ceil(size(g, 2)/2))
    fill!(doublemissing, zero(Int))
    @inbounds for j in 1:J
        for i in 1:I
            if g[i, 2(j-1)+1] == 3 && g[i, 2j] == 3
                doublemissing[i] += 1
            end
        end
    end
    doublemissing
end

function project_p!(p::AbstractMatrix{T}, idx::AbstractVector{Int}, K; pseudocount=T(1e-5)) where T
    J = size(p, 2) ÷ 4
    @inbounds for j in 1:J
        for k in 1:K
            OpenADMIXTURE.project_q!(@view(p[k, 4(j-1)+1:4j]), idx)
        end
    end
    p
end
