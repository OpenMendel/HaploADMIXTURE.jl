# const SAT{T} = SubArray{T, 2, Matrix{T}, 
#     Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
const TwoDSlice{T} = Vector{SubArray{T, 2, Array{T, 3}, 
    Tuple{Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
const OneDSlice{T} = Vector{SubArray{T, 1, Matrix{T}, 
    Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}
const TILE = Ref(512) # this is now a length, in bytes!
function tile_maxiter(::Type{<:AbstractArray{T}}) where {T}
    isbitstype(T) || return TILE[] ÷ 8
    max(TILE[] ÷ sizeof(T), 4)
end
tile_maxiter(::Type{AT}) where {AT} = TILE[] ÷ 8 # treat anything unkown like Float64

mutable struct AdmixData2{T,T2}
    I           ::Int
    J           ::Int
    K           ::Int
    Q           ::Int
    # Q: K x I, P: K x J x 4. 

    x           ::Matrix{T} # K x (I + 4J)
    x_next      ::Matrix{T}
    x_next2     ::Matrix{T}
    x_tmp       ::Matrix{T}

    x_flat      ::Vector{T} # K(I + 4J)
    x_next_flat ::Vector{T}
    x_next2_flat::Vector{T}
    x_tmp_flat  ::Vector{T}

    # intermediate vectors for LBQN
    x_qq        ::Vector{T}
    x_rr        ::Vector{T}

    doublemissing::Matrix{T}
    q           ::Matrix{T} # K x I
    q_next      ::Matrix{T}
    q_next2     ::Matrix{T}
    q_tmp       ::Matrix{T}
    q_T2        ::Matrix{T2}

    p           ::Matrix{T} # K x 4J
    p_next      ::Matrix{T}
    p_next2     ::Matrix{T}
    p_tmp       ::Matrix{T}
    p_T2        ::Matrix{T2}

    XtX_q       ::Array{T, 3} # K x K x I
    Xtz_q       ::Matrix{T}   # K x I
    XtX_p       ::Array{T, 3} # K x K x 4J
    Xtz_p       ::Matrix{T}   # K x 4J

    XtX_q_T2       ::Array{T2, 3} # K x K x I
    Xtz_q_T2       ::Matrix{T2}   # K x I
    XtX_p_T2       ::Array{T2, 3} # K x K x 4J
    Xtz_p_T2       ::Matrix{T2}   # K x 4J

    # views
    qv          :: OneDSlice{T}
    q_nextv     :: OneDSlice{T}
    q_tmpv      :: OneDSlice{T}
    pv          :: OneDSlice{T}
    p_nextv     :: OneDSlice{T}
    p_tmpv      :: OneDSlice{T}

    XtX_qv      :: TwoDSlice{T}
    Xtz_qv      :: OneDSlice{T}
    XtX_pv      :: TwoDSlice{T}
    Xtz_pv      :: OneDSlice{T}

    qp_small00    :: Array{T2, 3}   # 64 x 64
    qp_smallv00   :: TwoDSlice{T2}
    qp_small01    :: Array{T2, 3}   # 64 x 64
    qp_smallv01   :: TwoDSlice{T2}
    qp_small10    :: Array{T2, 3}   # 64 x 64
    qp_smallv10   :: TwoDSlice{T2}
    qp_small11    :: Array{T2, 3}   # 64 x 64
    qp_smallv11   :: TwoDSlice{T2}

    qp_small00_    :: Array{T2, 3}   # 64 x 64
    qp_smallv00_   :: TwoDSlice{T2}
    qp_small01_    :: Array{T2, 3}   # 64 x 64
    qp_smallv01_   :: TwoDSlice{T2}
    qp_small10_    :: Array{T2, 3}   # 64 x 64
    qp_smallv10_   :: TwoDSlice{T2}
    qp_small11_    :: Array{T2, 3}   # 64 x 64
    qp_smallv11_   :: TwoDSlice{T2}

    U           ::Matrix{T}   # K(I + 4J) x Q
    V           ::Matrix{T}   # K(I + 4J) x Q

    # for QP
    v_kk        ::Matrix{T}   # K x K, a full svd of ones(1, K)
    v_4k4k      ::Matrix{T}   # 4K x 4K, a full svd of [I(K) I(K) I(K) I(K)]

    tmp_k       ::Matrix{T}
    # tmp_k1      ::Matrix{T}
    # tmp_k1_      ::Matrix{T}
    tmp_k2      ::Matrix{T}
    tmp_k2_      ::Matrix{T}
    tmp_XtX_p    ::Array{T, 3}
    tmp_4k_k     ::Array{T, 3}
    tmp_4k_k_2   ::Array{T, 3}
    tmp_5k1      ::Matrix{T}
    tmp_5k1_     ::Matrix{T}

    # tableau_k1  ::Array{T, 3}   # (K + 1) x (K + 1)
    tableau_k2  ::Array{T, 3}   # (K + 2) x (K + 2)
    tableau_5k1 ::Array{T, 3} # (5K+1) x (5K+1)
    swept       ::Matrix{Bool}
    swept_4k    ::Matrix{Bool}

    tmp_kv       ::OneDSlice{T}
    # tmp_k1v      ::OneDSlice{T}
    # tmp_k1_v      ::OneDSlice{T}
    tmp_k2v      ::OneDSlice{T}
    tmp_k2_v      ::OneDSlice{T}
    # tableau_k1v  ::TwoDSlice{T}   # (K + 1) x (K + 1)
    tmp_XtX_pv   ::TwoDSlice{T}
    tmp_4k_kv    ::TwoDSlice{T}
    tmp_4k_k_2v  ::TwoDSlice{T}
    tmp_5k1v      ::OneDSlice{T}
    tmp_5k1_v     ::OneDSlice{T}

    tableau_k2v  ::TwoDSlice{T}   # (K + 2) x (K + 2)
    tableau_5k1v ::TwoDSlice{T}
    sweptv       ::OneDSlice{Bool}
    swept_4kv    ::OneDSlice{Bool}

    idx         ::Matrix{Int}
    idxv        ::OneDSlice{Int}
    idx4        ::Matrix{Int}
    idx4v       ::OneDSlice{Int}

    # loglikelihoods
    ll_prev     ::Float64
    ll_new      ::Float64
end

"""
    AdmixData2{T}(I, J, K, Q; skipmissing=true, rng=Random.GLOBAL_RNG)
Constructor for Admixture information.

# Arguments:
- I: Number of samples
- J: Number of variants
- K: Number of clusters
- Q: Number of steps used for quasi-Newton update
- skipmissing: skip computation of loglikelihood for missing values. Should be kept `true` in most cases
- rng: Random number generation.
"""
function AdmixData2{T,T2}(I, J, K, Q, g; rng=Random.GLOBAL_RNG) where {T, T2}
    NT = nthreads()
    x = convert(Array{T}, rand(rng, K, I + 4J))
    x_next = similar(x)
    x_next2 = similar(x)
    x_tmp = similar(x)

    x_flat = reshape(x, :)
    x_next_flat = reshape(x_next, :)
    x_next2_flat = reshape(x_next2, :)
    x_tmp_flat = reshape(x_tmp, :)

    x_qq = similar(x_flat)
    x_rr = similar(x_flat)

    doublemissing = zeros(Int, 1, I)
    count_double_missing!(doublemissing, g)
    q       = view(x      , :, 1:I)#rand(T, K, J) 
    q       = unsafe_wrap(Array, pointer(q), size(q))
    q_next  = view(x_next , :, 1:I)
    q_next  = unsafe_wrap(Array, pointer(q_next), size(q_next))
    q_next2 = view(x_next2, :, 1:I)
    q_next2  = unsafe_wrap(Array, pointer(q_next2), size(q_next2))
    q_tmp   = view(x_tmp, :, 1:I)
    q_tmp  = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))
    q ./= sum(q, dims=1)
    q_T2 = convert(Array{T2}, q)

    p       = view(x      , :, (I+1):(I+4J))#rand(T, K, J) 
    p       = unsafe_wrap(Array, pointer(p), size(p))
    p_next  = view(x_next , :, (I+1):(I+4J))
    p_next  = unsafe_wrap(Array, pointer(p_next), size(p_next))
    p_next2 = view(x_next2, :, (I+1):(I+4J))
    p_next2 = unsafe_wrap(Array, pointer(p_next2), size(p_next2))
    p_tmp   = view(x_tmp, :, (I+1):(I+4J))
    p_tmp   = unsafe_wrap(Array, pointer(p_tmp), size(p_tmp))
    for j in 1:J
        for k in 1:K
            s = zero(T)
            for l in 1:4
                s += p[k, 4(j-1)+l]
            end
            for l in 1:4
                p[k, 4(j-1)+l] = p[k, 4(j-1)+l] / s
            end
        end
    end
    p_T2 = convert(Array{T2}, p)

    XtX_q = convert(Array{T}, rand(rng, K, K, I))
    Xtz_q = convert(Array{T}, rand(rng, K, I))

    XtX_p = convert(Array{T}, rand(rng, K, K, 4J))
    Xtz_p = convert(Array{T}, rand(rng,  K, 4J))

    XtX_q_T2 = convert(Array{T2}, XtX_q)
    Xtz_q_T2 = convert(Array{T2}, Xtz_q)

    XtX_p_T2 = convert(Array{T2}, XtX_p)
    Xtz_p_T2 = convert(Array{T2}, Xtz_p)

    qv          = [view(q, :, i) for i in 1:I]
    q_nextv     = [view(q_next, :, i) for i in 1:I]
    q_tmpv     = [view(q_tmp, :, i) for i in 1:I]

    pv          = [view(reshape(p, 4K, :), :, j) for j in 1:J]
    p_nextv     = [view(reshape(p_next, 4K, :), :, j) for j in 1:J]
    p_tmpv     = [view(reshape(p_tmp, 4K, :), :, j) for j in 1:J]

    XtX_qv      = [view(XtX_q, :, :, i) for i in 1:I] # K x K.
    Xtz_qv      = [view(Xtz_q, :, i) for i in 1:I]

    XtX_pv      = [view(reshape(XtX_p, K * K, 4, J), :, :, j) for j in 1:J] # K^2 x 4. 
    Xtz_pv      = [view(reshape(Xtz_p, 4K, :), :, j) for j in 1:J]

    # q, 
    # q_arr = unsafe_wrap(Array, pointer(q), size(q))
    # q_next_arr = unsafe_wrap(Array, pointer(q_next), size(q_next))
    # q_tmp_arr = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))

    # f_arr = unsafe_wrap(Array, pointer(q), size(q))
    # f_next_arr = unsafe_wrap(Array, pointer(q_next), size(q_next))
    # f_tmp_arr = unsafe_wrap(Array, pointer(q_tmp), size(q_tmp))


    # qf = rand(T, I, J);
    # maxL = tile_maxiter(typeof(Xtz_p))
    # qp_small00 = convert(Array{T2}, rand(rng, I, J, NT))
    maxL = OpenADMIXTURE.tile_maxiter(typeof(Xtz_p))
    qp_small00 = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv00 = [view(qp_small00, :, :, t) for t in 1:NT]

    # qp_small01 = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small01 = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv01 = [view(qp_small01, :, :, t) for t in 1:NT]

    # qp_small10 = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small10 = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv10 = [view(qp_small10, :, :, t) for t in 1:NT]

    # qp_small11 = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small11 = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv11 = [view(qp_small11, :, :, t) for t in 1:NT]


    # qp_small00_ = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small00_ = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv00_ = [view(qp_small00, :, :, t) for t in 1:NT]

    # qp_small01_ = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small01_ = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv01_ = [view(qp_small01, :, :, t) for t in 1:NT]

    # qp_small10_ = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small10_ = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv10_ = [view(qp_small10, :, :, t) for t in 1:NT]

    # qp_small11_ = convert(Array{T2}, rand(rng, I, J, NT))
    qp_small11_ = convert(Array{T2}, rand(rng, maxL, maxL, NT))
    qp_smallv11_ = [view(qp_small11, :, :, t) for t in 1:NT]
    # qf_thin  = rand(T, I, maxL)
    # f_tmp = similar(f)
    # q_tmp = similar(q);

    V = convert(Array{T}, rand(rng, K * (I+4J), Q))
    # V_q = view(reshape(V, K, (I+J), Q), :, 1:I, :)
    # V_f = view(reshape(V, K, (I+J), Q), :, (I+1):(I+J), :)
    U = convert(Array{T}, rand(rng, K * (I+4J), Q))

# TODO: Determine temporary storages for QP. Examine Search.jl code.
    _, _, v_kk = svd(ones(T, 1, K), full=true)
    IK = collect(LinearAlgebra.I(K))
    _, _, v_4k4k = svd([IK IK IK IK], full=true)
    tmp_k = Matrix{T}(undef, K, NT)
    tmp_kv = [view(tmp_k, :, t) for t in 1:NT]
    # tmp_k1 = Matrix{T}(undef, K+1, NT)
    # tmp_k1v = [view(tmp_k1, :, t) for t in 1:NT]
    # tmp_k1_ = similar(tmp_k1)
    # tmp_k1_v = [view(tmp_k1_, :, t) for t in 1:NT]
    tmp_k2 = Matrix{T}(undef, K+2, NT)
    tmp_k2v = [view(tmp_k2, :, t) for t in 1:NT]
    tmp_k2_ = similar(tmp_k2)
    tmp_k2_v = [view(tmp_k2_, :, t) for t in 1:NT]
    # tableau_k1 = Array{T, 3}(undef, K+1, K+1, NT)
    # tableau_k1v = [view(tableau_k1, :, :, t) for t in 1:NT]

    tmp_XtX_p = Array{T, 3}(undef, 4K, 4K, NT)
    tmp_XtX_pv = [view(tmp_XtX_p, :, :, t) for t in 1:NT]

    tmp_4k_k = Array{T, 3}(undef, 4K, K, NT)
    tmp_4k_kv = [view(tmp_4k_k, :, :, t) for t in 1:NT]
    tmp_4k_k_2 = Array{T, 3}(undef, 4K, K, NT)
    tmp_4k_k_2v = [view(tmp_4k_k_2, :, :, t) for t in 1:NT]

    tmp_5k1 = Matrix{T}(undef, 5K+1, NT)
    tmp_5k1v = [view(tmp_5k1, :, t) for t in 1:NT]
    tmp_5k1_ = similar(tmp_5k1)
    tmp_5k1_v = [view(tmp_5k1_, :, t) for t in 1:NT]


    tableau_k2 = Array{T, 3}(undef, K+2, K+2, NT)
    tableau_k2v = [view(tableau_k2, :, :, t) for t in 1:NT]
    tableau_5k1 = Array{T, 3}(undef, 5K+1, 5K+1, NT)
    tableau_5k1v = [view(tableau_5k1, :, :, t) for t in 1:NT]
    swept = convert(Matrix{Bool}, trues(K, NT))
    sweptv = [view(swept, :, t) for t in 1:NT]
    swept_4k = convert(Matrix{Bool}, trues(4K, NT))
    swept_4kv = [view(swept_4k, :, t) for t in 1:NT]
    idx = Array{Int}(undef, K, NT)
    idxv = [view(idx, :, t) for t in 1:NT]
    idx4 = Array{Int}(undef, 4, NT)
    idx4v = [view(idx4, :, t) for t in 1:NT]

    AdmixData2{T, T2}(I, J, K, Q, 
        x, x_next, x_next2, x_tmp, 
        x_flat, x_next_flat, x_next2_flat, x_tmp_flat,
        x_qq, x_rr,
        doublemissing,
        q, q_next, q_next2, q_tmp, q_T2,
        p, p_next, p_next2, p_tmp, p_T2,
        XtX_q, Xtz_q, XtX_p, Xtz_p,
        XtX_q_T2, Xtz_q_T2, XtX_p_T2, Xtz_p_T2, 
        qv, q_nextv, q_tmpv, pv, p_nextv, p_tmpv, 
        XtX_qv, Xtz_qv, XtX_pv, Xtz_pv,
        qp_small00, qp_smallv00,
        qp_small01, qp_smallv01,
        qp_small10, qp_smallv10,
        qp_small11, qp_smallv11, 
        qp_small00_, qp_smallv00_,
        qp_small01_, qp_smallv01_,
        qp_small10_, qp_smallv10_,
        qp_small11_, qp_smallv11_, 
        U, V, 
        v_kk, v_4k4k,
        tmp_k, tmp_k2, tmp_k2_, tmp_XtX_p,
        tmp_4k_k, tmp_4k_k_2,
        tmp_5k1, tmp_5k1_,
        tableau_k2, tableau_5k1, 
        swept, swept_4k,
        tmp_kv, tmp_k2v, tmp_k2_v, tmp_XtX_pv,
        tmp_4k_kv, tmp_4k_k_2v,
        tmp_5k1v, tmp_5k1_v,
        tableau_k2v, tableau_5k1v, 
        sweptv, swept_4kv,
        idx, idxv,
        idx4, idx4v,
        NaN, NaN)
end

struct QPThreadLocal{T}
    tmp_k   :: Vector{T}
    tmp_k1  :: Vector{T}
    tmp_k1_ :: Vector{T}
    tmp_k2  :: Vector{T}
    tmp_k2_ :: Vector{T}
    tmp_XtX_p :: Matrix{T}
    tmp_4k_k :: Matrix{T}
    tmp_4k_k_2 :: Matrix{T}
    tmp_4k1 :: Vector{T}
    tmp_4k1_ :: Vector{T}
    tmp_5k1 :: Vector{T}
    tmp_5k1_ :: Vector{T}
    tableau_k1 :: Matrix{T}
    tableau_k2 :: Matrix{T}
    tableau_4k1 :: Matrix{T}
    tableau_5k1 :: Matrix{T}
    swept :: Vector{Bool}
    swept_4k :: Vector{Bool}
    idx :: Vector{Int}
    idx4 :: Vector{Int}
end
function QPThreadLocal{T}(K::Int) where T
    tmp_k = Vector{T}(undef, K)
    tmp_k1 = Vector{T}(undef, K+1)
    tmp_k1_ = Vector{T}(undef, K+1)
    tmp_k2 = Vector{T}(undef, K+2)
    tmp_k2_ = similar(tmp_k2)
    tmp_XtX_p = Array{T, 2}(undef, 4K, 4K)
    tmp_4k_k = Array{T, 2}(undef, 4K, K)
    tmp_4k_k_2 = Array{T, 2}(undef, 4K, K)
    tmp_4k1 = Vector{T}(undef, 4K+1)
    tmp_4k1_ = similar(tmp_4k1) 
    tmp_5k1 = Vector{T}(undef, 5K+1)
    tmp_5k1_ = similar(tmp_5k1)
    tableau_k1 = Array{T, 2}(undef, K+1, K+1)
    tableau_k2 = Array{T, 2}(undef, K+2, K+2)
    tableau_4k1 = Array{T, 2}(undef, 4K+1, 4K+1)
    tableau_5k1 = Array{T, 2}(undef, 5K+1, 5K+1)
    swept = convert(Vector{Bool}, trues(K))
    swept_4k = convert(Vector{Bool}, trues(4K))
    idx = Array{Int}(undef, K)
    idx4 = Array{Int}(undef, 4)
    QPThreadLocal{T}(tmp_k, tmp_k1, tmp_k1_, tmp_k2, tmp_k2_, tmp_XtX_p, 
        tmp_4k_k, tmp_4k_k_2, tmp_4k1, tmp_4k1_, tmp_5k1, tmp_5k1_, 
        tableau_k1, tableau_k2, tableau_4k1, tableau_5k1, swept, swept_4k, idx, idx4)
end
