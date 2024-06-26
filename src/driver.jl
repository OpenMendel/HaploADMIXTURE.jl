"""
    run_admixture(filename, I, J, K;
        rng=Random.GLOBAL_RNG, 
        sparsity=nothing,
        prefix=filename[1:end-4],
        skfr_tries = 1,
        skfr_max_inner_iter = 50,
        admix_n_iter = 1000,
        admix_rtol=1e-7,
        admix_n_em_iter = 5,
        T = Float64,
        Q = 3,
        use_gpu=false,
        verbose=false,
        progress_bar=true)

The main runner function for admixture. 

# Input: 
- `filename``: the PLINK BED file name to analyze, including the extension.
- `I`: number of individuals. First I will be used.
- `J`: number of haplotype pairs. First 2J AIMs are used.
- `K`: number of clusters.
- `rng`: random number generator.
- `sparsity`: number of blocks to be selected by SKFR. `nothing` to not run the SKFR step. 
    Note: 2 * sparsity is the AIMs being selected.
- `prefix`: prefix used for the output PLINK file if SKFR is used.
- `skfr_tries`: number of repeats of SKFR with different initializations.
- `skfr_max_inner_iter`: maximum number of iterations for each call for SKFR.
- `admix_n_iter`: number of Admixture iterations
- `admix_rtol`: relative tolerance for Admixture
- `admix_n_em_iters`: number of iterations for EM initialization
- `T`: Internal type for floating-point numbers
- `Q`: number of steps used for quasi-Newton acceleration
- `use_gpu`: whether to use GPU for computation
- `progress_bar`: whether to show a progress bar for main loop
"""
function run_admixture(filename, I, J, K; 
    rng=Random.GLOBAL_RNG, 
    sparsity=nothing, 
    prefix=filename[1:end-4],
    skfr_tries = 1, 
    skfr_max_inner_iter=50,
    skfr_mode=:global, 
    admix_n_iter=1000, 
    admix_rtol=1e-7, 
    admix_n_em_iters = 5, 
    T=Float64, 
    T2=Float64,
    Q=3, 
    use_gpu=false,
    verbose=false,
    progress_bar=false,
    fix_q=false,
    fix_p=false,
    init_q=nothing,
    init_p=nothing)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    println("Using $filename as input.")
    if sparsity !== nothing
        ftn = if skfr_mode == :global
            SparseKmeansFeatureRanking.sparsekmeans1
        elseif skfr_mode == :local
            SparseKmeansFeatureRanking.sparsekmeans2
        else
            @assert false "skfr_mode can only be :global or :local"
        end
        admix_input, clusters, aims = _filter_SKFR(filename, K, sparsity; rng=rng, prefix=prefix, 
            tries=skfr_tries, max_inner_iter=skfr_max_inner_iter, ftn=ftn)
    else
        admix_input = filename
        clusters, aims = nothing, nothing
    end
    d = _admixture_base(admix_input, I, J, K; 
        n_iter=admix_n_iter, rtol=admix_rtol, rng=rng, em_iters=admix_n_em_iters, 
        T=T, T2=T2, Q=Q, use_gpu=use_gpu,
        fix_q=fix_q, fix_p=fix_p, init_q=init_q, init_p=init_p)
    d, clusters, aims
end

"""
Run SKFR then filter the PLINK file to only keep AIMs.
"""
function _filter_SKFR(filename, K, sparsity::Integer; 
    rng=Random.GLOBAL_RNG,
    prefix=filename[1:end-4],
    tries = 10,
    max_inner_iter = 50, 
    ftn = SparseKmeansFeatureRanking.sparsekmeans1)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    g = SnpArray(filename)
    ISM = SparseKmeansFeatureRanking.ImputedSnpMatrix{Float64}(g, K; rng=rng, blocksize=2)
    if tries == 1
        (clusters, _, aims, _, _) = ftn(ISM, sparsity; max_iter = max_inner_iter, squares=false)
    else
        (clusters, _, aims, _, _, _) = SparseKmeansFeatureRanking.sparsekmeans_repeat(ISM, sparsity; iter = tries, 
            max_inner_iter=max_inner_iter, ftn=ftn)
    end
    I, J = size(g)
    aims_sorted = sort(aims)
    des = "$(prefix)_$(K)_$(sparsity * 2)aims"
    println(des)
    SnpArrays.filter(filename[1:end-4], trues(I), aims_sorted; des=des)
    des * ".bed", clusters, aims
end

"""
Run SKFR then filter the PLINK file to only keep AIMs. Run for multiple sparsities (in decreasing order)
"""
function _filter_SKFR(filename, K, sparsities::AbstractVector{<:Integer}; 
    rng=Random.GLOBAL_RNG,
    prefix=filename[1:end-4],
    tries = 10,
    max_inner_iter = 50)
    @assert endswith(filename, ".bed") "filename should end with .bed"
    g = SnpArray(filename)
    ISM = SparseKmeansFeatureRanking.ImputedSnpMatrix{Float64}(g, K; rng=rng, blocksize=2)
    if typeof(sparsities) <: AbstractVector
        @assert issorted(sparsities; rev=true) "sparsities should be decreasing"
        (clusters, aims) = SparseKmeansFeatureRanking.sparsekmeans_path(ISM, sparsities; iter=tries, max_inner_iter=max_inner_iter)
    end
    I, J = size(g)
    outputfiles = String[]
    for (s, aimlist) in zip(sparsities, aims)
        aimlist_sorted = sort(aimlist)
        des = "$(prefix)_$(K)_$(s)aims"
        SnpArrays.filter(filename[1:end-4], trues(I), aimlist_sorted; des=des)
        push!(outputfiles, des)
    end
    outputfiles, clusters, aims
end

function _admixture_base(filename, I, J, K; 
    n_iter=1000, 
    rtol=1e-7, 
    rng=Random.GLOBAL_RNG,
    em_iters = 5, 
    T=Float64, 
    T2=Float64,
    Q=3, 
    use_gpu=false,
    verbose=false,
    fix_q=false,
    fix_p=false,
    init_q=nothing,
    init_p=nothing)
    println("Loading genotype data...")
    g = SnpArray(filename)
    g_la = SnpLinAlg{T2}(g)

    println("Loaded $I samples and $(2J) SNPs")
    d = AdmixData2{T, T2}(I, J, K, Q, g; rng=rng)
    if use_gpu
        d_cu, g_cu = _cu_admixture_base(d, g_la, I, J)
    else 
        d_cu = nothing
        g_cu = nothing
    end
    if verbose
        if init_q === nothing || init_p === nothing
            @time init_em!(d, g_la, em_iters;
                        d_cu = d_cu, g_cu=g_cu)
        end
        if init_q !== nothing
            d.q .= init_q
        end
        if init_p !== nothing
            d.p .= init_p
        end
        @time if progress_bar
            messages = @capture_out admixture_qn!(d, g_la, n_iter, rtol;
                d_cu = d_cu, g_cu = g_cu, mode=:ZAL,
                fix_q=fix_q, fix_p=fix_p)
            println(messages)
        else
            admixture_qn!(d, g_la, n_iter, rtol;
                d_cu = d_cu, g_cu = g_cu, mode=:ZAL,
                fix_q=fix_q, fix_p=fix_p)
        end
    else
        if init_q === nothing || init_p === nothing
            init_em!(d, g_la, em_iters; d_cu = d_cu, g_cu=g_cu)
        end
        if init_q !== nothing
            d.q .= init_q
        end
        if init_p !== nothing
            d.p .= init_p
        end
        # if progress_bar
        #     admixture_qn!(d, g_la, n_iter, rtol;
        #         d_cu = d_cu, g_cu = g_cu, mode=:ZAL, #progress_bar=true,
        #         fix_q=fix_q, fix_p=fix_p)
        # else
        admixture_qn!(d, g_la, n_iter, rtol;
            d_cu = d_cu, g_cu = g_cu, mode=:ZAL,
            fix_q=fix_q, fix_p=fix_p)
        # end
    end
    d
end

function _cu_admixture_base(d, g_la, I, J)
    # dummy, main body defined inside CUDA portion.
end
