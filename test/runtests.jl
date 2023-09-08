using HaploADMIXTURE, SnpArrays, StableRNGs
using Test

@testset "HaploADMIXTURE.jl" begin
    EUR = SnpArrays.datadir("EUR_subset.bed")
    rng = StableRNG(7856)
    d, _, _ = HaploADMIXTURE.run_admixture(EUR, 379, 27025, 4; rng=rng, admix_rtol=1e-5)
    @test d.ll_new ≈ -1.3344074094647754e7
    d, _, _ = HaploADMIXTURE.run_admixture(EUR, 379, 1000, 4; sparsity=1000, rng=rng, prefix="test", admix_rtol=1e-5)
    @test d.ll_new ≈ -440149.1599139798
    Sys.iswindows() || rm("test_4_2000aims.bed", force=true)
    Sys.iswindows() || rm("test_4_2000aims.bim", force=true)
    Sys.iswindows() || rm("test_4_2000aims.fam", force=true)
end
