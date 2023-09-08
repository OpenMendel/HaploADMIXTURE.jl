# HaploADMIXTURE.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://OpenMendel.github.io/HaploADMIXTURE.jl/dev)

This software package is an open-source Julia implementation of HaploADMIXTURE, ancestry inference by modeling haplotypes. By modeling haplotypes, we use information between nearby SNPs, obtaining more accurate ancestry estimates.

It supports acceleartion through multithreading and graphic processing units (GPUs). By directly utilizing the data format of the PLINK BED file, the memory usage is highly efficient. 

It estimates ancestry with maximum-likelihood method for a large SNP genotype datasets, where individuals are assumed to be unrelated. The input is binary PLINK 1 BED-formatted file (`.bed`). Also, you will need an idea of $K$, the number of ancestral populations. One possible way to figure out a good value of $K$ is through Akaike information criterion. If the number of SNPs is too large, you may choose to run on a subset of SNPs selected by their information content, using the blockwise [sparse $K$-means via feature ranking](https://github.com/kose-y/SKFR.jl) (SKFR) method. When SKFR is applied, it selects given number of blocks of two nearby SNPs.

## Installation

This package requires Julia v1.7 or later, which can be obtained from
<https://julialang.org/downloads/> or by building Julia from the sources in the
<https://github.com/JuliaLang/julia> repository.

The package can be installed by running the following code:
```julia
using Pkg
pkg"add https://github.com/kose-y/SKFR.jl"
pkg"add https://github.com/OpenMendel/OpenADMIXTURE.jl"
pkg"add https://github.com/OpenMendel/HaploADMIXTURE.jl"
```
For running the examples in our documentation, the following are also necessary. 
```julia
pkg"add SnpArrays DelimitedFiles StableRNGs"
```

For GPU support, an Nvidia GPU is required. Also, the following package has to be installed:
```julia
pkg"add CUDA"
```

## Citation
The methods and applications of this software package are detailed in the following publication:

_To be updated._

If you use OpenMendel analysis packages in your research, please cite the following reference in the resulting publications:

_Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. OPENMENDEL: a cooperative programming project for statistical genetics. Hum Genet. 2020 Jan;139(1):61-71. doi: 10.1007/s00439-019-02001-z. Epub 2019 Mar 26. PMID: 30915546; PMCID: [PMC6763373](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6763373/)._
