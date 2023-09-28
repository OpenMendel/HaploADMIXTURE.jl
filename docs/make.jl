using HaploADMIXTURE
using Documenter

makedocs(;
    modules=[HaploADMIXTURE],
    authors="Seyoon Ko <kos@ucla.edu> and contributors",
    repo="https://github.com/OpenMendel/OpenADMIXTURE.jl/blob/{commit}{path}#L{line}",
    sitename="HaploADMIXTURE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://OpenMendel.github.io/HaploADMIXTURE.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly=true
)

deploydocs(;
    repo="github.com/OpenMendel/HaploADMIXTURE.jl",
    devbranch = "main"
)
