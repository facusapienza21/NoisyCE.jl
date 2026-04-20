using NoisyCE
using Documenter

DocMeta.setdocmeta!(NoisyCE, :DocTestSetup, :(using NoisyCE); recursive=true)

makedocs(;
    modules=[NoisyCE],
    authors="Facundo Sapienza",
    sitename="NoisyCE.jl",
    format=Documenter.HTML(;
        canonical="https://facusapienza21.github.io/NoisyCE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/facusapienza21/NoisyCE.jl",
    devbranch="main",
)
