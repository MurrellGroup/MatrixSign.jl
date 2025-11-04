using MatrixSign
using Documenter

DocMeta.setdocmeta!(MatrixSign, :DocTestSetup, :(using MatrixSign); recursive=true)

makedocs(;
    modules=[MatrixSign],
    authors="Anton Oresten <antonoresten@gmail.com> and contributors",
    sitename="MatrixSign.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/MatrixSign.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/MatrixSign.jl",
    devbranch="main",
)
