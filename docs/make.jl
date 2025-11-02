using MatrixSign
using Documenter

DocMeta.setdocmeta!(MatrixSign, :DocTestSetup, :(using MatrixSign); recursive=true)

makedocs(;
    modules=[MatrixSign],
    authors="AntonOresten <antonoresten@gmail.com> and contributors",
    sitename="MatrixSign.jl",
    format=Documenter.HTML(;
        canonical="https://AntonOresten.github.io/MatrixSign.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Methods" => "methods.md",
        "API" => "API.md",
    ],
)

deploydocs(;
    repo="github.com/AntonOresten/MatrixSign.jl",
    devbranch="main",
)
