using RealLabelNormalization
using Documenter

DocMeta.setdocmeta!(RealLabelNormalization, :DocTestSetup, :(using RealLabelNormalization); recursive=true)

makedocs(;
    modules=[RealLabelNormalization],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="RealLabelNormalization.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/RealLabelNormalization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/RealLabelNormalization.jl",
    devbranch="main",
)
