using GradedArrays: GradedArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(GradedArrays, :DocTestSetup, :(using GradedArrays); recursive = true)

include("make_index.jl")

makedocs(;
    modules = [GradedArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "GradedArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/GradedArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"],
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(; repo = "github.com/ITensor/GradedArrays.jl", devbranch = "main", push_preview = true)
