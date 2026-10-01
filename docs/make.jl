using Documenter: Documenter, DocMeta, deploydocs, makedocs
using GradedArrays
using ITensorFormatter: ITensorFormatter

# `using GradedArrays` above is whole-module rather than an explicit list on purpose:
# Documenter renders an `@example` result against the `Main` of this process rather than the
# page's own module, so an exported name prints unqualified only if it is in scope here.
DocMeta.setdocmeta!(GradedArrays, :DocTestSetup, :(using GradedArrays); recursive = true)

ITensorFormatter.make_index!(pkgdir(GradedArrays))

makedocs(;
    modules = [GradedArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "GradedArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/GradedArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = [
        "Home" => "index.md",
        "User Interface" => [
            "Symmetry sectors" => "user_interface/sectors.md",
            "Graded arrays" => "user_interface/graded_arrays.md",
        ],
        "Reference" => "reference.md",
        "Internals" => "internals.md",
    ]
)

deploydocs(;
    repo = "github.com/ITensor/GradedArrays.jl",
    devbranch = "main",
    push_preview = true
)
