module DifferentiableObjects

using   ForwardDiff,
        DiffResults,
        LineSearches,
        TriangularMatrices,
        Parameters


export  OnceDifferentiable,
        TwiceDifferentiable,
        optimize_light,
        LightOptions,
        BFGS



abstract type DifferentiableObject{P} end

debug() = false

include("forward_diff_differentiable.jl")
include("bfgs.jl")
include("optimize.jl")

end # module
