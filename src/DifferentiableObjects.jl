
__precompile__()
module DifferentiableObjects

using   ForwardDiff,
        DiffResults,
        LineSearches,
        TriangularMatrices,
        Parameters,
        LinearAlgebra

import  NLSolversBase: AbstractObjective,
        value!!,
        value!,
        value,
        gradient,
        gradient!,
        value_gradient!,
        value_gradient!!,
        hessian!,
        hessian!!
        
export  DifferentiableObject,
        GradientConfiguration,
        HessianConfiguration,
        OnceDifferentiable,
        TwiceDifferentiable,
        optimize_light,
        LightOptions,
        BFGS,
        value,
        value!,
        gradient,
        gradient!,
        hessian,
        hessian!,
        value_gradient!



abstract type DifferentiableObject{P} <: AbstractObjective end

debug() = true

include("forward_diff_differentiable.jl")
include("bfgs.jl")
include("optimize.jl")
include("miscellaneous.jl")

end # module
