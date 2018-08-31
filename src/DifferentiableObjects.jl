
__precompile__()
module DifferentiableObjects

using   ForwardDiff

using   DiffResults,
        # LineSearches,
        StaticArrays, # Explicitly support both
        jBLAS,
        SIMDArrays,
        Parameters,
        LinearAlgebra,
        Parameters,
        NaNMath

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

const SizedVector{P,T} = Union{SizedSIMDVector{P,T},MVector{P,T}}
const SizedSquareMatrix{P,T} = Union{SizedSIMDMatrix{P,P,T},MMatrix{P,P,T}}
const SymmetricMatrix{P,T} = Symmetric{T, <: SizedSIMDMatrix{P,P,T}}
# debug() = true

include("initial_linesearch_guess.jl")
include("backtracking.jl")

include("forward_diff_differentiable.jl")
include("bfgs.jl")
include("optimize.jl")
include("optimize_light.jl")
# include("miscellaneous.jl")




end # module
