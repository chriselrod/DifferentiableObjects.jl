
module DifferentiableObjects

using   ForwardDiff,
        DiffResults,
        # StaticArrays, # Explicitly support both
        NaNMath,
        SIMDPirates,
        PaddedMatrices,
        Parameters,
        LinearAlgebra,
        VectorizationBase

using SIMDPirates: vadd, vmul, evmul, vbroadcast, Vec
using PaddedMatrices:   AbstractFixedSizeVector, AbstractFixedSizeMatrix,
                        AbstractMutableFixedSizeVector, AbstractMutableFixedSizeMatrix,
                        ConstantFixedSizeVector, ConstantFixedSizeMatrix,
                        MutableFixedSizeVector, MutableFixedSizeMatrix,
                        AbstractFixedSizeArray, PtrVector, PtrMatrix

# import  NLSolversBase: AbstractObjective,
#         value!!,
#         value!,
#         value,
#         gradient,
#         gradient!,
#         value_gradient!,
#         value_gradient!!,
#         hessian!,
#         hessian!!

export  AbstractDifferentiableObject,
        GradientConfiguration,
        HessianConfiguration,
        OnceDifferentiable,
        TwiceDifferentiable,
        optimize_light!,
        optimize_scale!,
        value,
        value!,
        gradient,
        gradient!,
        hessian,
        hessian!,
        value_gradient!



abstract type AbstractDifferentiableObject{P,T} end #<: AbstractObjective end

# const SizedVector{P,T} = Union{AbstractFixedSizeVector{P,T},MVector{P,T}}
# const SizedSquareMatrix{P,T} = Union{AbstractFixedSizeMatrix{P,P,T},MMatrix{P,P,T}}
const SizedVector{P,T} = AbstractFixedSizeVector{P,T}
const SizedSquareMatrix{P,T} = AbstractFixedSizeMatrix{P,P,T}
const SymmetricMatrix{P,T} = Symmetric{T, <: AbstractFixedSizeMatrix{P,P,T}}
# debug() = true

###
### This replaces the default fallback, which uses unsafe_load.
###
@inline function VectorizationBase.load(ptr::Ptr{D}) where {T, V, N, D <: ForwardDiff.Dual{T,V,N}}
    vptr = Base.unsafe_convert(Ptr{V}, ptr)
    D(
        VectorizationBase.load(vptr),
        ForwardDiff.Partials{N,V}(
            ntuple(n -> VectorizationBase.load(vptr + n*sizeof(V)), Val(N))
        )
    )
end
### This replaces the default fallback, which uses unsafe_store!
@inline function VectorizationBase.store!(ptr::Ptr{D}, v::D) where {T,V,N, D <: ForwardDiff.Dual{T,V,N}}
    vptr = Base.unsafe_convert(Ptr{V}, ptr)
    val = v.value
    partials = v.partials
    VectorizationBase.store!(vptr, val)
    # Compiler should unroll
    @inbounds @simd for n ∈ 1:N
        VectorizationBase.store!(vptr + n*sizeof(V), partials[n])
    end
    v
end

include("initial_linesearch_guess.jl")
include("backtracking.jl")

include("forward_diff_differentiable.jl")
include("scaled_gradients.jl")
include("bfgs.jl")
# include("optimize.jl")
include("optimize_light.jl")
# include("miscellaneous.jl")




end # module
