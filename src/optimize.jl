### This code was taken and modified from jl.
# jl is licensed under the MIT License:

# > Copyright (c) 2012: John Myles White, Tim Holy, and other contributors.
# > Copyright (c) 2016: Patrick Kofod Mogensen, John Myles White, Tim Holy, 
# >                     and other contributors. 
# > Copyright (c) 2017: Patrick Kofod Mogensen, Asbjørn Nilsen Riseth, 
# >                     John Myles White, Tim Holy, and other contributors.
# >
# > Permission is hereby granted, free of charge, to any person
# > obtaining a copy of this software and associated documentation files
# > (the "Software"), to deal in the Software without restriction,
# > including without limitation the rights to use, copy, modify, merge,
# > publish, distribute, sublicense, and/or sell copies of the Software,
# > and to permit persons to whom the Software is furnished to do so,
# > subject to the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# > NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# > BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# > ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# > CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# > SOFTWARE.



struct LightOptions{T} #add iterations?
    x_tol::T
    f_tol::T
    g_tol::T
end
LightOptions() = LightOptions{Float64}(0.0,0.0,1e-8)



# mutable struct BFGSState{Tx, Tm, T,G} <: AbstractOptimizerState
#     x::Tx
#     x_previous::Tx
#     g_previous::G
#     f_x_previous::T
#     dx::Tx
#     dg::Tx
#     u::Tx
#     invH::Tm
#     s::Tx
#     @add_linesearch_fields()
# end
function initial_state!(state, method::BFGS{M}, d, initial_x::AbstractArray{T}) where {T,M}
    n = length(initial_x)
    copyto!(state.x, initial_x)
    retract!(M, initial_x)
    value_gradient!!(d, initial_x)
    project_tangent!(M, gradient(d), initial_x)
    copyto!(state.g_previous, gradient(d))
    method.initial_invH((state.invH, initial_x))
    state.alpha = one(T)#/2
    nothing
end
# function initial_state!(state, method::BFGS, d, initial_x::AbstractArray{T}) where T
#     n = length(initial_x)
#     copyto!(state.x, initial_x)
#     retract!(method.manifold, initial_x)
#     value_gradient!!(d, initial_x)
#     project_tangent!(method.manifold, gradient(d), initial_x)
#     copyto!(state.g_previous, gradient(d))
#     method.initial_invH(state.invH, initial_x)
#     state.alpha = one(T)
#     nothing
# end

function uninitialized_state(initial_x::RecursiveVector{T,P}) where {T,P}
    BFGSState(similar(initial_x), # Maintain current state in state.x
        similar(initial_x), # Maintain previous state in state.x_previous
        similar(initial_x), # Store previous gradient in state.g_previous
        T(NaN), # Store previous f in state.f_x_previous
        similar(initial_x), # Store changes in position in state.dx
        similar(initial_x), # Store changes in gradient in state.dg
        similar(initial_x), # Buffer stored in state.u
        SymmetricMatrix{T,P}(),
        #RecursiveMatrix{T,P,P}(), # Store current invH in state.invH
        similar(initial_x), # Store current search direction in state.s
        # T(NaN),            # Keep track of previous descent value ⟨∇f(x_{k-1}), s_{k-1}⟩
        similar(initial_x), # Buffer of x for line search in state.x_ls
        one(T))
end
function initial_convergence(d, state, method::AbstractOptimizer, initial_x, options)
    gradient!(d, initial_x)
    norm(gradient(d), Inf) < options.g_tol
end

function optimize_light(d::D, initial_x::Tx, method::M,
                  options::LightOptions = LightOptions(),
                  state = initial_state(method, options, d, initial_x)) where {D<:DifferentiableObject, M<:AbstractOptimizer, Tx <: AbstractArray}
    # @show state

    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased = false, false, false

    g_converged = initial_convergence(d, state, method, initial_x, options)
    # @show g_converged
    converged = g_converged

    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0
    # println("\n")
    while !converged && iteration < 60#options.iterations #uncomment if you add back iterations field to LightOptions
        iteration += 1
        # @show gradient(d)
        update_state!(d, state, method) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS, or linesearch errors)
        # @show state.x_previous, state.g_previous
        update_g!(d, state, method) # TODO: Should this be `update_fg!`?
        x_converged, f_converged,
        g_converged, converged, f_increased = assess_convergence(state, d, options)
        !converged && update_h!(d, state, method) # only relevant if not converged
        # debug() && # @show state.f_x_previous, value(d)
        # println("\n")
    end # while
    # println("\n")
    # @show iteration, x_converged, f_converged, g_converged, converged
    f_incr_pick = disallow_f_increases(options) && f_increased
    f_incr_pick ? ( state.x_previous, state.f_x_previous ) : ( state.x, value(d) )
end
disallow_f_increases(options) = false
# disallow_f_increases(options::Options) = options.allow_f_increases

function update_g!(d, state, method::FirstOrderOptimizer{M}) where M# Union{FirstOrderOptimizer, Newton}
    # Update the function value and gradient
    value_gradient!(d, state.x)
    # if M <: FirstOrderOptimizer #only for methods that support manifold optimization
    project_tangent!(M, gradient(d), state.x)
    # end
end