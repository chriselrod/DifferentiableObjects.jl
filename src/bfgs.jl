### This code was taken and modified from Optim.jl.
# Optim.jl is licensed under the MIT License:

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

abstract type AbstractOptimizer end
abstract type FirstOrderOptimizer <: AbstractOptimizer end

@with_kw struct BFGS{IL, L, H<:Function} <: FirstOrderOptimizer
    alphaguess!::IL = LineSearches.InitialStatic()
    linesearch!::L = LineSearches.HagerZhang()
    initial_invH::H = initial_diaginvH
    manifold::Manifold = Flat()
end

# Base.summary(::BFGS) = "BFGS"

function initial_diaginvH(::RecursiveVector{T,L}, α::T = one(T)) where {T,L}
    H = SymmetricMatrix{T,L}()
    ind = 0
    for i = 1:L
        for j ∈ 1:i-1
            ind += 1
            H[ind] = zero(T)
        end
        ind += 1
        H[ind] = α
    end
    H
end
function initial_diaginvH((H,V)::Tuple{SymmetricMatrix{T,N,L},RecursiveVector{T,N}}, α = one(T)) where {T,N,L}
    ind = 0
    for i = 1:L
        for j ∈ 1:i-1
            ind += 1
            H[ind] = zero(T)
        end
        ind += 1
        H[ind] = α
    end
    H
end

mutable struct BFGSState{Tx, Tm, T,G} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    f_x_previous::T
    dx::Tx
    dg::Tx
    u::Tx
    invH::Tm
    s::Tx
    x_ls::Tx
    alpha::T
end

function initial_state(method::BFGS, options, d, initial_x::AbstractArray{T}) where T
    n = length(initial_x)
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)
    # Maintain a cache for line search results
    # Trace the history of states visited
    BFGSState(
        initial_x, # Maintain current state in state.x
        similar(initial_x), # Maintain previous state in state.x_previous
        copy(gradient(d)), # Store previous gradient in state.g_previous
        real(T)(NaN), # Store previous f in state.f_x_previous
        similar(initial_x), # Store changes in position in state.dx
        similar(initial_x), # Store changes in gradient in state.dg
        similar(initial_x), # Buffer stored in state.u
        method.initial_invH(initial_x), # Store current invH in state.invH
        similar(initial_x), # Store current search direction in state.s
        similar(initial_x), # Buffer of x for line search in state.x_ls
        real(one(T))
    )
end

# mutable struct
struct ManifoldObjective{T<:DifferentiableObject} <: DifferentiableObject
    manifold::Manifold
    inner_obj::T
end
abstract type Manifold end
"""Flat Euclidean space {R,C}^N, with projections equal to the identity."""
struct Flat <: Manifold end
@inline retract(M::Flat, x) = x
@inline retract!(M::Flat,x) = x
@inline project_tangent(M::Flat, g, x) = g
@inline project_tangent!(M::Flat, g, x) = g
"""Spherical manifold {|x| = 1}."""
struct Sphere <: Manifold end
@inline retract!(S::Sphere, x) = normalize!(x)
@inline project_tangent!(S::Sphere,g,x) = (g .-= real(vecdot(x,g)).*x)



function reset_search_direction!(state, d, method::BFGS)
    method.initial_invH((state.invH, state.x))
    state.s .= .-gradient(d)
    return true
end
function update_state!(d, state::BFGSState, method::BFGS)
    n = length(state.x)

    # Set the search direction
    # Search direction is the negative gradient divided by the approximate Hessian
    A_mul_B!(vec(state.s), state.invH, vec(gradient(d)))
    scale!(state.s, -1)
    project_tangent!(method.manifold, state.s, state.x)

    # Maintain a record of the previous gradient
    copy!(state.g_previous, gradient(d))

    # Determine the distance of movement along the search line
    # This call resets invH to initial_invH is the former in not positive
    # semi-definite
    lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

    # Update current position
    state.dx .= state.alpha.*state.s
    state.x .= state.x .+ state.dx
    retract!(method.manifold, state.x)

    lssuccess == false # break on linesearch error
end
function perform_linesearch!(state, method, d)
    # Calculate search direction dphi0
    dphi_0 = real(vecdot(gradient(d), state.s))
    # reset the direction if it becomes corrupted
    if dphi_0 >= zero(dphi_0) && reset_search_direction!(state, d, method)
        dphi_0 = real(vecdot(gradient(d), state.s)) # update after direction reset
    end
    phi_0  = value(d)

    # Guess an alpha
    method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)

    # Store current x and f(x) for next iteration
    state.f_x_previous = phi_0
    copy!(state.x_previous, state.x)

    # Perform line search; catch LineSearchException to allow graceful exit
    try
        state.alpha, ϕalpha =
            method.linesearch!(d, state.x, state.s, state.alpha,
                               state.x_ls, phi_0, dphi_0)
        return true # lssuccess = true
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            state.alpha = ex.alpha
            Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.\nThe linesearch exited with message:\n$(ex.message)")
            return false # lssuccess = false
        else
            rethrow(ex)
        end
    end
end



function update_h!(d, state, method::BFGS)
    n = length(state.x)
    # Measure the change in the gradient
    state.dg .= gradient(d) .- state.g_previous

    # Update the inverse Hessian approximation using Sherman-Morrison
    dx_dg = real(vecdot(state.dx, state.dg))
    if dx_dg == 0.0
        return true # force stop
    end
    A_mul_B!(vec(state.u), state.invH, vec(state.dg))

    c1 = (dx_dg + real(vecdot(state.dg, state.u))) / (dx_dg' * dx_dg)
    c2 = 1 / dx_dg

    # TODO BLASify this
    # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
    for i in 1:n
        @simd for j in 1:n
            @inbounds state.invH[i, j] += c1 * state.dx[i] * state.dx[j]' - c2 * (state.u[i] * state.dx[j]' + state.u[j]' * state.dx[i])
        end
    end
end

function maxdiff(x::AbstractArray, y::AbstractArray)
    res = real(zero(x[1] - y[1]))
    @inbounds for i in 1:length(x)
        delta = abs(x[i] - y[i])
        if delta > res
            res = delta
        end
    end
    return res
end
f_abschange(d::DifferentiableObject, state) = f_abschange(value(d), state.f_x_previous)
f_abschange(f_x::T, f_x_previous) where T = abs(f_x - f_x_previous)
x_abschange(state) = x_abschange(state.x, state.x_previous)
x_abschange(x, x_previous) = maxdiff(x, x_previous)
g_residual(d::DifferentiableObject) = g_residual(gradient(d))
gradient_convergence_assessment(state::AbstractOptimizerState, d, options) = g_residual(gradient(d)) ≤ options.g_tol
gradient_convergence_assessment(state::ZerothOrderState, d, options) = false

function assess_convergence(state::BFGSState, d, options)
  default_convergence_assessment(state, d, options)
end
function default_convergence_assessment(state::AbstractOptimizerState, d, options)
    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    # TODO: Create function for x_convergence_assessment
    if x_abschange(state.x, state.x_previous) ≤ options.x_tol
        x_converged = true
    end

    # Relative Tolerance
    # TODO: Create function for f_convergence_assessment
    f_x = value(d)
    if f_abschange(f_x, state.f_x_previous) ≤ options.f_tol*abs(f_x)
        f_converged = true
    end

    if f_x > state.f_x_previous
        f_increased = true
    end

    g_converged = gradient_convergence_assessment(state,d,options)

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, f_increased
end


