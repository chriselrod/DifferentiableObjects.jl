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

# mutable struct
abstract type Manifold end
struct ManifoldObjective{M <: Manifold, P, T<:DifferentiableObject{P}} <: DifferentiableObject{P}
    inner_obj::T
end
function ManifoldObjective(::M, inner_obj::T) where {M <: Manifold, P, T <: DifferentiableObject{P}}
    ManifoldObjective{M,P,T}(inner_obj)
end
# @inline function Base.getproperty(MO::ManifoldObjective{M}, s::Symbol) where M
#     if s == :inner_obj
#         return MO.inner_obj
#     elseif s == :manifold
#         return M
#     end
#     throw("ManifoldObjective has no field $s.")
# end

"""Flat Euclidean space {R,C}^N, with projections equal to the identity."""
struct Flat <: Manifold end
ManifoldObjective(::Flat, inner_obj::DifferentiableObject) = inner_obj
@inline retract(::Flat, x) = x
@inline retract!(::Flat,x) = x
@inline project_tangent(::Flat, g, x) = g
@inline project_tangent!(::Flat, g, x) = g
"""Spherical manifold {|x| = 1}."""
struct Sphere <: Manifold end
@inline retract!(::Sphere, x) = normalize!(x)
@inline project_tangent!(::Sphere, g, x) = (g .-= real(dot(x,g)).*x)
# @inline gradient(m::ManifoldObjective) = gradient(m.inner_obj)
function value!(obj::ManifoldObjective{M}, x) where M
    xin = retract(M, x)
    value!(obj.inner_obj, xin)
end
function value(obj::ManifoldObjective{M}) where M
    value(obj.inner_obj)
end
function gradient(obj::ManifoldObjective{M}) where M
    gradient(obj.inner_obj)
end
function gradient(obj::ManifoldObjective{M}, i::Int) where M
    gradient(obj.inner_obj,i)
end
function gradient!(obj::ManifoldObjective{M},x) where M
    xin = retract(M, x)
    gradient!(obj.inner_obj,xin)
    project_tangent!(M,gradient(obj.inner_obj),xin)
    return gradient(obj.inner_obj)
end
function value_gradient!(obj::ManifoldObjective{M}, x) where M
    xin = retract(M, x)
    value_gradient!(obj.inner_obj,xin)
    project_tangent!(M,gradient(obj.inner_obj),xin)
    return value(obj.inner_obj)
end


# abstract type AbstractOptimizer{M<:Manifold} end
# abstract type AbstractOptimizerState end
# abstract type FirstOrderOptimizer{M} <: AbstractOptimizer{M} end

# struct BFGS{M, IL, L, H<:Function} <: FirstOrderOptimizer{M}
#     alphaguess!::IL# = LineSearches.InitialStatic()
#     linesearch!::L# = LineSearches.HagerZhang()
#     initial_invH::H# = initial_diaginvH
#     manifold::M
# end
# function BFGS(;
#         alphaguess::IL = InitialStatic(),
#         linesearch::L = BackTracking(),
#         initial_invH::H = initial_diaginvH,
#         manifold::M = Flat()
#     ) where {IL, L, H, M <: Manifold}
#     BFGS{M, IL, L, H}(alphaguess, linesearch, initial_invH, manifold)
# end
# function BFGS{T}(;
#         alphaguess::IL = InitialStatic{T,false}(one(T)),
#         linesearch::L = BackTracking{T}(),
#         initial_invH::H = initial_diaginvH,
#         manifold::M = Flat()
#     ) where {T, IL <: InitialStatic{T}, L <: BackTracking{T}, H, M <: Manifold}
#     BFGS{M, IL, L, H}(alphaguess, linesearch, initial_invH, manifold)
# end

# # Base.summary(::BFGS) = "BFGS"

# @generated function initial_diaginvH(v::MVector{P,T}, α::T = one(T)) where {T,P}
#     quote
#         H = Symmetric(MMatrix{$P,$P,$T}(undef))
#         ind = 0
#         @inbounds for i = 1:$P
#             for j ∈ 1:i-1
#                 H.data[ind] = zero($T)
#             end
#             H.data[ind] = α
#             for j ∈ i+1:$P
#                 H.data[ind] = zero($T)
#             end
#         end
#         H
#     end
# end
# @generated function initial_diaginvH(v::SizedSIMDVector{P,T}, α::T = one(T)) where {T,P}
#     quote
#         H = Symmetric(SizedSIMDArray(undef, Val(($P,$P)), $T))
#         ind = 0
#         @inbounds for i = 1:$P
#             for j ∈ 1:i-1
#                 H.data[j,i] = zero($T)
#             end
#             H.data[i,i] = α
#             for j ∈ i+1:$P
#                 H.data[j,i] = zero($T)
#             end
#         end
#         H
#     end
# end
# @inline function initial_diaginvH(H::SymmetricMatrix{P,T},V::SizedVector{P,T}, α = one(T)) where {T,P}
#     # println("Resseting H.")
#     # @show H
#     @inbounds for i = 1:P
#         for j ∈ 1:i-1
#             H.data[j,i] = zero(T)
#         end
#         H.data[i,i] = α
#         for j ∈ i+1:P
#             H.data[j,i] = zero(T)
#         end
#     end
#     H
# end
# @inline function initial_diaginvH(H::SymmetricMatrix{P,T}, α = one(T)) where {T,P}
#     # println("Resseting H.")
#     # @show H
#     @inbounds for i = 1:P
#         for j ∈ 1:i-1
#             H.data[j,i] = zero(T)
#         end
#         H.data[i,i] = α
#         for j ∈ i+1:P
#             H.data[j,i] = zero(T)
#         end
#     end
#     H
# end

# mutable struct BFGSState{P,T,L,SV<:SizedSIMDVector{P,T,L},SM<:SymmetricMatrix{P,T}} <: AbstractOptimizerState
#     x::SV
#     x_previous::SV
#     g_previous::SV
#     f_x_previous::T
#     dx::SV
#     dg::SV
#     u::SV
#     invH::SM
#     s::SV
#     x_ls::SV
#     alpha::T
# end

# function initial_state(method::BFGS{M}, options, d, initial_x::SizedVector{P,T}) where {M,T,P}
#     initial_x = copy(initial_x)
#     retract!(method.manifold, initial_x)

#     value_gradient!!(d, initial_x)

#     project_tangent!(method.manifold, gradient(d), initial_x)
#     # Maintain a cache for line search results
#     # Trace the history of states visited
#     BFGSState(
#         initial_x, # Maintain current state in state.x
#         similar(initial_x), # Maintain previous state in state.x_previous
#         copy(gradient(d)), # Store previous gradient in state.g_previous
#         real(T)(NaN), # Store previous f in state.f_x_previous
#         similar(initial_x), # Store changes in position in state.dx
#         similar(initial_x), # Store changes in gradient in state.dg
#         similar(initial_x), # Buffer stored in state.u
#         method.initial_invH(initial_x), # Store current invH in state.invH
#         similar(initial_x), # Store current search direction in state.s
#         similar(initial_x), # Buffer of x for line search in state.x_ls
#         real(one(T))
#     )
# end



# function reset_search_direction!(state, d, method::BFGS)
#     method.initial_invH(state.invH, state.x)
#     grad = gradient(d)
#     @inbounds for i ∈ eachindex(state.s, grad)
#         state.s[i] = - grad[i]
#     end
#     return true
# end
# function update_state!(d, state::AbstractOptimizerState, method)

#     # @show state.s, state.invH, vec(gradient(d))
#     # Set the search direction
#     # Search direction is the negative gradient divided by the approximate Hessian
#     mul!(vec(state.s), state.invH.data, vec(gradient(d)))
#     SIMDArrays.scale!(state.s, -one(eltype(state.s)))
#     # # @show state.s
#     project_tangent!(method.manifold, state.s, state.x)
#     # @show state.s, state.x

#     # Maintain a record of the previous gradient
#     copyto!(state.g_previous, gradient(d))

#     # Determine the distance of movement along the search line
#     # This call resets invH to initial_invH is the former in not positive
#     # semi-definite
#     # lssuccess = perform_linesearch!(state, method, ManifoldObjective(M, d))
#     perform_linesearch!(state, method, ManifoldObjective(method.manifold, d)) # 120 ns
#     # @show state.alpha, state.s

#     # Update current position
#     @inbounds @simd for i ∈ 1:full_length(state.dx)
#         state.dx[i] = state.alpha * state.s[i]
#         state.x[i] = state.x[i] + state.dx[i]
#     end
#     retract!(method.manifold, state.x)
#     # @show state.x

#     # lssuccess == false # break on linesearch error
#     false
# end
# function perform_linesearch!(state, method, d)
#     # Calculate search direction dphi0
#     # # @show typeof(d)
#     # # @show @which gradient(d)
#     # @show gradient(d), state.s
#     dphi_0 = real(dot(gradient(d), state.s))
#     # @show dphi_0
#     # reset the direction if it becomes corrupted
#     if dphi_0 >= zero(dphi_0) && reset_search_direction!(state, d, method)
#         dphi_0 = real(dot(gradient(d), state.s)) # update after direction reset
#     end
#     phi_0  = value(d)

#     # Guess an alpha
#     method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)

#     # Store current x and f(x) for next iteration
#     state.f_x_previous = phi_0
#     copyto!(state.x_previous, state.x)

#     # Perform line search; catch LineSearchException to allow graceful exit
#     # try
#         # state.alpha, ϕalpha =
#     linesearch!(method.linesearch!, d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)
#     return true # lssuccess = true
#     # catch ex
#     #     if isa(ex, LineSearches.LineSearchException)
#     #         state.alpha = ex.alpha
#     #         Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.\nThe linesearch exited with message:\n$(ex.message)")
#     #         return false # lssuccess = false
#     #     else
#     #         rethrow(ex)
#     #     end
#     # end
# end
# #
# # function create_quote()
# #     q = quote @fastmath @inbounds begin end end
# #     @static if VERSION > v"0.7-"
# #         qa = q.args[2].args[3].args[3].args
# #     else
# #         qa = q.args[2].args[2].args[2].args
# #     end
# #     q, qa
# # end


# function update_h!(d, state::BFGSState{P,T,L}, method) where {P,T,L}
#     # Measure the change in the gradient
#     ∇ = gradient(d)
#     @inbounds @simd for i ∈ 1:L
#         state.dg[i] =  ∇[i] - state.g_previous[i]
#     end

#     # @show gradient(d), state.g_previous, state.dg
#     # Update the inverse Hessian approximation using Sherman-Morrison
#     dx_dg = real(dot(state.dx, state.dg))
#     dx_dg == 0.0 && return true # force stop
#     # @show dx_dg, state.invH
#     # @show state.u
#     # I may not support the use of `vec` here.
#     mul!(vec(state.u), state.invH.data, vec(state.dg))


#     c2 = 1 / dx_dg
#     c1 = fma(real(dot(state.dg, state.u)), c2*c2, c2)

#     # c1v2 = ( dx_dg + real(dot(state.dg, state.u))) / (dx_dg' * dx_dg)
#     # c2v2 = 1 / dx_dg
#     # @show state.u
#     # @show state.dx
#     # @show state.dg

#     # hx = copy(state.invH)
#     # @inbounds for i ∈ 1:P
#     #     sdxi = state.dx[i]
#     #     sui = state.u[i]
#     #     @simd for j ∈ 1:L
#     #         hx[j,i] += c1 * sdxi * state.dx[j] - c2 * (sui * state.dx[j] + state.u[j] * sdxi)
#     #     end
#     # end
#     # @show hx

#     # @show state.x, state.invH
#     SIMDArrays.BFGS_update!(state.invH, state.dx, state.u, c1, c2)
#     # @show state.x, state.invH
#     false
# end

# function maxdiff(x::AbstractArray{T1}, y::AbstractArray{T2}) where {T1,T2}
#     res = real(zero(promote_type(T1,T2)))
#     @inbounds for i in 1:length(x)
#         delta = abs(x[i] - y[i])
#         if delta > res
#             res = delta
#         end
#     end
#     return res
# end
# f_abschange(d::DifferentiableObject, state) = f_abschange(value(d), state.f_x_previous)
# f_abschange(f_x::T, f_x_previous) where T = abs(f_x - f_x_previous)
# x_abschange(state) = x_abschange(state.x, state.x_previous)
# x_abschange(x, x_previous) = maxdiff(x, x_previous)
# g_residual(d::DifferentiableObject) = g_residual(gradient(d))
# # g_residual(g) = norm(g, Inf)
# @inline g_residual(g) = SIMDArrays.maximum_abs(g)
# function gradient_convergence_assessment(state::AbstractOptimizerState, d, options)
#     g_residual(gradient(d)) ≤ options.g_tol
# end
# # gradient_convergence_assessment(state::ZerothOrderState, d, options) = false

# function assess_convergence(state::BFGSState, d, options)
#     default_convergence_assessment(state, d, options)
# end
# function default_convergence_assessment(state::AbstractOptimizerState, d, options)
#     x_converged, f_converged, f_increased, g_converged = false, false, false, false

#     # TODO: Create function for x_convergence_assessment
#     if x_abschange(state.x, state.x_previous) ≤ options.x_tol
#         x_converged = true
#     end

#     # Relative Tolerance
#     # TODO: Create function for f_convergence_assessment
#     f_x = value(d)
#     if f_abschange(f_x, state.f_x_previous) ≤ options.f_tol*abs(f_x)
#         f_converged = true
#     end

#     if f_x > state.f_x_previous
#         f_increased = true
#     end
#     g_converged = gradient_convergence_assessment(state,d,options)

#     converged = x_converged || f_converged || g_converged

#     return x_converged, f_converged, g_converged, converged, f_increased
# end

function BFGS_update_quote(Mₖ,Pₖ,stride_AD,T)
    T_size = sizeof(T)
    AD_stride = stride_AD * T_size
    W = VectorizationBase.REGISTER_SIZE ÷ T_size
    Q, r = divrem(stride_AD, W) #Assuming stride_AD is a multiple of W
    if Q > 0
        r == 0 || throw("Number of rows plus padding $stride_AD not a multiple of register size: $(VectorizationBase.REGISTER_SIZE).")
        L = VectorizationBase.CACHELINE_SIZE
    else
        W = r
        Q = 1
    end
    V = Vec{W,T}
    C = VectorizationBase.CACHELINE_SIZE ÷ T_size
    common = quote
        vC1 = vbroadcast($V, c1)
        vC2 = vbroadcast($V, -c2)
        ptr_invH, ptr_S, ptr_U = pointer(invH), pointer(s), pointer(u)
    end
    if Q > 1
        return quote
            $common
            # @nexprs $Pₖ p -> @nexprs $Q q -> invH_p_q = vload($V, ptr_invH + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))

            # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
            # @nexprs $Pₖ p -> begin
            for p ∈ 0:$T_size:$((Pₖ-1)*T_size)
                # @nexprs $Q q -> invH_q = vload($V, ptr_invH + $REGISTER_SIZE*(q-1) + $AD_stride*(p-1))
                # @nexprs $Q q -> invH_q = vload($V, ptr_invH + $REGISTER_SIZE*(q-1) + $AD_stride*p)
                # vSb = vbroadcast($V, unsafe_load(ptr_S + p-1 ))
                vSb = vbroadcast($V, VectorizationBase.load(ptr_S + p ))
                vSbc1 = vmul(vSb, vC1)
                vSbc2 = vmul(vSb, vC2)
                # vUbc2 = vbroadcast($V, unsafe_load(ptr_U + p-1 )) * vC2
                vUbc2 = vmul(vbroadcast($V, VectorizationBase.load(ptr_U + p )), vC2)
                # or
                # vSbc1 = vbroadcast($V, c1*unsafe_load(ptr_S + p-1 ))
                # vSbc2 = vbroadcast($V, c2*unsafe_load(ptr_S + p-1 ))
                # vUbc2 = vbroadcast($V, c2*unsafe_load(ptr_U + p-1 ))
                Base.Cartesian.@nexprs $Q q -> begin # I am concerned over the size of these dependency chains.
                    invH_q = vload($V, ptr_invH + $(VectorizationBase.REGISTER_SIZE)*(q-1) + $stride_AD*p)
                    vU_q = vload($V, ptr_U + $(VectorizationBase.REGISTER_SIZE)*(q-1))
                    invH_q = vmuladd(vU_q, vSbc2, invH_q)
                    vS_q = vload($V, ptr_S + $(VectorizationBase.REGISTER_SIZE)*(q-1))
                    invH_q = vmuladd(vS_q, vSbc1, vmuladd(vS_q, vUbc2, invH_q))
                    vstore!(ptr_invH + $(VectorizationBase.REGISTER_SIZE)*(q-1) + $stride_AD*p, invH_q)
                end
            end
            nothing
        end
    else
        return quote
            $common
            # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
            Base.Cartesian.@nexprs $Pₖ p -> begin
                invH_p = vload($V, ptr_invH + $AD_stride*(p-1))
                vSb = vbroadcast($V, VectorizationBase.load(ptr_S + (p-1)*$T_size ))
                vS = vload($V, ptr_S)
                # Split up the dependency chain and reduce number of operations, when we can't save on vC1 and vC2 multiplications.
                invH_p = vmuladd(vC2, vmuladd( vload($V, ptr_U), vSb, vmul(vS, vbroadcast($V, VectorizationBase.load(ptr_U + (p-1)*$T_size )))), vmuladd( vS, vmul(vSb,vC1), invH_p ))
                vstore!(ptr_invH + $AD_stride*(p-1), invH_p)
            end
            nothing
        end
    end
end

@generated function BFGS_update!(invH::Union{Symmetric{T,<:AbstractMutableFixedSizePaddedMatrix{P,P,T,R,L}},<:AbstractMutableFixedSizePaddedMatrix{P,P,T,R,L}},
    s::AbstractMutableFixedSizePaddedVector{P,T,R}, u::AbstractMutableFixedSizePaddedVector{P,T,R}, c1::T, c2::T) where {P,T,L,R}

    BFGS_update_quote(P,P,R,T)

end
