

struct BackTracking2{O, TF, TI}
    c_1::TF
    ρ_hi::TF
    ρ_lo::TF
    iterations::TI
    maxstep::TF
end
function BackTracking2(::Val{O} = Val(3); c_1::TF = 1e-4, ρ_hi::TF = 0.5, ρ_lo::TF = 0.1, iterations::TI = 1_000, maxstep::TF = Inf) where {O,TF,TI}
    BackTracking2{O,TF,TI}(c_1, ρ_hi, ρ_lo, iterations, maxstep)
end

struct StaticOptimizationResults{T, Tf}
    # initial_x::Tx
    # minimizer::Tx
    minimum::Tf
    iterations::Int
    g_tol::T
    f_calls::Int
    g_calls::Int
    g_converged::Bool
    # h::Th
end
# function Base.show(io::IO, r::StaticOptimizationResults)
#     @printf io "Results of Static Optimization Algorithm\n"
#     @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
#     @printf io " * Minimum: [%s]\n" join(r.minimum, ",")
#     @printf io " * Hf(x): [%s]\n" join(r.h, ",")
#     @printf io " * Number of iterations: [%s]\n" join(r.iterations, ",")
#     @printf io " * Number of function calls: [%s]\n" join(r.f_calls, ",")
#     @printf io " * Number of gradient calls: [%s]\n" join(r.g_calls, ",")
#     @printf io " * Converged: [%s]\n" join(r.g_converged, ",")
#     nothing
# end

struct BFGSState2{P,T,L,LT}
    invH::SizedSIMDMatrix{P,P,T,L,LT}
    x_old::SizedSIMDVector{P,T,L}
    x_new::SizedSIMDVector{P,T,L}
    ∇_old::SizedSIMDVector{P,T,L}
    # ∇_new::SizedSIMDVector{P,T,L}
    δ∇::SizedSIMDVector{P,T,L}
    u::SizedSIMDVector{P,T,L}
    s::SizedSIMDVector{P,T,L}
end
function BFGSState2(::Val{P}, ::Type{T} = Float64) where {P,T}
    BFGSState2(
        SizedSIMDArray(undef, Val((P,P)), T),
        SizedSIMDArray(undef, Val((P,)), T),
        SizedSIMDArray(undef, Val((P,)), T),
        SizedSIMDArray(undef, Val((P,)), T),
        SizedSIMDArray(undef, Val((P,)), T),
        SizedSIMDArray(undef, Val((P,)), T),
        SizedSIMDArray(undef, Val((P,)), T)
    )
end
function initial_invH!(state::BFGSState2{P,T}) where {P,T}
    invH = state.invH
    @inbounds for i = 1:P
        for j ∈ 1:i-1
            invH[j,i] = zero(T)
        end
        invH[i,i] = one(T)
        for j ∈ i+1:P
            invH[j,i] = zero(T)
        end
    end
end

function optimize_light!(state, obj, x::SizedSIMDVector{P,T,L}, ls::BackTracking2{order}; tol = 1e-8) where {P,T,L,order}
    # res = DiffResults.GradientResult(x)
    # ls = BackTracking()
    # order = ordernum(bto)
    # copyto!(state.xinit, x)
    # copyto!(state.x_new, x)
    # xinit = copy(x)
    # x_new = copy(x)
    copyto!(state.x_old, x)
    initial_invH!(state)
    # hx = SMatrix{P,P,T}(I)
    # if !(hguess isa Nothing)
    #     hx = hguess * hx
    # end
    # hold = copy(hx)
    # jold = copy(x); s = copy(x)
    # @unpack c_1, ρ_hi, ρ_lo, iterations = ls
    c_1, ρ_hi, ρ_lo, iterations = ls.c_1, ls.ρ_hi, ls.ρ_lo, ls.iterations
    iterfinitemax = -log2(eps(eltype(x)))
    sqrttol = sqrt(eps(Float64))
    α_0 = one(T)
    N = 200
    f_calls = 0
    g_calls = 0
    for n = 1:N
        # res = ForwardDiff.gradient!(res, f, x); f_calls +=1; g_calls +=1; # Obtain gradient
        # ∇ = gradient!(d, state.x_old)
        # # ϕ_0 = DiffResults.value(res)
        # ϕ_0 = value(d)

        ϕ_0 = fdf(obj, state.x_old); f_calls +=1; g_calls +=1;
        ∇ = gradient(obj)

        isfinite(ϕ_0) || return NaN
        SIMDArrays.maximum_abs(∇) < tol && return ϕ_0
        if n > 1 # update hessian
            # y = jx - jold
            # hx = norm(y) < eps(eltype(x)) ? hx : hx + y*y' / (y'*s) - (hx*(s*s')*hx)/(s'*hx*s)


            # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
            @inbounds @simd for i ∈ 1:L
                state.δ∇[i] =  ∇[i] - state.∇_old[i]
            end
            # Update the inverse Hessian approximation using Sherman-Morrison
            dx_dg = real(dot(state.s, state.δ∇))
            # dx_dg == 0.0 && return true # force stop
            mul!(state.u, state.invH, state.δ∇)
            c2 = 1 / dx_dg
            c1 = fma(real(dot(state.δ∇, state.u)), c2*c2, c2)
            SIMDArrays.BFGS_update!(state.invH, state.s, state.u, c1, c2)
        end
        mul!(state.s, state.invH, ∇)
        SIMDArrays.scale!(state.s, -one(eltype(state.s)))
        # s = -hx\jx # Obtain direction
        dϕ_0 = dot(∇, state.s)
        if dϕ_0 >= 0. # If bad, reset search direction
            initial_invH!(state)
            # hx = hold
            s = -jx
            @inbounds @simd for i ∈ 1:L
                s[i] = -∇[i]
            end
            dϕ_0 = dot(∇, state.s)
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        @fastmath @inbounds @simd for i ∈ 1:L
            state.x_new[i] = state.x_old[i] + α_1*state.s[i]
        end
        # ϕx_1 = f(x + α_1*s); f_calls += 1;
        ϕx_1 = f(obj, state.x_new); f_calls += 1;

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = α_1/2
            @fastmath @inbounds @simd for i ∈ 1:L
                state.x_new[i] = state.x_old[i] + α_2*state.s[i]
            end
            # ϕx_1 = f(x + α_2*s); f_calls += 1;
            ϕx_1 = f(obj, state.x_new); f_calls += 1;
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            if iteration > iterations
                error("Linesearch failed to converge, reached maximum iterations $(iterations).",
                α_2)
            end

            # Shrink proposed step-size:
            if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                α_tmp = - (dϕ_0 * α_2^2) / ( 2 * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            else
                div = 1. / (α_1^2 * α_2^2 * (α_2 - α_1))
                a = (α_1^2*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2^2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                b = (-α_1^3*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2^3*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

                if norm(a) <= eps(Float64) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (2*b)
                else
                    # discriminant
                    d = max(b^2 - 3*a*dϕ_0, 0.)
                    # quadratic equation root
                    @fastmath α_tmp = (sqrt(d) - b) / (3*a)
                end
            end
            α_1 = α_2

            α_tmp = NaNMath.min(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = NaNMath.max(α_tmp, α_2*ρ_lo) # avoid too big reductions
            # α_tmp = min(α_tmp, α_2*ρ_hi) # avoid too small reductions
            # α_2 = max(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            # ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
            @fastmath @inbounds @simd for i ∈ 1:L
                state.x_new[i] = state.x_old[i] + α_2*state.s[i]
            end
            ϕx_0, ϕx_1 = ϕx_1, f(obj, state.x_new); f_calls += 1;
        end
        alpha, fpropose = α_2, ϕx_1

        # s = alpha*s
        # x = x + s # Update x
        @fastmath @inbounds @simd for i ∈ 1:L
            state.s[i] *= alpha
            state.x_old[i] += state.s[i]
        end

        # jold = copy(jx)
        copyto!(state.∇_old, ∇)
    end
    # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), state.x_old
    NaN
end
