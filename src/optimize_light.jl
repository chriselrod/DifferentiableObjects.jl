

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
function BackTracking2{TF}(::Val{O} = Val(3); c_1 = TF(1e-4), ρ_hi = TF(0.5), ρ_lo = TF(0.1), iterations::TI = 1_000, maxstep = TF(Inf)) where {O,TF,TI}
    BackTracking2{O,TF,TI}(c_1, ρ_hi, ρ_lo, iterations, maxstep)
end


# struct StaticOptimizationResults{T, Tf}
#     # initial_x::Tx
#     # minimizer::Tx
#     minimum::Tf
#     iterations::Int
#     g_tol::T
#     f_calls::Int
#     g_calls::Int
#     g_converged::Bool
#     # h::Th
# end
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

mutable struct BFGSState2{P,T,L,LT}
    invH::ConstantFixedSizePaddedMatrix{P,P,T,L,LT}
    x_old::ConstantFixedSizePaddedVector{P,T,L,L}
    x_new::ConstantFixedSizePaddedVector{P,T,L,L}
    ∇_old::ConstantFixedSizePaddedVector{P,T,L,L}
    # ∇_new::SizedSIMDVector{P,T,L}
    δ∇::ConstantFixedSizePaddedVector{P,T,L,L}
    u::ConstantFixedSizePaddedVector{P,T,L,L}
    s::ConstantFixedSizePaddedVector{P,T,L,L}
    function BFGSState2{P,T,L,LT}(::UndefInitializer) where {P,T,L,LT}
        new{P,T,L,LT}()
    end
    @generated function BFGSState2{P}(::UndefInitializer) where {P}
        L = PaddedMatrices.calc_padding(P, Float64)
        :(BFGSState2{$P,Float64,$L,$(P*L)}(undef))
    end
    @generated function BFGSState2{P,T}(::UndefInitializer) where {P,T}
        L = PaddedMatrices.calc_padding(P, T)
        :(BFGSState2{$P,$T,$L,$(P*L)}(undef))
    end
end
BFGSState2(::Val{P}, ::Type{T} = Float64) where {P,T} = BFGSState2{P,T}(undef)
@inline Base.pointer(s::BFGSState2{P,T})  where {P,T} = Base.unsafe_convert(Ptr{T}, pointer_from_objref(s))

@inline get_invH(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrMatrix{P,P,T,L,LT}(pointer(s))
@inline get_x_old(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + LT*sizeof(T))
@inline get_x_new(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + (LT+L)*sizeof(T))
@inline get_∇_old(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + (LT+2L)*sizeof(T))
@inline get_δ∇(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + (LT+3L)*sizeof(T))
@inline get_u(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + (LT+4L)*sizeof(T))
@inline get_s(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + (LT+5L)*sizeof(T))

function initial_invH!(state::BFGSState2{P,T}) where {P,T}
    invH = get_invH(state)
    fill!(invH, zero(T))
    @inbounds for p = 1:P
        invH[p,p] = one(T)
    end
end
@inline optimum(s::BFGSState2{P,T,L,LT}) where {P,T,L,LT} = PtrVector{P,T,L,L}(pointer(s) + LT*sizeof(T))

@noinline linesearch_failure(iterations) = error("Linesearch failed to converge, reached maximum iterations $(iterations).")

"""
Optimum value is stored in state.x_old.

"""
function optimize_light!(state, obj, x::AbstractFixedSizePaddedVector{P,T,L}, ls::BackTracking2{order}, tol = 1e-8) where {P,T,L,order}
    # res = DiffResults.GradientResult(x)
    # ls = BackTracking()
    # order = ordernum(bto)
    # copyto!(state.xinit, x)
    # copyto!(state.x_new, x)
    # xinit = copy(x)
    # x_new = copy(x)
    x_old = get_x_old(state)
    ∇_old = get_∇_old(state)
    invH = get_invH(state)
    δ∇ = get_δ∇(state)
    s = get_s(state)
    u = get_u(state)
    x_new = get_x_new(state)
    copyto!(x_old, x)
    initial_invH!(state)
    # @show x_old
    # hx = SMatrix{P,P,T}(I)
    # if !(hguess isa Nothing)
    #     hx = hguess * hx
    # end
    # hold = copy(hx)
    # jold = copy(x); s = copy(x)
    # @unpack c_1, ρ_hi, ρ_lo, iterations = ls
    c_1, ρ_hi, ρ_lo, iterations = ls.c_1, ls.ρ_hi, ls.ρ_lo, ls.iterations
    iterfinitemax = round(Int,-log2(eps(eltype(x))))
    sqrttol = sqrt(eps(T))
    α_0 = one(T)
    N = 200
    f_calls = 0
    g_calls = 0
    for n = 1:N
        # res = ForwardDiff.gradient!(res, f, x); f_calls +=1; g_calls +=1; # Obtain gradient
        # ∇ = gradient!(d, x_old)
        # # ϕ_0 = DiffResults.value(res)
        # ϕ_0 = value(d)

        ϕ_0 = fdf(obj, x_old); f_calls +=1; g_calls +=1;
        ∇ = gradient(obj)
        # @show ∇, x_old
        # @show maximum(abs, ∇), tol
        isfinite(ϕ_0) || return T(NaN)
        maximum(abs, ∇) < tol && return ϕ_0
        if n > 1 # update hessian
            # y = jx - jold
            # hx = norm(y) < eps(eltype(x)) ? hx : hx + y*y' / (y'*s) - (hx*(s*s')*hx)/(s'*hx*s)


            # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
            # @show L
            # @show length(δ∇), length(∇), length(∇_old)
            # @show full_length(δ∇), full_length(∇), full_length(∇_old)
            # @show typeof(δ∇), typeof(∇), typeof(∇_old)
            @inbounds @simd for i ∈ 1:L
                δ∇[i] =  ∇[i] - ∇_old[i]
            end
            # SIMDArrays.vsub!(δ∇, ∇, ∇_old)
            # Update the inverse Hessian approximation using Sherman-Morrison
            dx_dg = real(dot(s, δ∇))
            # dx_dg == 0.0 && return true # force stop
            mul!(u, invH, δ∇)
            c2 = one(T) / dx_dg
            c1 = fma(real(dot(δ∇, u)), c2*c2, c2)
            BFGS_update!(invH, s, u, c1, c2)
        end
        mul!(s, invH, ∇)
        # SIMDArrays.scale!(s, -one(eltype(s)))
        # SIMDArrays.reflect!(s)
        @inbounds @simd for i ∈ 1:L
            s[i] *= -1
        end
        # s = -hx\jx # Obtain direction
        dϕ_0 = dot(∇, s)

        if dϕ_0 >= zero(T) # If bad, reset search direction
            initial_invH!(state)
            # hx = hold
            # s = -jx
            # @show L, s, ∇
            # @show length(s)
            # @show length(∇)
            # @inbounds @simd for i ∈ 1:L
            #     s[i] = -∇[i]
            # end
            # SIMDArrays.reflect!(s, ∇)
            @inbounds @simd for i ∈ 1:L
                s[i] = - ∇[i]
            end
            dϕ_0 = dot(∇, s)
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        @inbounds @simd for i ∈ 1:L
            x_new[i] = x_old[i] + α_1*s[i]
        end
        # SIMDArrays.vadd!(x_new, x_old, α_1, s)
        # ϕx_1 = f(x + α_1*s); f_calls += 1;
        ϕx_1 = f(obj, state.x_new); f_calls += 1;

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = T(0.5)*α_1
            @inbounds @simd for i ∈ 1:L
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            # ϕx_1 = f(x + α_2*s); f_calls += 1;
            ϕx_1 = f(obj, state.x_new); f_calls += 1;
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            iteration > iterations && linesearch_failure(iterations)

            # Shrink proposed step-size:
            @fastmath if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                α_tmp = - (dϕ_0 * α_2*α_2) / ( T(2) * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            else
                div = one(T) / (α_1*α_1 * α_2*α_2 * (α_2 - α_1))
                a = (α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                b = (-α_1*α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2*α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

                if norm(a) <= eps(Float64) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (T(2)*b)
                else
                    # discriminant
                    d = max(b*b - T(3)*a*dϕ_0, zero(T))
                    # quadratic equation root
                    α_tmp = (sqrt(d) - b) / (T(3)*a)
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
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            ϕx_0, ϕx_1 = ϕx_1, f(obj, state.x_new); f_calls += 1;
        end
        alpha, fpropose = α_2, ϕx_1

        # s = alpha*s
        # x = x + s # Update x
        # @inbounds @simd for i ∈ 1:L
        #     s[i] *= alpha
        #     x_old[i] += s[i]
        # end
        update_state!(s, x_old, alpha)
        # jold = copy(jx)
        copyto!(∇_old, ∇)
    end
    # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), x_old
    NaN
end


@generated function update_state!(C::AbstractFixedSizePaddedArray{S,T,N,R,L}, B::AbstractFixedSizePaddedArray{S,T,N,R,L}, α::T) where {S,T,N,R,L}
    T_size = sizeof(T)
    VL = min(VectorizationBase.REGISTER_SIZE ÷ T_size, L)
    VLT = VL * T_size
    V = Vec{VL,T}

    iter = L ÷ VL
    q = quote
        ptr_C = pointer(C)
        ptr_B = pointer(B)
        vα = vbroadcast($V, α)
    end


    rep, rem = divrem(iter, 4)
    if (rep == 1 && rem == 0) || (rep >= 1 && rem != 0)
        rep -= 1
        rem += 4
    end
    if rep > 0
        push!(q.args,
            quote
                for i ∈ 0:$(4VLT):$(4VLT*(rep-1))
                    vs_0 = vmul(vload($V, ptr_C + i), vα)
                    vstore!(ptr_C + i, vs_0)
                    vstore!(ptr_B + i, vadd(vload($V, ptr_B + i), vs_0))

                    vs_1 = vmul(vload($V, ptr_C + i + $VLT), vα)
                    vstore!(ptr_C + i + $VLT, vs_1)
                    vstore!(ptr_B + i + $VLT, vadd(vload($V, ptr_B + i + $VLT), vs_1))

                    vs_2 = vmul(vload($V, ptr_C + i + $(2VLT)), vα)
                    vstore!(ptr_C + i + $(2VLT), vs_2)
                    vstore!(ptr_B + i + $(2VLT), vadd(vload($V, ptr_B + i + $(2VLT)), vs_2))

                    vs_3 = vmul(vload($V, ptr_C + i + $(3VLT)), vα)
                    vstore!(ptr_C + i + $(3VLT), vs_3)
                    vstore!(ptr_B + i + $(3VLT), vadd(vload($V, ptr_B + i + $(3VLT)), vs_3))
                end
            end
        )
    end
    for i ∈ 0:(rem-1)
        offset = VLT*(i + 4rep)
        push!(q.args,
            quote
                $(Symbol(:vs_,i)) = vmul(vload($V, ptr_C + $offset), vα)
                vstore!(ptr_C + $offset, $(Symbol(:vs_,i)))
                vstore!(ptr_B + $offset, vadd(vload($V, ptr_B + $offset), $(Symbol(:vs_,i))))
            end
        )
    end

    push!(q.args, :(nothing))
    q
end

"""
Similar to optimize_light!, but it scales the function so that
norm(gradient(f, initial_x)) ≈ 10
"""
function optimize_scale!(state, obj, x::AbstractFixedSizePaddedVector{P,T,L}, ls::BackTracking2{order}, scale_target=T(10), tol = T(1e-8)) where {P,T,L,order}
    # res = DiffResults.GradientResult(x)
    # ls = BackTracking()
    # order = ordernum(bto)
    # copyto!(state.xinit, x)
    # copyto!(state.x_new, x)
    # xinit = copy(x)
    # x_new = copy(x)
    x_old = get_x_old(state)
    ∇_old = get_∇_old(state)
    invH = get_invH(state)
    δ∇ = get_δ∇(state)
    s = get_s(state)
    u = get_u(state)
    x_new = get_x_new(state)
    copyto!(x_old, x)
    initial_invH!(state)
    # hx = SMatrix{P,P,T}(I)
    # if !(hguess isa Nothing)
    #     hx = hguess * hx
    # end
    # hold = copy(hx)
    # jold = copy(x); s = copy(x)
    # @unpack c_1, ρ_hi, ρ_lo, iterations = ls
    c_1, ρ_hi, ρ_lo, iterations = ls.c_1, ls.ρ_hi, ls.ρ_lo, ls.iterations
    iterfinitemax = round(Int,-log2(eps(eltype(x))))
    sqrttol = sqrt(eps(T))
    α_0 = one(T)
    N = 200
    f_calls = 0
    g_calls = 0
    local scale::T
    local ϕ_0::T
    for n = 1:N
        # res = ForwardDiff.gradient!(res, f, x); f_calls +=1; g_calls +=1; # Obtain gradient
        # ∇ = gradient!(d, x_old)
        # # ϕ_0 = DiffResults.value(res)
        # ϕ_0 = value(d)

        if n == 1 # calculate scale
            ϕ_0, scale = scale_fdf(obj, x_old, scale_target); f_calls +=1; g_calls +=1;
            # @show scale
            ∇ = gradient(obj)

            isfinite(ϕ_0) || return T(NaN), scale
            maximum(abs, ∇) < tol && return ϕ_0, scale
        else # update hessian
            ϕ_old = ϕ_0
            ϕ_0 = scaled_fdf(obj, x_old, scale); f_calls +=1; g_calls +=1;
            ∇ = gradient(obj)

            norm(ϕ_0 - ϕ_old) <= tol*max(norm(ϕ_0),norm(ϕ_old)) && return ϕ_0, scale
            isfinite(ϕ_0) || return T(NaN), scale
            maximum(abs, ∇) < tol && return ϕ_0, scale
            # y = jx - jold
            # hx = norm(y) < eps(eltype(x)) ? hx : hx + y*y' / (y'*s) - (hx*(s*s')*hx)/(s'*hx*s)


            # invH = invH + c1 * (s * s') - c2 * (u * s' + s * u')
            # @show L
            # @show length(δ∇), length(∇), length(∇_old)
            # @show full_length(δ∇), full_length(∇), full_length(∇_old)
            # @show typeof(δ∇), typeof(∇), typeof(∇_old)
            @inbounds @simd for i ∈ 1:L
                δ∇[i] =  ∇[i] - ∇_old[i]
            end
            # SIMDArrays.vsub!(δ∇, ∇, ∇_old)
            # Update the inverse Hessian approximation using Sherman-Morrison
            dx_dg = real(dot(s, δ∇))
            # dx_dg == 0.0 && return true # force stop
            mul!(u, invH, δ∇)
            c2 = one(T) / dx_dg
            c1 = fma(real(dot(δ∇, u)), c2*c2, c2)
            BFGS_update!(invH, s, u, c1, c2)
        end
        mul!(s, invH, ∇)
        # SIMDArrays.scale!(s, -one(eltype(s)))
        @inbounds @simd for i ∈ 1:L
            s[i] *= -1
        end
        # SIMDArrays.reflect!(s)
        # s = -hx\jx # Obtain direction
        dϕ_0 = dot(∇, s)

        if dϕ_0 >= zero(T) # If bad, reset search direction
            initial_invH!(state)
            # hx = hold
            # s = -jx
            # @show L, s, ∇
            # @show length(s)
            # @show length(∇)
            @inbounds @simd for i ∈ 1:L
                s[i] = -∇[i]
            end
            # SIMDArrays.reflect!(s, ∇)
            dϕ_0 = dot(∇, s)
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        @inbounds @simd for i ∈ 1:L
            x_new[i] = x_old[i] + α_1*s[i]
        end
        # SIMDArrays.vadd!(x_new, x_old, α_1, s)
        # ϕx_1 = f(x + α_1*s); f_calls += 1;
        ϕx_1_before = ϕx_1
        ϕx_1 = f(obj, state.x_new) * scale; f_calls += 1;
        # @show typeof(ϕx_1_before), typeof(ϕx_1), typeof(f(obj, state.x_new)), typeof(scale)

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = T(0.5)*α_1
            @inbounds @simd for i ∈ 1:L
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2, s)
            # ϕx_1 = f(x + α_2*s); f_calls += 1;
            ϕx_1 = f(obj, state.x_new) * scale; f_calls += 1;
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            iteration > iterations && linesearch_failure(iterations)

            # Shrink proposed step-size:
            @fastmath if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                # @show typeof(dϕ_0), typeof(ϕx_1), typeof(ϕ_0)
                α_tmp = - (dϕ_0 * α_2*α_2) / ( T(2) * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            else
                div = one(T) / (α_1*α_1 * α_2*α_2 * (α_2 - α_1))
                a = (α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                b = (-α_1*α_1*α_1*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2*α_2*α_2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                # @show norm(a) <= eps(T) + sqrttol*norm(a)
                if norm(a) <= eps(T) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (T(2)*b)
                else
                    # discriminant
                    d = max(b*b - T(3)*a*dϕ_0, zero(T))
                    # quadratic equation root
                    α_tmp = (sqrt(d) - b) / (T(3)*a)
                end
            end
            α_1 = α_2

            # @show typeof(α_tmp)
            α_tmp = NaNMath.min(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = NaNMath.max(α_tmp, α_2*ρ_lo) # avoid too big reductions
            # α_tmp = min(α_tmp, α_2*ρ_hi) # avoid too small reductions
            # α_2 = max(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            # ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
            @fastmath @inbounds @simd for i ∈ 1:L
                x_new[i] = x_old[i] + α_2*s[i]
            end
            # SIMDArrays.vadd!(x_new, x_old, α_2::T, s)
            ϕx_0, ϕx_1 = ϕx_1, f(obj, state.x_new) * scale; f_calls += 1;
        end
        alpha, fpropose = α_2, ϕx_1

        # s = alpha*s
        # x = x + s # Update x
        # @inbounds @simd for i ∈ 1:L
        #     s[i] *= alpha
        #     x_old[i] += s[i]
        # end
        update_state!(s, x_old, alpha)
        # jold = copy(jx)
        copyto!(∇_old, ∇)
    end
    # return StaticOptimizationResults(NaN, N, tol, f_calls, g_calls, false), x_old
    T(NaN), scale
end
