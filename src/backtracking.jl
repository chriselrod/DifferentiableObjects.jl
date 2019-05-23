# This code was originally taken and modifed from LineSearches.jl for the sake of
# improving performance. Attribution / credit belongs to the original authors.
# This was the attached liscense agreement:
#
# The original code has been separated out from the Optim.jl package,
# licensed under the MIT License:
# > Copyright (c) 2012: John Myles White and other contributors.
#
# The LineSearches.jl package is licensed under the MIT "Expat" License:
#
# > Copyright (c) 2016: Asbjørn Nilsen Riseth and other contributors.
# >
# > Permission is hereby granted, free of charge, to any person obtaining
# > a copy of this software and associated documentation files (the
# > "Software"), to deal in the Software without restriction, including
# > without limitation the rights to use, copy, modify, merge, publish,
# > distribute, sublicense, and/or sell copies of the Software, and to
# > permit persons to whom the Software is furnished to do so, subject to
# > the following conditions:
# >
# > The above copyright notice and this permission notice shall be
# > included in all copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# > EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# > MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# > IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# > CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# > TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# > SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




@with_kw struct BackTracking{TF, TI}
    c_1::TF = 1e-4
    ρ_hi::TF = 0.5
    ρ_lo::TF = 0.1
    iterations::TI = 1_000
    order::TI = 3
    maxstep::TF = Inf
end
BackTracking{TF}(args...; kwargs...) where TF = BackTracking{TF,Int}(args...; kwargs...)

function linesearch!(ls::BackTracking, df, x::AbstractArray{T}, s::AbstractArray{T},
                            α_0::Tα, x_new::AbstractArray{T}, ϕ_0, dϕ_0) where {T, Tα}

    alphamax = convert(real(T), Inf)
    # ϕ, dϕ = make_ϕ_dϕ(df, x_new, x, s)



    # α_0 = min(α_0, min(alphamax, ls.maxstep / norm(s, Inf)))
    α_0 = min(α_0, min(alphamax, ls.maxstep / SIMDArrays.maximum_abs(s)))
    # ls(ϕ, α_0, ϕ_0, dϕ_0)


    c_1, ρ_hi, ρ_lo, iterations, order = ls.c_1, ls.ρ_hi, ls.ρ_lo, ls.iterations, ls.order
    # @unpack c_1, ρ_hi, ρ_lo, iterations, order = ls

    iterfinitemax = -log2(eps(real(Tα)))

    # @assert order in (2,3)
    # Check the input is valid, and modify otherwise
    #backtrack_condition = 1.0 - 1.0/(2*ρ) # want guaranteed backtrack factor
    #if c_1 >= backtrack_condition
    #    warn("""The Armijo constant c_1 is too large; replacing it with
    #                   $(backtrack_condition)""")
    #   c_1 = backtrack_condition
    #end

    # Count the total number of iterations
    iteration = 0

    ϕx_0, ϕx_1 = ϕ_0, ϕ_0

    α_1, α_2 = α_0, α_0

    @fastmath @inbounds @simd for i ∈ 1:full_length(x_new)
        x_new[i] = x[i] + α_1 * s[i]
    end
    # Evaluate f(x+α*s)

    ϕx_1 = value!(df, x_new)
    # @show ϕx_1, α_1

    # Hard-coded backtrack until we find a finite function value
    iterfinite = 0
    while !isfinite(ϕx_1) && iterfinite < iterfinitemax
        iterfinite += 1
        α_1 = α_2
        α_2 = 0.5α_1

        @fastmath @inbounds @simd for i ∈ 1:full_length(x_new)
            x_new[i] = x[i] + α_2 * s[i]
        end
        # Evaluate f(x+α*s)

        ϕx_1 = value!(df, x_new)
        # @show ϕx_1, α_2
    end

    # Backtrack until we satisfy sufficient decrease condition
    while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        # if iteration > iterations
        #     throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(iterations).",
        #                               α_2))=
        # end
        # iteration > iterations && throw("Exceeded maximum number of iterations.")
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
            div = one(Tα) / (α_1^2 * α_2^2 * (α_2 - α_1))
            a = (α_1^2*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2^2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
            b = (-α_1^3*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2^3*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

            if isapprox(a, zero(a), atol=eps(real(Tα)))
                α_tmp = dϕ_0 / (2*b)
            else
                # discriminant
                d = max(b^2 - 3*a*dϕ_0, Tα(0))
                # quadratic equation root
                @fastmath α_tmp = (-b + sqrt(d)) / (3*a)
            end
        end

        α_1 = α_2

        # α_tmp = NaNMath.min(α_tmp, α_2*ρ_hi) # avoid too small reductions
        # α_2 = NaNMath.max(α_tmp, α_2*ρ_lo) # avoid too big reductions
        α_tmp = min(α_tmp, α_2*ρ_hi) # avoid too small reductions
        α_2 = max(α_tmp, α_2*ρ_lo) # avoid too big reductions

        @fastmath @inbounds @simd for i ∈ 1:full_length(x_new)
            x_new[i] = x[i] + α_2 * s[i]
        end

        # Evaluate f(x) at proposed position
        ϕx_0, ϕx_1 = ϕx_1, value!(df, x_new)
        # @show ϕx_1, α_2
    end

    return α_2, ϕx_1
end
