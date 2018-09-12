# using Test

using SIMDArrays, DifferentiableObjects, StaticArrays


# @inline rosenbrock(x) =  (@inbounds out = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2; out)

function rosenbrock(x::Union{<:SVector{P,T},SizedSIMDVector{P,T}}) where {P,T}
    out = zero(T)
    @inbounds for i ∈ 2:2:P
        out += 100abs2(abs2(x[i-1]) - x[i]) + abs2(x[i-1] - 1)
    end
    out
end

const P = 6

state = DifferentiableObjects.BFGSState2(Val(P));
initial_x = fill(SizedSIMDVector{P,Float64}, 3.2);
ls2 = DifferentiableObjects.BackTracking2(Val(2));
ls3 = DifferentiableObjects.BackTracking2(Val(3));
obj = OnceDifferentiable(rosenbrock, initial_x);
DifferentiableObjects.optimize_light!(state, obj, initial_x, ls2), state.x_old
DifferentiableObjects.optimize_light!(state, obj, initial_x, ls3), state.x_old
DifferentiableObjects.optimize_scale!(state, obj, initial_x, ls2), state.x_old
DifferentiableObjects.optimize_scale!(state, obj, initial_x, ls3), state.x_old

using BenchmarkTools
@benchmark DifferentiableObjects.optimize_light!($state, $obj, $initial_x, $ls2)
@benchmark DifferentiableObjects.optimize_light!($state, $obj, $initial_x, $ls3)
@benchmark DifferentiableObjects.optimize_scale!($state, $obj, $initial_x, $ls2)
@benchmark DifferentiableObjects.optimize_scale!($state, $obj, $initial_x, $ls3)

using StaticArrays, StaticOptim

sx = @SVector fill(3.2, P);
soptimize(rosenbrock, sx)
@benchmark soptimize(rosenbrock, $sx)
@benchmark soptimize(rosenbrock, $sx, StaticOptim.Order3())





# initial_x = zero(SizedSIMDVector{2,Float64});
# d = OnceDifferentiable(rosenbrock, initial_x);
# # method = BFGS(;linesearch = DifferentiableObjects.BackTracking())
# method = BFGS();
# options = LightOptions();
# state = DifferentiableObjects.initial_state(method, options, d, initial_x);
#
# optimum, optimum_val = optimize_light(d, initial_x, method, options, state)
#
#
#
# using BenchmarkTools
# opt_bench = @benchmarkable optimize_light($d, $initial_x, $method, $options, init_state) setup=(init_state=DifferentiableObjects.initial_state($method, $options, $d, $initial_x); $initial_x .= 0)
# run(opt_bench)
#
# using LinearAlgebra, ForwardDiff
# dphi_0 = real(dot(gradient(d), state.s))
# # @show dphi_0
# # reset the direction if it becomes corrupted
# if dphi_0 >= zero(dphi_0) && DifferentiableObjects.reset_search_direction!(state, d, method)
#     dphi_0 = real(dot(gradient(d), state.s)) # update after direction reset
# end
# phi_0  = value(d)
# obj, x = d, state.x
#
# @benchmark DifferentiableObjects.update_g!($d, $state, $method)
# @benchmark DifferentiableObjects.value_gradient!($d, $state.x)
# @benchmark DifferentiableObjects.fdf($d, $state.x)
# @code_warntype DifferentiableObjects.fdf(d, state.x)
#
# using Profile
# @profile optimize_light(d, initial_x, method, options, state)
# Profile.print()
#
#
#
# @benchmark DifferentiableObjects.linesearch!($method.linesearch!, $d, $state.x, $state.s, $state.alpha, $state.x_ls, $phi_0, $dphi_0)
# @code_warntype DifferentiableObjects.linesearch!(method.linesearch!, d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)
#
# @benchmark $method.linesearch!($d, $state.x, $state.s, $state.alpha, $state.x_ls, $phi_0, $dphi_0)
# @code_warntype method.linesearch!(d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)
#
#
# @show optimum
# @test all(i -> optimum[i] ≈ 1, 1:length(optimum))
#
#
# using StaticArrays, StaticOptim
#
# sx = @SVector zeros(P)
# soptimize(rosenbrock, sx)
# @benchmark soptimize(rosenbrock, $sx)
