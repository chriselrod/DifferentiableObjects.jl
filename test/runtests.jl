using Test

using SIMDArrays, DifferentiableObjects


@inline rosenbrock(x) =  (@inbounds out = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2; out)


state = DifferentiableObjects.BFGSState2(Val(2));
initial_x = zero(SizedSIMDVector{2,Float64});
ls = DifferentiableObjects.BackTracking2();
obj = OnceDifferentiable(rosenbrock, initial_x);
DifferentiableObjects.optimize_light!(state, obj, initial_x, ls)

using BenchmarkTools
@benchmark DifferentiableObjects.optimize_light!($state, $obj, $initial_x, $ls)


initial_x = zero(SizedSIMDVector{2,Float64});
d = OnceDifferentiable(rosenbrock, initial_x);
# method = BFGS(;linesearch = DifferentiableObjects.BackTracking())
method = BFGS();
options = LightOptions();
state = DifferentiableObjects.initial_state(method, options, d, initial_x);

optimum, optimum_val = optimize_light(d, initial_x, method, options, state)



using BenchmarkTools
opt_bench = @benchmarkable optimize_light($d, $initial_x, $method, $options, init_state) setup=(init_state=DifferentiableObjects.initial_state($method, $options, $d, $initial_x); $initial_x .= 0)
run(opt_bench)

using LinearAlgebra, ForwardDiff
dphi_0 = real(dot(gradient(d), state.s))
# @show dphi_0
# reset the direction if it becomes corrupted
if dphi_0 >= zero(dphi_0) && DifferentiableObjects.reset_search_direction!(state, d, method)
    dphi_0 = real(dot(gradient(d), state.s)) # update after direction reset
end
phi_0  = value(d)
obj, x = d, state.x

@benchmark DifferentiableObjects.update_g!($d, $state, $method)
@benchmark DifferentiableObjects.value_gradient!($d, $state.x)
@benchmark DifferentiableObjects.fdf($d, $state.x)
@code_warntype DifferentiableObjects.fdf(d, state.x)

using Profile
@profile optimize_light(d, initial_x, method, options, state)
Profile.print()



@benchmark DifferentiableObjects.linesearch!($method.linesearch!, $d, $state.x, $state.s, $state.alpha, $state.x_ls, $phi_0, $dphi_0)
@code_warntype DifferentiableObjects.linesearch!(method.linesearch!, d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)

@benchmark $method.linesearch!($d, $state.x, $state.s, $state.alpha, $state.x_ls, $phi_0, $dphi_0)
@code_warntype method.linesearch!(d, state.x, state.s, state.alpha, state.x_ls, phi_0, dphi_0)


@show optimum
@test all(i -> optimum[i] ≈ 1, 1:length(optimum))


using StaticArrays, StaticOptim

sx = @SVector zeros(2)
rosenbrock(x) =  (@inbounds out = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2; out)
soptimize(rosenbrock, sx)
