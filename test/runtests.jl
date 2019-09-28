# using Test

using PaddedMatrices, DifferentiableObjects


function rosenbrock(x::AbstractArray{T}) where T
    out = zero(T)
    @inbounds for i ∈ 2:2:length(x)
        out += 100abs2(abs2(x[i-1]) - x[i]) + abs2(x[i-1] - 1)
    end
    out
end

P = 6

state = DifferentiableObjects.BFGSState2(Val(P));
initial_x = fill!(PaddedMatrices.MutableFixedSizeVector{P,Float64}(undef), 3.2);
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






using PaddedMatrices, DifferentiableObjects
Base.IndexStyle(::PaddedMatrices.AbstractPaddedVector) = IndexLinear()

function rosenbrock(x::AbstractArray{T}) where {T}
    out = zero(T)
    @inbounds for i ∈ 2:2:length(x)
        out += 100abs2(abs2(x[i-1]) - x[i]) + abs2(x[i-1] - 1)
    end
    out
end

P = 6
initial_x = fill!(PaddedMatrices.MutableFixedSizeVector{P,Float64}(undef), 3.2);
obj2 = TwiceDifferentiable(rosenbrock, initial_x);

using ForwardDiff, DiffResults, BenchmarkTools
x = initial_x;
c = obj2.config;

hess = DiffResults.hessian(c.result);
grad = DiffResults.gradient(c.result);
cfg = c.jacobian_config;

# hessian!(obj2.config, initial_x)

@which ForwardDiff.jacobian!(hess, c, grad, x, cfg, Val{false}())

result = hess;
f! = c;
y = grad;
@which ForwardDiff.vector_mode_jacobian!(result, f!, y, x, cfg)


# function vector_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
#     ydual = vector_mode_dual_eval(f!, y, x, cfg)
#     map!(d -> value(T,d), y, ydual)
#     extract_jacobian!(T, result, ydual, N)
#     extract_value!(T, result, y, ydual)
#     return result
# end

T, V, N = ForwardDiff.Tag{typeof(rosenbrock),Float64},Float64,6,Tuple{MutableFixedSizeArray{Tuple{6},ForwardDiff.Dual{ForwardDiff.Tag{typeof(rosenbrock),Float64},Float64,6},1,Tuple{1},6},MutableFixedSizeArray{Tuple{6},ForwardDiff.Dual{ForwardDiff.Tag{typeof(rosenbrock),Float64},Float64,6},1,Tuple{1},6}};

@which ForwardDiff.vector_mode_dual_eval(f!, y, x, cfg)

###
### Here, I can run random benchamrks without a problem.
###

ydual, xdual = cfg.duals;
typeof(ydual), typeof(xdual)

ForwardDiff.seed!(xdual, x, cfg.seeds)
ForwardDiff.seed!(ydual, y)
# seem to be in the clear, based on repeated
@benchmark ForwardDiff.seed!(xdual, x, cfg.seeds)
@benchmark ForwardDiff.seed!(ydual, y)



# Segfaults
# f!(ydual, xdual)


# segfaults
@which ForwardDiff.gradient!(c.inner_result, c.f, xdual, c.gradient_config, Val{false}())
@which ForwardDiff.vector_mode_gradient!(c.inner_result, c.f, xdual, c.gradient_config)
@which ForwardDiff.vector_mode_dual_eval(c.f, xdual, c.gradient_config)


xdualdual = c.gradient_config.duals;
# underlying tuple of length 0!!!
f, x = rosenbrock, initial_x;
result = DifferentiableObjects.HessianDiffResult(x);
chunk = DifferentiableObjects.Chunk(Val{P}())
# tag = ForwardDiff.Tag(f, T)

# here we construct a JacobianConfig so that yduals is a PtrVector to DiffResults.gradient(inner_result)
tag_instance = ForwardDiff.Tag(f, T)
tag_type = DifferentiableObjects.extract_tag(tag_instance)
# @show tag
seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{P,Float64})
# inner_result = GradientDiffResult(jacobian_config.duals[2])
inner_result = DifferentiableObjects.GradientDiffResult{ForwardDiff.Dual{tag_type,Float64,P},P}(undef)
duals = (DiffResults.gradient(inner_result), similar(x, ForwardDiff.Dual{tag_type,Float64,P}))
jacobian_config = ForwardDiff.JacobianConfig{tag_type,Float64,P,typeof(duals)}(seeds, duals)


# jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
# jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag_instance)



# segfault
# @benchmark ForwardDiff.seed!(xdualdual, xdual, c.gradient_config.seeds)
@which ForwardDiff.seed!(xdualdual, xdual, c.gradient_config.seeds)
# xdualdual's tuple is of length 0!!!

c.f(xdual)

# segfaults
# @benchmark ForwardDiff.vector_mode_dual_eval(c.f, xdual, c.gradient_config)

ydualdual = ForwardDiff.vector_mode_dual_eval(c.f, xdual, c.gradient_config)

# Segfaults on call
@which ForwardDiff.extract_gradient!(ForwardDiff.Tag{typeof(rosenbrock),Float64}, c.inner_result, ydualdual)

# @which -> Segfaults!!! Yikes!?!?
v = ForwardDiff.value(ForwardDiff.Tag{typeof(rosenbrock),Float64}, ydualdual)
@benchmark ForwardDiff.value(ForwardDiff.Tag{typeof(rosenbrock),Float64}, ydualdual)

# body of extract_gradient! in:
# extract_gradient!(::Type{T}, result::DiffResults.DiffResult, dual::ForwardDiff.Dual) where T in ForwardDiff at /home/chriselrod/.julia/packages/ForwardDiff/N0wMF/src/gradient.jl:71
inner_result = DiffResults.value!(c.inner_result, ForwardDiff.value(ForwardDiff.Tag{typeof(rosenbrock),Float64}, ydualdual));

partials = ForwardDiff.partials(ForwardDiff.Tag{typeof(rosenbrock),Float64}, ydualdual);
grad_inner = DiffResults.gradient(c.inner_result);


typeof(grad_inner), typeof(partials)

function Base.copyto!()

end

@benchmark copyto!(grad_inner, partials)


@which DiffResults.gradient!(c.inner_result, partials)
# forwards to derivative!(r::Union{DifferentiableObjects.GradientDiffResult{V,P,R}, DifferentiableObjects.HessianDiffResult{V,P,R,L} where L} where R where P where V, x::AbstractArray) in DifferentiableObjects at /home/chriselrod/.julia/dev/DifferentiableObjects/src/forward_diff_differentiable.jl:61
# DifferentiableObjects/src/forward_diff_differentiable.jl:61: DiffResults.derivative!
# 
grad_inner = DiffResults.gradient(c.inner_result);
# @code_warntype copyto!(grad_inner, partials)
@which copyto!(grad_inner, partials)

DiffResults.derivative!(c.inner_result, partials);
# benchmark segfaults

DiffResults.gradient!(c.inner_result, partials);
# segfault


@which copyto!(grad_inner, partials)
copyto!(grad_inner, partials);
# segfault

# segfaults
inner_result = DiffResults.gradient!(c.inner_result, ForwardDiff.partials(ForwardDiff.Tag{typeof(rosenbrock),Float64}, ydualdual))


result = ForwardDiff.extract_gradient!(ForwardDiff.Tag{typeof(rosenbrock),Float64}, c.inner_result, ydualdual)


# Segfaults
ForwardDiff.gradient!(c.inner_result, c.f, xdual, c.gradient_config, Val{false}())

DiffResults.value!(c.result, ForwardDiff.value(DiffResults.value(c.inner_result)))

ydual

ydual = ForwardDiff.vector_mode_dual_eval(f!, y, x, cfg)

map!(d -> ForwardDiff.value(T,d), y, ydual)
ForwardDiff.extract_jacobian!(T, result, ydual, N)
ForwardDiff.extract_value!(T, result, y, ydual)










ForwardDiff.jacobian!(hess, c, grad, x, jconfig, Val{false}())





hessian!(obj2, initial_x)
hessian(obj2)

@which hessian!(obj2, initial_x)
@which hessian(obj2)


using StaticArrays, StaticOptim

sx = @SVector fill(3.2, P);
soptimize(rosenbrock, sx)
@benchmark soptimize(rosenbrock, $sx)
@benchmark soptimize(rosenbrock, $sx, StaticOptim.Order3())

using Optim, LineSearches
xa = fill(3.2, 6);
hz = Optim.BFGS();
backtrack2 = Optim.BFGS(linesearch = LineSearches.BackTracking(order = 2));
backtrack3 = Optim.BFGS(linesearch = LineSearches.BackTracking(order = 3));
@benchmark optimize(rosenbrock, $xa, $hz)
@benchmark optimize(rosenbrock, $xa, $hz; autodiff = :forward)
@benchmark optimize(rosenbrock, $xa, $backtrack2)
@benchmark optimize(rosenbrock, $xa, $backtrack2; autodiff = :forward)
@benchmark optimize(rosenbrock, $xa, $backtrack3)
@benchmark optimize(rosenbrock, $xa, $backtrack3; autodiff = :forward)




# @inline rosenbrock(x) =  (@inbounds out = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2; out)

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
