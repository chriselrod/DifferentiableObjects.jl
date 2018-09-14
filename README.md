The current focus of this library is on simulations of relatively small dimensional problems. The scope is likely to increase when I begin working on larger dimensional problems.

The current autodiff backend is `ForwardDiff.jl`. I am excited about adding support for `Zygote.jl` and `Capstan.jl` as soon as they have tagged releases, or perhaps sooner.


The target is applications that require repeated optimizations. Thus the API is a little unwieldy at the momeny, forcing you to manually preallocate all the memory needed. Example use:

```julia
julia> using SIMDArrays, DifferentiableObjects, StaticArrays

julia> function rosenbrock(x::AbstractArray{T}) where T
           out = zero(T)
           @inbounds for i ∈ 2:2:length(x)
               out += 100abs2(abs2(x[i-1]) - x[i]) + abs2(x[i-1] - 1)
           end
           out
       end
rosenbrock (generic function with 1 method)

julia> const P = 6
6

julia> state = DifferentiableObjects.BFGSState2(Val(P));

julia> initial_x = fill(SizedSIMDVector{P,Float64}, 3.2);

julia> ls2 = DifferentiableObjects.BackTracking2(Val(2));

julia> ls3 = DifferentiableObjects.BackTracking2(Val(3));

julia> obj = OnceDifferentiable(rosenbrock, initial_x);

julia> DifferentiableObjects.optimize_light!(state, obj, initial_x, ls2), state.x_old
(4.837813336258326e-21, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

julia> DifferentiableObjects.optimize_light!(state, obj, initial_x, ls3), state.x_old
(7.182559121415813e-21, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

julia> DifferentiableObjects.optimize_scale!(state, obj, initial_x, ls2), state.x_old
((6.289355761132598e-20, 0.0006327206696054021), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

julia> DifferentiableObjects.optimize_scale!(state, obj, initial_x, ls3), state.x_old
((6.289355761132598e-20, 0.0006327206696054021), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
```
The scaled version automatically scales the objective so that the norm of the gradient at the first evaluation is 1.

This library aims to be fast. For the 6 dimensional Rosenbrock:
```julia
julia> using BenchmarkTools

julia> @benchmark DifferentiableObjects.optimize_light!($state, $obj, $initial_x, $ls2)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     8.977 μs (0.00% GC)
  median time:      9.020 μs (0.00% GC)
  mean time:        9.164 μs (0.00% GC)
  maximum time:     19.943 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     3

julia> @benchmark DifferentiableObjects.optimize_light!($state, $obj, $initial_x, $ls3)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     15.681 μs (0.00% GC)
  median time:      15.741 μs (0.00% GC)
  mean time:        15.938 μs (0.00% GC)
  maximum time:     29.434 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark DifferentiableObjects.optimize_scale!($state, $obj, $initial_x, $ls2)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     8.740 μs (0.00% GC)
  median time:      8.773 μs (0.00% GC)
  mean time:        8.924 μs (0.00% GC)
  maximum time:     20.256 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     3

julia> @benchmark DifferentiableObjects.optimize_scale!($state, $obj, $initial_x, $ls3)
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     8.487 μs (0.00% GC)
  median time:      8.529 μs (0.00% GC)
  mean time:        8.592 μs (0.00% GC)
  maximum time:     19.046 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     3
```

For comparison, here is `Optim.jl`, Julia's premier optimization library:
```julia
julia> xa = fill(3.2, 6);

julia> hz = Optim.BFGS();

julia> backtrack2 = Optim.BFGS(linesearch = LineSearches.BackTracking(order = 2));

julia> backtrack3 = Optim.BFGS(linesearch = LineSearches.BackTracking(order = 3));

julia> @benchmark optimize(rosenbrock, $xa, $hz)
BenchmarkTools.Trial: 
  memory estimate:  56.05 KiB
  allocs estimate:  1387
  --------------
  minimum time:     121.565 μs (0.00% GC)
  median time:      127.300 μs (0.00% GC)
  mean time:        142.322 μs (8.95% GC)
  maximum time:     55.663 ms (99.68% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark optimize(rosenbrock, $xa, $hz; autodiff = :forward)
BenchmarkTools.Trial: 
  memory estimate:  72.55 KiB
  allocs estimate:  1842
  --------------
  minimum time:     114.016 μs (0.00% GC)
  median time:      120.061 μs (0.00% GC)
  mean time:        143.228 μs (11.14% GC)
  maximum time:     56.916 ms (99.69% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark optimize(rosenbrock, $xa, $backtrack2)
BenchmarkTools.Trial: 
  memory estimate:  14.42 KiB
  allocs estimate:  478
  --------------
  minimum time:     63.710 μs (0.00% GC)
  median time:      67.307 μs (0.00% GC)
  mean time:        77.963 μs (9.80% GC)
  maximum time:     55.385 ms (99.82% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark optimize(rosenbrock, $xa, $backtrack2; autodiff = :forward)
BenchmarkTools.Trial: 
  memory estimate:  18.47 KiB
  allocs estimate:  529
  --------------
  minimum time:     62.236 μs (0.00% GC)
  median time:      65.865 μs (0.00% GC)
  mean time:        77.376 μs (10.97% GC)
  maximum time:     59.504 ms (99.81% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark optimize(rosenbrock, $xa, $backtrack3)
BenchmarkTools.Trial: 
  memory estimate:  21.19 KiB
  allocs estimate:  812
  --------------
  minimum time:     102.088 μs (0.00% GC)
  median time:      107.848 μs (0.00% GC)
  mean time:        120.182 μs (7.05% GC)
  maximum time:     57.059 ms (99.72% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark optimize(rosenbrock, $xa, $backtrack3; autodiff = :forward)
BenchmarkTools.Trial: 
  memory estimate:  26.58 KiB
  allocs estimate:  838
  --------------
  minimum time:     96.801 μs (0.00% GC)
  median time:      103.685 μs (0.00% GC)
  mean time:        117.980 μs (7.66% GC)
  maximum time:     55.781 ms (99.75% GC)
  --------------
  samples:          10000
  evals/sample:     1
```
DifferentiableObjects is over 7 times faster when comparing best times, and doesn't suffer the slow downs and noise caused by a triggered garbage collector.


Finally, `StaticOptim.jl` is an effort at fast, small dimensional optimization. It is supposed to avoid allocations via using stack allocated rather than heap allocated arrays. However, even for 6 dimensional problems, it starts running into issues. I haven't dove into it, but I suspect a solve method in the BFGS update state hasn't been implemented for `StaticArrays` beyond a certain size, forcing it to fall back to a generic method.
```julia
julia> using StaticArrays, StaticOptim

julia> sx = @SVector fill(3.2, P);

julia> soptimize(rosenbrock, sx)
Results of Static Optimization Algorithm
 * Initial guess: [3.2,3.2,3.2,3.2,3.2,3.2]
 * Minimizer: [1.0000000000241218,1.0000000000499618,0.9999999999733606,0.9999999999462557,0.9999999999674698,0.9999999999366639]
 * Minimum: [2.963943558125197e-21]
 * Hf(x): [767.2723368096578,-383.5927829099585,16.429281168983145,-8.952240892461559,11.458606255937537,-6.458340882131898,-383.59278290997,192.27464228435383,-7.439709736469863,4.079222810294128,-6.050831862108683,3.3792683249077182,16.429281168895386,-7.439709736424559,791.0050687676064,-393.79457637541947,47.44792400688988,-22.820811525527454,-8.952240892415649,4.079222810270448,-393.7945763754243,196.54049462906383,-24.36254089525885,11.72507793010618,11.458606255773256,-6.050831862026001,47.447924006637365,-24.36254089512562,728.5355553491363,-363.8616557113817,-6.45834088204949,3.3792683248663153,-22.820811525412417,11.72507793004528,-363.8616557113952,182.22283851703085]
 * Number of iterations: [58]
 * Number of function calls: [132]
 * Number of gradient calls: [58]
 * Converged: [true]


julia> @benchmark soptimize(rosenbrock, $sx)
BenchmarkTools.Trial: 
  memory estimate:  10.47 KiB
  allocs estimate:  292
  --------------
  minimum time:     40.137 μs (0.00% GC)
  median time:      40.846 μs (0.00% GC)
  mean time:        49.019 μs (13.15% GC)
  maximum time:     59.951 ms (99.92% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark soptimize(rosenbrock, $sx, StaticOptim.Order3())
BenchmarkTools.Trial: 
  memory estimate:  9.97 KiB
  allocs estimate:  305
  --------------
  minimum time:     68.971 μs (0.00% GC)
  median time:      69.591 μs (0.00% GC)
  mean time:        78.581 μs (10.15% GC)
  maximum time:     60.726 ms (99.84% GC)
  --------------
  samples:          10000
  evals/sample:     1
```
Another issue is that `StaticArrays.jl` results in much longer compile times than `SIMDArrays.jl`, although the former has more general support. I wrote the latter, so it does have support in the places where I needed it, but not much beyond that.


