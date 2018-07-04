using Test, TriangularMatrices, DifferentiableObjects


rosenbrock(x) =  (@inbounds out = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2; out)


initial_x = RecursiveVector{Float64,2}()
initial_x .= 0;
d = OnceDifferentiable(rosenbrock, initial_x)
method = BFGS()
options = LightOptions()

result = optimize_light(d, initial_x, method, options, DifferentiableObjects.initial_state(method, options, d, initial_x))