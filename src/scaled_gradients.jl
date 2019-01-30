#
# Here, we aim to slightly more efficiently calculate gradients that may be
# scaled to behave better under optimization algorithms.
#

function scale_gradient!(result::GradResult{V,P,G}, f::F, x::AbstractArray, scale_target,
        cfg::ForwardDiff.GradientConfig{T} = ForwardDiff.GradientConfig(f, x), ::Val{CHK}=Val{true}()) where {T, CHK, F, V,P,L,G<:SizedSIMDVector{P,V,L,L}}
    CHK && ForwardDiff.checktag(T, f, x)
    if ForwardDiff.chunksize(cfg) == length(x)
        ForwardDiff.vector_mode_gradient!(result, f, x, cfg)
    else
        ForwardDiff.chunk_mode_gradient!(result, f, x, cfg)
    end
    ∇ = DiffResults.gradient(result)
    # Need to track down how the buffer becomes NaN in the first place
    # Insert check before multiplication??
    @inbounds for i ∈ P+1:L
        ∇[i] = zero(V)
    end
    # Do not scale up the gradient.
    scale = min(one(V), scale_target / norm(∇))
    result.value *= scale
    SIMDArrays.scale!(∇, scale)
    return scale
end
function scaled_gradient!(result::GradResult, f::F, x::AbstractArray, scale,
        cfg::ForwardDiff.GradientConfig{T} = ForwardDiff.GradientConfig(f, x), ::Val{CHK}=Val{true}()) where {T, CHK, F}
    CHK && ForwardDiff.checktag(T, f, x)
    if ForwardDiff.chunksize(cfg) == length(x)
        scaled_vector_mode_gradient!(result, f, x, scale, cfg)
    else
        scaled_chunk_mode_gradient!(result, f, x, scale, cfg)
    end
    return result
end

function scaled_vector_mode_gradient!(result, f::F, x, scale, cfg::ForwardDiff.GradientConfig{T}) where {T, F}
    ydual = ForwardDiff.vector_mode_dual_eval(f, x, cfg) * scale
    result = ForwardDiff.extract_gradient!(T, result, ydual)
    return result
end

# function extract_gradient!(::Type{T}, result::DiffResult, dual::Dual) where {T}
#     result = DiffResults.value!(result, value(T, dual))
#     result = DiffResults.gradient!(result, partials(T, dual))
#     return result
# end

function chunk_mode_gradient_expr(N, xlen, ydual_expr)

    remainder = xlen % N
    lastchunksize = ifelse(remainder == 0, N, remainder)
    # precalculate loop bounds
    lastchunkindex = xlen - lastchunksize + 1
    middlechunks = 2:div(xlen - lastchunksize, N)
    quote


        # seed work vectors
        xdual = cfg.duals
        seeds = cfg.seeds
        ForwardDiff.seed!(xdual, x)

        # do first chunk manually to calculate output type
        ForwardDiff.seed!(xdual, x, 1, seeds)
        $(ydual_expr)

        ForwardDiff.extract_gradient_chunk!(T, result, ydual, 1, $N)
        ForwardDiff.seed!(xdual, x, 1)

        # do middle chunks
        for c in $middlechunks
            i = ((c - 1) * N + 1)
            ForwardDiff.seed!(xdual, x, i, seeds)
            $(ydual_expr)
            ForwardDiff.extract_gradient_chunk!(T, result, ydual, i, $N)
            ForwardDiff.seed!(xdual, x, i)
        end

        # do final chunk
        ForwardDiff.seed!(xdual, x, $lastchunkindex, seeds, $lastchunksize)
        $(ydual_expr)
        ForwardDiff.extract_gradient_chunk!(T, result, ydual, $lastchunkindex, $lastchunksize)

        # get the value, this is a no-op unless result is a DiffResult
        ForwardDiff.extract_value!(T, result, ydual)

        return result
    end
end


## method defined for better performance of regular chunk mode gradients.
@generated function ForwardDiff.chunk_mode_gradient!(result::GradResult{V,P}, f::F, x,
                                            cfg::ForwardDiff.GradientConfig{T,V,N}) where {F,V,T,N,P}
    chunk_mode_gradient_expr(N, P, :(ydual = f(xdual)))
end
@generated function scaled_chunk_mode_gradient!(result::GradResult{V,P}, f::F, x,
                                            scale, cfg::ForwardDiff.GradientConfig{T,V,N}) where {F,T,V,N,P}
    chunk_mode_gradient_expr(N, P, :(ydual = f(xdual) * scale))
end










#
# function DiffResults.derivative!(r::GradResult, x::AbstractArray)
#     copyto!(r.grad, x)
#     return r
# end
# function DiffResults.derivative!(r::GradResult, x::AbstractArray, ::Type{Val{1}})
#     copyto!(r.grad, x)
#     return r
# end
# function DiffResults.derivative!(r::HessianDiffResult, x::AbstractArray, ::Type{Val{2}})
#     copyto!(r.hess, x)
#     return r
# end
#
# @inline function hessian!(c::HessianConfiguration, x::AbstractArray)
#     ForwardDiff.jacobian!(DiffResults.hessian(c.result), c, DiffResults.gradient(c.result), x, c.jacobian_config, Val{false}())
#     # DiffResults.hessian(c.result)
#     c.result.hess
# end
#
# function jacobian!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {T,CHK}
#     CHK && checktag(T, f!, x)
#     if chunksize(cfg) == length(x)
#         vector_mode_jacobian!(result, f!, y, x, cfg)
#     else
#         chunk_mode_jacobian!(result, f!, y, x, cfg)
#     end
#     return result
# end
# function vector_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
#     ydual = vector_mode_dual_eval(f!, y, x, cfg)
#     map!(d -> value(T,d), y, ydual)
#     extract_jacobian!(T, result, ydual, N)
#     extract_value!(T, result, y, ydual)
#     return result
# end
#
#
# @generated function chunk_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
#     jacobian_chunk_mode_expr(quote
#                                    ydual, xdual = cfg.duals
#                                    seed!(xdual, x)
#                                end,
#                                :(f!(seed!(ydual, y), xdual)),
#                                :(),
#                                :(extract_value!(T, result, y, ydual))))
# end
