


# This needs actual performance testing.
@generated function Chunk(::Val{N}) where N
    if 4N <= VectorizationBase.REGISTER_SIZE
        P = N
    else
        p1 = round(Int, 4N / VectorizationBase.REGISTER_SIZE, RoundUp)
        P = round(Int, N / p1, RoundUp)
    end
    ForwardDiff.Chunk{P}()
end
@generated ValP1(::Val{N}) where N = Val{N+1}()
@generated ValM1(::Val{N}) where N = Val{N-1}()


mutable struct GradientDiffResult{V,P,R} <: DiffResults.DiffResult{1,V,Tuple{ConstantFixedSizeVector{P,V,R,R}}}
    grad::ConstantFixedSizeVector{P,V,R,R}
    value::V
    function GradientDiffResult{V,P,R}(::UndefInitializer) where {V, P, R}
        new{V,P,R}()
    end
    @generated function GradientDiffResult{V,P}(::UndefInitializer) where {V, P}
        R = PaddedMatrices.calc_padding(P, V)
        :(GradientDiffResult{$V,$P,$R}(undef))
    end
    function GradientDiffResult(g::ConstantFixedSizeVector{P,V,R,R}, v::V = zero(V)) where {P,V,R}
        GradientDiffResult{P,V,R}(g, v)
    end
end
mutable struct HessianDiffResult{V,P,R,L} <: DiffResults.DiffResult{2,V,Tuple{ConstantFixedSizeVector{P,V,R,R},ConstantFixedSizeMatrix{P,P,V,R,L}}}
    grad::ConstantFixedSizeVector{P,V,R,R}
    hess::ConstantFixedSizeMatrix{P,P,V,R,L}
    value::V
    function HessianDiffResult{V,P,R,L}(::UndefInitializer) where {V,P,R,L}
        new{V,P,R,L}()
    end
    @generated function HessianDiffResult{V,P}(::UndefInitializer) where {V, P}
        N,R,L = PaddedMatrices.calc_NPL((P,P), V)
        :(HessianDiffResult{$V,$P,$R,$L}(undef))
    end
    function GradientDiffResult(g::ConstantFixedSizeVector{P,V,R,R}, H::ConstantFixedSizeMatrix{P,P,V,R,L}, v::V = zero(V)) where {P,V,R,L}
        GradientDiffResult{P,V,R,R}(g, H, v)
    end
end
const GradResult{V,P,R} = Union{GradientDiffResult{V,P,R},HessianDiffResult{V,P,R}}
@inline Base.pointer(result::GradResult{V}) where {V} = Base.unsafe_convert(Ptr{V}, pointer_from_objref(result))
@inline DiffResults.derivative(result::GradResult{V,P,R}) where {V,P,R} = PtrVector{P,V,R}(pointer(result))
@inline DiffResults.derivative(result::GradResult{V,P,R}, ::Type{Val{1}}) where {V,P,R} = PtrVector{P,V,R}(pointer(result))
@inline DiffResults.derivative(result::HessianDiffResult{V,P,R}, ::Type{Val{2}}) where {V,P,R} = PtrVector{P,V,R}(pointer(result) + sizeof(V)*R )
@inline DiffResults.gradient(result::GradResult{V,P,R}) where {V,P,R} = PtrVector{P,V,R}(pointer(result))
@inline DiffResults.hessian(result::HessianDiffResult{V,P,R})  where {V,P,R}= PtrMatrix{P,P,V,R}(pointer(result) + sizeof(V)*R )

@inline DiffResults.value!(r::GradResult, x::Number) = (r.value = x; return r)
# @inline DiffResults.value!(r::GradResult, x::AbstractArray) = (copyto!(value(r), x); return r)
@inline DiffResults.value!(f::F, r::GradResult, x::Number) where {F} = (r.value = f(x); return r)
# @inline DiffResults.value!(f, r::GradResult, x::AbstractArray) = (map!(f, value(r), x); return r)

function DiffResults.derivative!(r::GradResult, x::AbstractArray)
    copyto!(DiffResults.gradient(r), x)
    return r
end
function DiffResults.derivative!(r::GradResult, x::AbstractArray, ::Type{Val{1}})
    copyto!(DiffResults.gradient(r), x)
    return r
end
function DiffResults.derivative!(r::HessianDiffResult, x::AbstractArray, ::Type{Val{2}})
    copyto!(DiffResults.hessian(r), x)
    return r
end
function DiffResults.derivative!(r::GradResult{P,V,R}, x::ConstantFixedSizeVector{P,V,R}) where {P,V,R}
    r.grad = x
    return r
end
function DiffResults.derivative!(r::GradResult{P,V,R}, x::ConstantFixedSizeVector{P,V,R}, ::Type{Val{1}}) where {P,V,R}
    r.grad = x
    return r
end
function DiffResults.derivative!(r::HessianDiffResult{P,V,R,L}, x::ConstantFixedSizeMatrix{P,P,V,R,L}, ::Type{Val{2}}) where {P,V,R,L}
    r.hess = x
    return r
end
# function DiffResults.derivative!(f, r::GradResult, x::Number)
#     r.grad = f(x)
#     return r
# end
# function DiffResults.derivative!(f, r::GradResult, x::AbstractArray)
#     map!(f, r.grad, x)
#     return r
# end


function GradientDiffResult(x::SizedVector{P,T}) where {T,P}
    GradientDiffResult{T,P}(undef)
end
function HessianDiffResult(x::SizedVector{P,T}) where {T,P}
    HessianDiffResult{T,P}(undef)
    # HessianDiffResult(zero(T), similar(x), MutableFixedSizeMatrix{P,P,T}(undef))
end
# function HessianDiffResult(x::MVector{P,T}) where {T,P}
#     HessianDiffResult{T,P}(undef)
#     # HessianDiffResult(zero(T), similar(x), MMatrix{P,P,T}(undef))
# end

abstract type AbstractAutoDiffDifferentiable{P,T} <: AbstractDifferentiableObject{P,T} end
abstract type AbstractOnceDifferentiableObject{P,T} <: AbstractAutoDiffDifferentiable{P,T} end
abstract type AbstractTwiceDifferentiableObject{P,T} <: AbstractAutoDiffDifferentiable{P,T} end
abstract type Configuration{P,V,F} end

struct GradientConfiguration{P,V,F,T,ND,DG,R} <: Configuration{P,V,F}
    f::F
    result::GradientDiffResult{V,P,R}
    gconfig::ForwardDiff.GradientConfig{T,V,ND,DG}
end

struct HessianConfiguration{P,V,F,T,T2,ND,DJ,DG,DG2, R_V, R_D, LSQM} <: Configuration{P,V,F}
    f::F
    result::HessianDiffResult{V,P,R_V,LSQM}
    inner_result::GradientDiffResult{ForwardDiff.Dual{T,V,ND},P,R_D}
    jacobian_config::ForwardDiff.JacobianConfig{T,V,ND,DJ}
    gradient_config::ForwardDiff.GradientConfig{T,ForwardDiff.Dual{T,V,ND},ND,DG}
    gconfig::ForwardDiff.GradientConfig{T2,V,ND,DG2}
end


struct TwiceDifferentiable{P,T,F,C<:Configuration{P,T,F}, SV_V <: SizedVector{P,T}} <: AbstractTwiceDifferentiableObject{P,T}
    x_f::SV_V # x used to evaluate f (stored in F)
    x_df::SV_V # x used to evaluate df (stored in DF)
    x_h::SV_V #??
    config::C
end
struct OnceDifferentiable{P,T,F,C<:GradientConfiguration{P,T,F}, SV_V <: SizedVector{P,T}} <: AbstractOnceDifferentiableObject{P,T}
    x_f::SV_V# x used to evaluate f (stored in F)
    x_df::SV_V # x used to evaluate df (stored in DF)
    config::C
end


function GradientConfiguration(f::F, x::SizedVector{P,T}) where {F,T,P}
    result = GradientDiffResult(x)
    chunk = Chunk(Val{P}())
    tag = ForwardDiff.Tag(f, T)
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)

    GradientConfiguration(f, result, gconfig)
end
extract_tag(::T) where {T} = T
function HessianConfiguration(f::F, x::SizedVector{P,T}) where {F,T,P}
    result = HessianDiffResult(x)
    chunk = Chunk(Val{P}())
    # tag = ForwardDiff.Tag(f, T)

    # here we construct a JacobianConfig so that yduals is a PtrVector to DiffResults.gradient(inner_result)
    tag_instance = ForwardDiff.Tag(f, T)
    tag_type = extract_tag(tag_instance)
    # @show tag
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{P,T})
    # inner_result = GradientDiffResult(jacobian_config.duals[2])
    inner_result = GradientDiffResult{ForwardDiff.Dual{tag_type,T,P},P}(undef)
    duals = (DiffResults.gradient(inner_result), similar(x, ForwardDiff.Dual{tag_type,T,P}))
    jacobian_config = ForwardDiff.JacobianConfig{tag_type,T,P,typeof(duals)}(seeds, duals)


    # jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    # jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag_instance)
    
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag_instance)

    HessianConfiguration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
end

TwiceDifferentiable(f::F, ::Val{P}) where {F,P} = TwiceDifferentiable(f, MutableFixedSizeVector{P,Float64}(undef))
function TwiceDifferentiable(f::F, x::SV_V) where {F,T,P,SV_V <: SizedVector{P,T}}
    TwiceDifferentiable(x, similar(x), similar(x), HessianConfiguration(f, x))
end
function TwiceDifferentiable(x_f::SV_V,x_df::SV_V,x_h::SV_V,config::C) where {T,F,C<:Configuration{T,F},P,SV_V <: SizedVector{P,T}}
    TwiceDifferentiable{P,T,F,C}(x_f, x_df, x_h, config)
end



function OnceDifferentiable(f, x_f::SizedVector{P,T}) where {T,P}
    OnceDifferentiable(x_f, similar(x_f), GradientConfiguration(f, x_f))
end
function OnceDifferentiable(x_f::SV_V, config::C) where {T,F,C<:GradientConfiguration{T,F},P,SV_V <: SizedVector{P,T}}
    OnceDifferentiable{P,T,F,C}(x_f, similar(x_f), config)
end

# ProfileDifferentiable(f::F, ::Val{N}) where {F,N} = ProfileDifferentiable(f, Vector{Float64}(undef, N), Val{N}())
# function ProfileDifferentiable(f::F, x::AbstractArray{T}, ::Val{N}) where {F,T,N}
#     x_f = Vector{T}(undef, N-1)
#     ProfileDifferentiable(x_f, similar(x_f), similar(x_f), Configuration(f, x_f, ValM1(Val{N}())), x, Ref(N), Ref{T}(),::Val{N})
# end
# function ProfileDifferentiable(x_f::Vector{T},x_df::Vector{T},x_h::Vector{T}, config::C, x,::A i::RefValue{Int}, v::RefValue{T},::Val{N}) where {T,A<:AbstractArray{T},C,N}
#     ProfileDifferentiable{N,T,A,C}(x_f, x_df, x_h, config, x, i, v)
# end


@inline DiffResults.value!(obj::AbstractAutoDiffDifferentiable, x::Real) = DiffResults.value!(obj.config.result, x)

@inline value(obj::AbstractAutoDiffDifferentiable) = obj.config.result.value
@inline gradient(obj::AbstractAutoDiffDifferentiable) = DiffResults.gradient(obj.config.result)
# @inline gradient(obj::AbstractAutoDiffDifferentiable, i::Integer) = DiffResults.gradient(obj.config.result)[i]
@inline hessian(obj::TwiceDifferentiable) = DiffResults.hessian(obj.config.result)


@inline f(obj::AbstractAutoDiffDifferentiable, x) = obj.config.f(x)

# """
# For DynamicHMC support. Copies data.
# """
# @inline function (obj::OnceDifferentiable)(x)
#     fdf(obj, x)
#     DiffResults.ImmutableDiffResult(-obj.config.result.value, (-1 .* obj.config.result.grad,))
# end
# @inline function (obj::TwiceDifferentiable)(x)
#     fdf(obj, x)
#     DiffResults.ImmutableDiffResult(-obj.config.result.value, (-1 .* obj.config.result.grad,))
# end

@inline function df(obj::AbstractAutoDiffDifferentiable, x)
    ForwardDiff.gradient!(gradient(obj), obj.config.f, x, obj.config.gconfig, Val{false}())
end

@inline function fdf(obj::AbstractAutoDiffDifferentiable, x)
    # obj.config.result.grad = gradient(obj)
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    # DiffResults.value(obj.config.result)
    obj.config.result.value
end
@inline function scale_fdf(obj::AbstractAutoDiffDifferentiable, x, scale_target)
    # obj.config.result.grad = gradient(obj)
    scale = scale_gradient!(obj.config.result, obj.config.f, x, scale_target, obj.config.gconfig, Val{false}())
    # DiffResults.value(obj.config.result)
    obj.config.result.value, scale
end
@inline function scaled_fdf(obj::AbstractAutoDiffDifferentiable, x, scale)
    # obj.config.result.grad = gradient(obj)
    scaled_gradient!(obj.config.result, obj.config.f, x, scale, obj.config.gconfig, Val{false}())
    # DiffResults.value(obj.config.result)
    obj.config.result.value
end

# function Configuration(f::F, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)) where {F,V}
#     result = DiffResults.HessianResult(x)
#     jacobian_config = ForwardDiff.JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
#     gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
#     inner_result = DiffResults.DiffResult(zero(eltype(jacobian_config.duals[2])), jacobian_config.duals[2])
#     gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)
#     Configuration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
# end

@inline function (c::HessianConfiguration)(y, z)
    # c.inner_result.grad = y #Already true?
    # @show typeof(y), typeof(z)
    ForwardDiff.gradient!(c.inner_result, c.f, z, c.gradient_config, Val{false}())
    DiffResults.value!(c.result, ForwardDiff.value(DiffResults.value(c.inner_result)))
    # copyto!(y, DiffResults.gradient(c.inner_result))
    y
end
@inline function hessian!(c::HessianConfiguration, x::AbstractArray)
    ForwardDiff.jacobian!(DiffResults.hessian(c.result), c, DiffResults.gradient(c.result), x, c.jacobian_config, Val{false}())
    # DiffResults.hessian(c.result)
    DiffResults.hessian(c.result)
end

"""
Force (re-)evaluation of the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
@inline function value!!(obj::AbstractAutoDiffDifferentiable, x)
    # obj.f_calls .+= 1
    # copyto!(obj.x_f, x)
    DiffResults.value!(obj, f(obj, x) )
    value(obj)
end
# """
# Evaluates the objective value at `x`.
#
# Returns `f(x)`, but does *not* store the value in `obj.F`
# """
# @inline function value(obj::AbstractAutoDiffDifferentiable, x)
#     if x != obj.x_f
#         # obj.f_calls .+= 1
#         value!!(obj, x)
#     end
#     value(obj)
# end
"""
Evaluates the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
@inline function value!(obj::AbstractAutoDiffDifferentiable, x)
    # if x != obj.x_f
        value!!(obj, x)
    # end
    value(obj)
end

"""
Evaluates the gradient value at `x`

This does *not* update `obj.DF`.
"""
@inline function gradient(obj::AbstractAutoDiffDifferentiable, x)
    DF = gradient(obj)
    gradient!!(obj, x)
    DF
end
"""
Evaluates the gradient value at `x`.

Stores the value in `obj.DF`.
"""
@inline function gradient!(obj::AbstractAutoDiffDifferentiable, x)
    # if x != obj.x_df
        gradient!!(obj, x)
    # end
    gradient(obj)
end
"""
Force (re-)evaluation of the gradient value at `x`.

Stores the value in `obj.DF`.
"""
@inline function gradient!!(obj::AbstractAutoDiffDifferentiable, x)
    # obj.df_calls .+= 1
    copyto!(obj.x_df, x)
    df(obj, x)
end

@inline function value_gradient!(obj::AbstractAutoDiffDifferentiable, x)
    # if x != obj.x_f && x != obj.x_df
        value_gradient!!(obj, x)
    # elseif x != obj.x_f
    #     value!!(obj, x)
    # elseif x != obj.x_df
    #     gradient!!(obj, x)
    # end
    value(obj)
end
@inline function value_gradient!!(obj::AbstractAutoDiffDifferentiable, x)
    # obj.f_calls .+= 1
    # obj.df_calls .+= 1
    # copyto!(obj.x_f, x)
    # copyto!(obj.x_df, x)
    DiffResults.value!(obj, fdf(obj, x))
end

@inline function hessian!(obj::AbstractAutoDiffDifferentiable, x)
    # if x != obj.x_h
        hessian!!(obj, x)
    # end
    nothing
end
@inline function hessian!!(obj::AbstractAutoDiffDifferentiable, x)
    # obj.h_calls .+= 1
    # copyto!(obj.x_h, x)
    hessian!(obj.config, x)
end



# function ForwardDiff.vector_mode_dual_eval(f::F, x::MutableFixedSizeVector, cfg::Union{JacobianConfig,GradientConfig}) where F
#     xdual = cfg.duals
#     seed!(xdual, x, cfg.seeds)
#     return f(xdual)
# end

@inline function ForwardDiff.seed!(duals::AbstractMutableFixedSizeVector{P,ForwardDiff.Dual{T,V,N}}, x,
                                    seed::ForwardDiff.Partials{N,V}) where {T,V,N,P}
    @inbounds for i in 1:N
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seed)
    end
    return duals
end

@inline function ForwardDiff.seed!(duals::AbstractMutableFixedSizeVector{P,ForwardDiff.Dual{T,V,N}}, x,
                                    seeds::NTuple{N,ForwardDiff.Partials{N,V}}) where {T,V,N,P}
    @inbounds for i in 1:N
        duals[i] = ForwardDiff.Dual{T,V,N}(x[i], seeds[i])
    end
    return duals
end

@inline function ForwardDiff.seed!(duals::AbstractMutableFixedSizeVector{P,ForwardDiff.Dual{T,V,N}}, x, index,
                                    seed::ForwardDiff.Partials{N,V} = zero(ForwardDiff.Partials{N,V})) where {T,V,N,P}
    offset = index - 1
    @inbounds for i in 1:N
        j = i + offset
        duals[j] = ForwardDiff.Dual{T,V,N}(x[j], seed)
    end
    return duals
end

@inline function ForwardDiff.seed!(duals::AbstractMutableFixedSizeVector{P,ForwardDiff.Dual{T,V,N}}, x, index,
                                    seeds::NTuple{N,ForwardDiff.Partials{N,V}}, chunksize = N) where {T,V,N,P}
    offset = index - 1
    @inbounds for i in 1:chunksize
        j = i + offset
        duals[j] = ForwardDiff.Dual{T,V,N}(x[j], seeds[i])
    end
    return duals
end
function ForwardDiff.extract_jacobian!(::Type{T}, result::AbstractMutableFixedSizeMatrix{M,N}, ydual::AbstractArray, n) where {M,N,T}
    # out_reshaped = reshape(result, length(ydual), n)
    # @inbounds for col in 1:size(out_reshaped, 2), row in 1:size(out_reshaped, 1)
    #     out_reshaped[row, col] = partials(T, ydual[row], col)
    # end
    @inbounds for col in 1:N, row in 1:M
        # result[row, col] = ForwardDiff.partials(T, ydual[row], col)
        result[row, col] = ForwardDiff.partials(T, ydual[col], row)
    end
    return result
end
