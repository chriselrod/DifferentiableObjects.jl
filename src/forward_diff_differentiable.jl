
# Could these be Base.@pure ?
@generated Chunk(::Val{N}) where N = ForwardDiff.Chunk(N)
@generated ValP1(::Val{N}) where N = Val{N+1}()
@generated ValM1(::Val{N}) where N = Val{N-1}()


abstract type AutoDiffDifferentiable{P,T,F} <: DifferentiableObject{P} end
abstract type Configuration{P,V,F} end

struct GradientConfiguration{P,V,F,T,ND,DG} <: Configuration{P,V,F}
    f::F
    result::DiffResults.MutableDiffResult{1,V,Tuple{RecursiveVector{V,P}}}
    gconfig::ForwardDiff.GradientConfig{T,V,ND,DG}
end

struct HessianConfiguration{P,V,F,T,T2,ND,DJ,DG,DG2,P2} <: Configuration{P,V,F}
    f::F
    result::DiffResults.MutableDiffResult{2,V,Tuple{RecursiveVector{V,P},Symmetric{V,RecursiveMatrix{V,P,P,P2}}}}
    inner_result::DiffResults.MutableDiffResult{1,ForwardDiff.Dual{T,V,ND},Tuple{RecursiveVector{ForwardDiff.Dual{T,V,ND},P}}}
    jacobian_config::ForwardDiff.JacobianConfig{T,V,ND,DJ}
    gradient_config::ForwardDiff.GradientConfig{T,ForwardDiff.Dual{T,V,ND},ND,DG}
    gconfig::ForwardDiff.GradientConfig{T2,V,ND,DG2}
end
struct TwiceDifferentiable{P,T,F,C<:Configuration{P,T,F}} <: AutoDiffDifferentiable{P,T,F}
    x_f::RecursiveVector{T,P} # x used to evaluate f (stored in F)
    x_df::RecursiveVector{T,P} # x used to evaluate df (stored in DF)
    x_h::RecursiveVector{T,P} #??
    config::C
end
struct OnceDifferentiable{P,T,F,C<:GradientConfiguration{P,T,F}} <: AutoDiffDifferentiable{P,T,F}
    x_f::RecursiveVector{T,P} # x used to evaluate f (stored in F)
    x_df::RecursiveVector{T,P} # x used to evaluate df (stored in DF)
    config::C
end


# HessianConfiguration(f::F, x::RecursiveVector{T,P}) where {F,T,P} = Configuration(f, x) 
function GradientConfiguration(f::F, x::RecursiveVector{T,P}) where {F,T,P}
    result = DiffResults.GradientResult(x)
    chunk = Chunk(Val{P}())
    tag = ForwardDiff.Tag(f, T)
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)

    GradientConfiguration(f, result, gconfig)
end
function HessianConfiguration(f::F, x::RecursiveVector{T,P}) where {F,T,P}
    result = DiffResults.HessianResult(x)
    chunk = Chunk(Val{P}())
    tag = ForwardDiff.Tag(f, T)

    jacobian_config = ForwardDiff.JacobianConfig((f,ForwardDiff.gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    inner_result = DiffResults.GradientResult(jacobian_config.duals[2])
    gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)

    HessianConfiguration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
end
TwiceDifferentiable(f::F, ::Val{N}) where {F,N} = TwiceDifferentiable(f, RecursiveVector{Float64,N}())
function TwiceDifferentiable(f::F, x::RecursiveVector{T,N}) where {F,T,N}
    TwiceDifferentiable(x, similar(x), similar(x), HessianConfiguration(f, x))
end
function TwiceDifferentiable(x_f::RecursiveVector{T,P},x_df::RecursiveVector{T,P},
                            x_h::RecursiveVector{T,P},config::C) where {T,F,C<:Configuration{T,F},P}
    TwiceDifferentiable{P,T,F,C}(x_f, x_df, x_h, config)
end



function OnceDifferentiable(f, x_f::RecursiveVector{T,P}) where {T,P}
    OnceDifferentiable(x_f, similar(x_f), GradientConfiguration(f, x_f))
end
function OnceDifferentiable(x_f::RecursiveVector{T,P}, config::C) where {T,F,C<:GradientConfiguration{T,F},P}
    OnceDifferentiable{P,T,F,C}(x_f, similar(x_f), config)
end
# function OnceDifferentiable(x_f::RecursiveVector{T,P},x_df::RecursiveVector{T,P},
#                                 config::C) where {T,P,F,C<:GradientConfiguration{P,T,F}}
#     OnceDifferentiable{P,T,F,C}(x_f, x_df, config)
# end

# ProfileDifferentiable(f::F, ::Val{N}) where {F,N} = ProfileDifferentiable(f, Vector{Float64}(undef, N), Val{N}())
# function ProfileDifferentiable(f::F, x::AbstractArray{T}, ::Val{N}) where {F,T,N}
#     x_f = Vector{T}(undef, N-1)
#     ProfileDifferentiable(x_f, similar(x_f), similar(x_f), Configuration(f, x_f, ValM1(Val{N}())), x, Ref(N), Ref{T}(),::Val{N})
# end
# function ProfileDifferentiable(x_f::Vector{T},x_df::Vector{T},x_h::Vector{T}, config::C, x,::A i::RefValue{Int}, v::RefValue{T},::Val{N}) where {T,A<:AbstractArray{T},C,N}
#     ProfileDifferentiable{N,T,A,C}(x_f, x_df, x_h, config, x, i, v)
# end

function DiffResults.GradientResult(x::RecursiveVector{T,L}) where {T,L}
    DiffResults.MutableDiffResult(zero(T), (similar(x),))
end
function DiffResults.HessianResult(x::RecursiveVector{T,L}) where {T,L}
    DiffResults.MutableDiffResult(zero(T), (similar(x), Symmetric(RecursiveMatrix{T,L,L}())))
end

@inline DiffResults.value!(obj::AutoDiffDifferentiable, x::Real) = DiffResults.value!(obj.config.result, x)

@inline value(obj::AutoDiffDifferentiable) = DiffResults.value(obj.config.result)
@inline gradient(obj::AutoDiffDifferentiable) = DiffResults.gradient(obj.config.result)
@inline gradient(obj::AutoDiffDifferentiable, i::Integer) = DiffResults.gradient(obj.config.result)[i]
@inline hessian(obj::TwiceDifferentiable) = DiffResults.hessian(obj.config.result)


@inline f(obj::AutoDiffDifferentiable, x) = obj.config.f(x)

"""
For DynamicHMC support. Copies data.
"""
function (obj::OnceDifferentiable)(x)
    fdf(obj, x)
    DiffResults.ImmutableDiffResult(-obj.config.result.value, (-1 .* obj.config.result.derivs[1],))
end
function (obj::TwiceDifferentiable)(x)
    fdf(obj, x)
    DiffResults.ImmutableDiffResult(-obj.config.result.value, (-1 .* obj.config.result.derivs[1],))
end

@inline function df(obj::AutoDiffDifferentiable, x)
    ForwardDiff.gradient!(gradient(obj), obj.config.f, x, obj.config.gconfig, Val{false}())
end
@inline function fdf(obj::TwiceDifferentiable, x)
    obj.config.result.derivs = (gradient(obj), hessian(obj))
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    DiffResults.value(obj.config.result)
end
@inline function fdf(obj::OnceDifferentiable, x)
    obj.config.result.derivs = (gradient(obj),)
    ForwardDiff.gradient!(obj.config.result, obj.config.f, x, obj.config.gconfig, Val{false}())
    DiffResults.value(obj.config.result)
end

# function Configuration(f::F, x::AbstractArray{V}, chunk::Chunk = Chunk(x), tag = Tag(f, V)) where {F,V}
#     result = DiffResults.HessianResult(x)
#     jacobian_config = ForwardDiff.JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
#     gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals[2], chunk, tag)
#     inner_result = DiffResults.DiffResult(zero(eltype(jacobian_config.duals[2])), jacobian_config.duals[2])
#     gconfig = ForwardDiff.GradientConfig(f, x, chunk, tag)
#     Configuration(f, result, inner_result, jacobian_config, gradient_config, gconfig)
# end

function (c::HessianConfiguration)(y, z)
    c.inner_result.derivs = (y,) #Already true?
    ForwardDiff.gradient!(c.inner_result, c.f, z, c.gradient_config, Val{false}())
    DiffResults.value!(c.result, ForwardDiff.value(DiffResults.value(c.inner_result)))
    y
end
function hessian!(c::HessianConfiguration, x::AbstractArray)
    ForwardDiff.jacobian!(DiffResults.hessian(c.result), c, DiffResults.gradient(c.result), x, c.jacobian_config, Val{false}())
    DiffResults.hessian(c.result)
end

"""
Force (re-)evaluation of the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function value!!(obj::AutoDiffDifferentiable, x)
    # obj.f_calls .+= 1
    copyto!(obj.x_f, x)
    DiffResults.value!(obj, f(obj, x) )
end
"""
Evaluates the objective value at `x`.

Returns `f(x)`, but does *not* store the value in `obj.F`
"""
function value(obj::AutoDiffDifferentiable, x)
    if x != obj.x_f
        # obj.f_calls .+= 1
        value!!(obj, x)
    end
    value(obj)
end
"""
Evaluates the objective value at `x`.

Returns `f(x)` and stores the value in `obj.F`
"""
function value!(obj::AutoDiffDifferentiable, x)
    if x != obj.x_f
        value!!(obj, x)
    end
    value(obj)
end

"""
Evaluates the gradient value at `x`

This does *not* update `obj.DF`.
"""
function gradient(obj::AutoDiffDifferentiable, x)
    DF = gradient(obj)
    if x != obj.x_df
        tmp = copy(DF)
        gradient!!(obj, x)
        @inbounds for i ∈ eachindex(tmp)
            tmp[i], DF[i] = DF[i], tmp[i]
        end
        return tmp
    end
    DF
end
"""
Evaluates the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function gradient!(obj::AutoDiffDifferentiable, x)
    if x != obj.x_df
        gradient!!(obj, x)
    end
    gradient(obj)
end
"""
Force (re-)evaluation of the gradient value at `x`.

Stores the value in `obj.DF`.
"""
function gradient!!(obj::AutoDiffDifferentiable, x)
    # obj.df_calls .+= 1
    copyto!(obj.x_df, x)
    df(obj, x)
end

function value_gradient!(obj::AutoDiffDifferentiable, x)
    if x != obj.x_f && x != obj.x_df
        value_gradient!!(obj, x)
    elseif x != obj.x_f
        value!!(obj, x)
    elseif x != obj.x_df
        gradient!!(obj, x)
    end
    value(obj)
end
function value_gradient!!(obj::AutoDiffDifferentiable, x)
    # obj.f_calls .+= 1
    # obj.df_calls .+= 1
    copyto!(obj.x_f, x)
    copyto!(obj.x_df, x)
    DiffResults.value!(obj, fdf(obj, x))
end

function hessian!(obj::AutoDiffDifferentiable, x)
    if x != obj.x_h
        hessian!!(obj, x)
    end
    nothing
end
function hessian!!(obj::AutoDiffDifferentiable, x)
    # obj.h_calls .+= 1
    copyto!(obj.x_h, x)
    hessian!(obj.config, x)
end

