#### Taken from LineSearches.jl. See backtracking for information.


"""
Provide static initial step length.

Keyword `alpha` corresponds to static step length, default is 1.0.
If keyword `scaled = true`, then the initial step length
is scaled with the `l_2` norm of the step direction.
"""
struct InitialStatic{T,S}
    alpha::T
    # scaled::Bool = false # Scales step. alpha ← min(alpha,||s||_2) / ||s||_2
end
InitialStatic() = InitialStatic{Float64,false}(1.0)
InitialStatic(α::T) where {T} = InitialStatic{T,false}(α)
InitialStatic(α::T,::Val{S}) where {T,S} = InitialStatic{T,S}(α)

function (is::InitialStatic{T,true})(ls, state, phi_0, dphi_0, df) where T
    PT = promote_type(T, real(eltype(state.s)))
    state.alpha = convert(PT, min(is.alpha, ns)) / ns
end

function (is::InitialStatic{T,false})(ls, state, phi_0, dphi_0, df) where T
    PT = promote_type(T, real(eltype(state.s)))
    state.alpha = convert(PT, is.alpha)
end
