
## Fix dep warning on BackTracking.
function (ls::LineSearches.BackTracking)(df::LineSearches.AbstractObjective, x::AbstractArray{T}, s::AbstractArray{T},
    α_0::Tα = real(T)(1), x_new::AbstractArray{T} = similar(x), ϕ_0 = nothing, dϕ_0 = nothing, alphamax = convert(real(T), Inf)) where {T, Tα}
    ϕ, dϕ = LineSearches.make_ϕ_dϕ(df, x_new, x, s)

    if ϕ_0 == nothing
    ϕ_0 = LineSearches.ϕ(Tα(0))
    end
    if dϕ_0 == nothing
    dϕ_0 = LineSearches.dϕ(Tα(0))
    end

    α_0 = min(α_0, min(alphamax, ls.maxstep / norm(s, Inf)))
    ls(ϕ, α_0, ϕ_0, dϕ_0)
end