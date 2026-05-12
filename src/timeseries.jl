using LinearAlgebra

export add_noise, ordinal_entropy_complexity, plot_CH_plane, shade_admissible_noise!

# ------------------------------------------------------------
# Noise injection
# ------------------------------------------------------------

"""
    add_noise(t, x; σ, λ=0.0, ϵ_reg=1e-20) -> Vector{Float64}

Add spatially-correlated Gaussian noise to signal `x` sampled at times `t`.

- `σ`     : noise standard deviation
- `λ`     : correlation length for the exponential (OU) kernel; `0.0` = white noise
- `ϵ_reg` : diagonal regularisation for numerical stability (only used when `λ > 0`)
"""
function add_noise(t::AbstractVector, x::AbstractVector;
                   σ, λ::Float64=0.0, ϵ_reg::Float64=1e-20)
    n = length(t)
    if λ == 0.0
        noise = σ .* randn(n)
    else
        K = σ^2 .* [exp(-abs(t[i] - t[j]) / λ) for i in 1:n, j in 1:n]
        K += ϵ_reg * I
        noise = cholesky(Symmetric(K)).L * randn(n)
    end
    return x .+ noise
end

# ------------------------------------------------------------
# Ordinal entropy and complexity
# ------------------------------------------------------------

"""
    ordinal_entropy_complexity(x; d=5, τ=1) -> (H, C)

Compute the normalized permutation entropy `H` and statistical complexity `C`
of a time series `x` using ordinal patterns of dimension `d` and delay `τ`.
"""
function ordinal_entropy_complexity(x::AbstractVector; d::Int=5, τ::Int=1)
    est = OrdinalPatterns(; m=d, τ=τ)
    H   = information_normalized(Shannon(), est, x)
    C   = complexity(StatisticalComplexity(; o=est), x)
    return H, C
end

# ------------------------------------------------------------
# CH plane plotting
# ------------------------------------------------------------

"""
    plot_CH_plane(p; d=5, τ=1, noise_curve=false, kwargs...)

Draw the theoretical max- and min-complexity boundary curves on the CH plane.
Pass `noise_curve=true` to overlay a colored-noise (OU) reference curve.
"""
function plot_CH_plane(p; d::Int=5, τ::Int=1,
                       xlim=(0.0, 1.0), ylim=(0.0, 0.5),
                       num_max::Int=1, num_min::Int=1000,
                       curve_color=:black, curve_linewidth=1,
                       noise_curve::Bool=false, noise_kwargs...)
    o = OrdinalPatterns(; m=d, τ=τ)
    c = StatisticalComplexity(; o=o)
    min_raw, max_raw = entropy_complexity_curves(c; num_max=num_max, num_min=num_min)

    H_max = [Float64(pt[1]) for pt in max_raw]
    C_max = [Float64(pt[2]) for pt in max_raw]
    H_min = [Float64(pt[1]) for pt in min_raw]
    C_min = [Float64(pt[2]) for pt in min_raw]

    ord_max = sortperm(H_max)
    ord_min = sortperm(H_min)

    plot!(p, H_max[ord_max], C_max[ord_max];
          color=curve_color, linewidth=curve_linewidth,
          linestyle=:dash, label="", xlims=xlim, ylims=ylim,
          xlabel="Entropy", ylabel="Complexity")
    plot!(p, H_min[ord_min], C_min[ord_min];
          color=curve_color, linewidth=curve_linewidth,
          linestyle=:dash, label="")

    noise_curve && plot_noise_curve!(p; d=d, τ=τ, noise_kwargs...)

    i_peak = argmax(C_max[ord_max])
    annotate!(p, H_max[ord_max][i_peak], C_max[ord_max][i_peak],
              text("max complexity", curve_color, :center, :bottom, 7))
    i_mid = length(H_min[ord_min]) ÷ 2
    annotate!(p, H_min[ord_min][i_mid], C_min[ord_min][i_mid],
              text("min complexity", curve_color, :center, :bottom, 7))

    return p
end

function plot_noise_curve!(p;
    d               ::Int     = 5,
    τ               ::Int     = 1,
    n_realizations  ::Int     = 20,
    σ                         = 1.0,
    λ_min           ::Float64 = 0.001,
    λ_max           ::Float64 = 2.0,
    n_λ             ::Int     = 10,
    curve_color               = :darkorange,
    curve_linewidth           = 1.5,
    )
    t_ref    = collect(range(-2.0, 0.0; step=0.001))
    x_flat   = zeros(length(t_ref))
    λ_values = [0.0; exp.(range(log(λ_min), log(λ_max); length=n_λ))]
    H_curve  = Vector{Float64}(undef, length(λ_values))
    C_curve  = Vector{Float64}(undef, length(λ_values))

    for (i, λ) in enumerate(λ_values)
        Hs = Vector{Float64}(undef, n_realizations)
        Cs = Vector{Float64}(undef, n_realizations)
        for r in 1:n_realizations
            x_noisy    = add_noise(t_ref, x_flat; σ=σ, λ=λ)
            Hs[r], Cs[r] = ordinal_entropy_complexity(x_noisy; d=d, τ=τ)
        end
        H_curve[i] = mean(Hs)
        C_curve[i] = mean(Cs)
    end

    plot!(p, H_curve, C_curve;
          color=curve_color, linewidth=curve_linewidth, label="colored noise (OU)")
    return p
end

# ------------------------------------------------------------
# Admissible noise region shading
# ------------------------------------------------------------

function _interp_curve(H_knots, C_knots, H_query)
    out = similar(H_query)
    for (i, h) in enumerate(H_query)
        j = searchsortedfirst(H_knots, h)
        if j == 1
            out[i] = C_knots[1]
        elseif j > length(H_knots)
            out[i] = C_knots[end]
        else
            α      = (h - H_knots[j-1]) / (H_knots[j] - H_knots[j-1])
            out[i] = C_knots[j-1] + α * (C_knots[j] - C_knots[j-1])
        end
    end
    return out
end

"""
    shade_admissible_noise!(p, H₀, C₀,
                            H_max_curve, C_max_curve,
                            H_min_curve, C_min_curve;
                            color, alpha=0.25, label="")

Shade the region of CH space reachable by any amount of additive noise starting from a
noiseless signal at (H₀, C₀).  Two constraints from Schur-convexity:
  (1)  H  ≥  H₀              (entropy can only increase)
  (2)  C/H ≤ C₀/H₀           (Q_JS = C/H can only decrease)
intersected with the physically realizable region between the provided max/min curves.
"""
function shade_admissible_noise!(p, H₀, C₀,
                                  H_max_curve, C_max_curve,
                                  H_min_curve, C_min_curve;
                                  color, alpha=0.25, label="")
    slope   = C₀ / H₀
    H_grid  = collect(range(H₀, 1.0; length=500))
    C_ray   = slope .* H_grid
    C_upper = min.(C_ray, _interp_curve(H_max_curve, C_max_curve, H_grid))
    C_lower = max.(0.0,   _interp_curve(H_min_curve, C_min_curve, H_grid))
    valid   = C_upper .>= C_lower
    plot!(p, H_grid[valid], C_upper[valid];
          fillrange  = C_lower[valid],
          fillalpha  = alpha,
          fillcolor  = color,
          linewidth  = 0,
          color      = color,
          alpha      = 0,
          label      = label)
end
