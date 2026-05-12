ENV["GKSwstype"] = "100"

import Pkg
Pkg.activate(@__DIR__)

using Revise
using NoisyCE
using Plots
using LaTeXStrings
using ComplexityMeasures
using DifferentialEquations
using StochasticDiffEq
using Statistics
using Random

default(fontfamily = "Computer Modern")

# ------------------------------------------------------------
# Constants  (same as example_majorization.jl)
# ------------------------------------------------------------

const figure_folder = joinpath(@__DIR__, "figures")

const d = 5
const τ = 1
const N = 10000

const t_synth  = collect(range(-4.0, 0.0; length=N))
const Δt_synth = t_synth[2] - t_synth[1]

# σ sweep: log-spaced, same as majorization example
const σ_grad   = 10.0 .^ collect(-5.0:0.5:0.0)
const n_σ_grad = length(σ_grad)

# Shared plasma colormap — identical scale in every panel
const σ_colors = [cgrad(:plasma)[x] for x in range(0.05, 0.95; length=n_σ_grad)]

# ------------------------------------------------------------
# Signal generation
# ------------------------------------------------------------

# 1. Hénon map
x_henon = let a = 1.4, b = 0.3, n_burn = 10_000
    x, y = 0.1, 0.0
    for _ in 1:n_burn
        x, y = 1 - a*x^2 + y, b*x
    end
    xs = zeros(N); xs[1] = x; ys = y
    for i in 2:N
        xs[i], ys = 1 - a*xs[i-1]^2 + ys, b*xs[i-1]
    end
    (xs .- mean(xs)) ./ std(xs)
end

# 2. Lorenz X-component
x_lorenz_norm = let
    function lorenz!(du, u, p, _)
        σ_l, ρ, β = p
        du[1] =  σ_l * (u[2] - u[1])
        du[2] =  u[1] * (ρ - u[3]) - u[2]
        du[3] =  u[1] * u[2] - β * u[3]
    end
    sol = solve(ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 0.1*(N-1)), (10.0, 28.0, 8/3)),
                Tsit5(); reltol=1e-10, abstol=1e-12, saveat=0.1, maxiters=1_000*N)
    x_raw = Array(sol)[1, :]
    (x_raw .- mean(x_raw)) ./ std(x_raw)
end

# 3. Ornstein-Uhlenbeck process
x_ou = let Δt = Δt_synth
    sol = solve(SDEProblem((u,p,_) -> -p*u, (_,p,_) -> sqrt(2p), 0.0,
                           (t_synth[1], t_synth[end]), 400.0),
                EM(); dt=Δt, saveat=t_synth, seed=1234)
    x_raw = sol.u
    (x_raw .- mean(x_raw)) ./ std(x_raw)
end

# 4. Stochastic harmonic oscillator
x_sho = let Δt = Δt_synth
    ω, ζ, σ_s = 2π/0.1, 0.2, 500.0
    sol = solve(SDEProblem(
                    (du, u, p, _) -> (du[1] = u[2]; du[2] = -p[1]^2*u[1] - 2p[2]*p[1]*u[2]),
                    (du, _, p, _) -> (du[1] = 0.0;  du[2] = p[3]),
                    [0.0, 1.0], (t_synth[1], t_synth[end]), (ω, ζ, σ_s)),
                EM(); dt=Δt, saveat=t_synth, seed=5678)
    x_raw = [u[1] for u in sol.u]
    (x_raw .- mean(x_raw)) ./ std(x_raw)
end

# ------------------------------------------------------------
# CH coordinates (clean signals)
# ------------------------------------------------------------

H_henon,  C_henon  = ordinal_entropy_complexity(x_henon;       d=d, τ=τ)
H_lorenz, C_lorenz = ordinal_entropy_complexity(x_lorenz_norm; d=d, τ=τ)
H_ou,     C_ou     = ordinal_entropy_complexity(x_ou;          d=d, τ=τ)
H_sho,    C_sho    = ordinal_entropy_complexity(x_sho;         d=d, τ=τ)

println("Hénon  — H: $(round(H_henon,  digits=4))  C: $(round(C_henon,  digits=4))")
println("Lorenz — H: $(round(H_lorenz, digits=4))  C: $(round(C_lorenz, digits=4))")
println("OU     — H: $(round(H_ou,     digits=4))  C: $(round(C_ou,     digits=4))")
println("SHO    — H: $(round(H_sho,    digits=4))  C: $(round(C_sho,    digits=4))")

# ------------------------------------------------------------
# CH trajectories under white-noise sweep (fixed realization)
# ------------------------------------------------------------

rng_white = MersenneTwister(2025)
ε_henon  = randn(rng_white, N)
ε_lorenz = randn(rng_white, N)
ε_ou     = randn(rng_white, N)
ε_sho    = randn(rng_white, N)

HC_henon_σ  = [ordinal_entropy_complexity(x_henon        .+ σ .* ε_henon;  d=d, τ=τ) for σ in σ_grad]
HC_lorenz_σ = [ordinal_entropy_complexity(x_lorenz_norm  .+ σ .* ε_lorenz; d=d, τ=τ) for σ in σ_grad]
HC_ou_σ     = [ordinal_entropy_complexity(x_ou           .+ σ .* ε_ou;     d=d, τ=τ) for σ in σ_grad]
HC_sho_σ    = [ordinal_entropy_complexity(x_sho          .+ σ .* ε_sho;    d=d, τ=τ) for σ in σ_grad]

# ------------------------------------------------------------
# Theoretical CH boundary curves (computed once, shared across panels)
# ------------------------------------------------------------

let
    global H_max_curve, C_max_curve, H_min_curve, C_min_curve
    o_ec = OrdinalPatterns(; m=d, τ=τ)
    c_ec = StatisticalComplexity(; o=o_ec)
    min_raw, max_raw = entropy_complexity_curves(c_ec; num_max=1, num_min=1000)
    ord_max = sortperm([Float64(pt[1]) for pt in max_raw])
    ord_min = sortperm([Float64(pt[1]) for pt in min_raw])
    H_max_curve = [Float64(pt[1]) for pt in max_raw][ord_max]
    C_max_curve = [Float64(pt[2]) for pt in max_raw][ord_max]
    H_min_curve = [Float64(pt[1]) for pt in min_raw][ord_min]
    C_min_curve = [Float64(pt[2]) for pt in min_raw][ord_min]
end

# ------------------------------------------------------------
# Helper: draw one CH panel
#   show_ylabel / show_xlabel control axis labels and margins
#   so that inner edges between panels have no wasted whitespace.
# ------------------------------------------------------------

function draw_CH_panel(H0, C0, hc_σ, marker, title_str;
                       show_ylabel=true, show_xlabel=true)
    p = plot(dpi=300, legend=false,
             guidefontsize=10, tickfontsize=9,
             framestyle=:box, grid=:none,
             xlims=(0.0, 1.0), ylims=(0.0, 0.5),
             xlabel = show_xlabel ? L"H" : "",
             ylabel = show_ylabel ? L"C" : "",
             title  = title_str,
             left_margin   = show_ylabel  ? 10Plots.mm : 1Plots.mm,
             right_margin  = 1Plots.mm,
             top_margin    = 4Plots.mm,
             bottom_margin = show_xlabel  ? 8Plots.mm  : 1Plots.mm)

    # CH boundary curves
    plot!(p, H_max_curve, C_max_curve; color=:black, linewidth=0.9, linestyle=:dash, label="")
    plot!(p, H_min_curve, C_min_curve; color=:black, linewidth=0.9, linestyle=:dash, label="")

    # Admissible noise region (same steelblue shade in every panel)
    shade_admissible_noise!(p, H0, C0,
                             H_max_curve, C_max_curve,
                             H_min_curve, C_min_curve;
                             color=:steelblue, alpha=0.20)

    # Trajectory line (light gray)
    Hs = [first(hc) for hc in hc_σ]
    Cs = [last(hc)  for hc in hc_σ]
    plot!(p, [H0; Hs], [C0; Cs]; color=:gray, linewidth=0.8, alpha=0.5, label="")

    # Noisy points: plasma color = σ level (shared scale)
    for i in eachindex(σ_grad)
        scatter!(p, [Hs[i]], [Cs[i]];
                 markercolor=σ_colors[i], markerstrokecolor=σ_colors[i],
                 markersize=5, markershape=marker, label="")
    end

    # Clean anchor (large black marker)
    scatter!(p, [H0], [C0];
             markersize=10, markershape=marker,
             markercolor=:black, markerstrokecolor=:white, markerstrokewidth=1.5,
             label="")

    return p
end

# top-left, top-right, bottom-left, bottom-right
p1 = draw_CH_panel(H_henon,  C_henon,  HC_henon_σ,  :diamond,   "H\u00e9non";            show_ylabel=true,  show_xlabel=false)
p2 = draw_CH_panel(H_lorenz, C_lorenz, HC_lorenz_σ, :pentagon,  "Lorenz";                show_ylabel=false, show_xlabel=false)
p3 = draw_CH_panel(H_ou,     C_ou,     HC_ou_σ,     :circle,    "Ornstein\u2013Uhlenbeck"; show_ylabel=true,  show_xlabel=true)
p4 = draw_CH_panel(H_sho,    C_sho,    HC_sho_σ,    :utriangle, "Stoch. harm. osc.";     show_ylabel=false, show_xlabel=true)

# ------------------------------------------------------------
# Shared legend panel
# ------------------------------------------------------------

p_leg = plot(; framestyle=:none, xticks=nothing, yticks=nothing,
               xlims=(0,1), ylims=(0,1), legend=:inside, legendfontsize=8,
               background_color_inside=:transparent, dpi=300)

scatter!(p_leg, [NaN], [NaN]; label="noiseless",
         markersize=9, markershape=:circle,
         markercolor=:black, markerstrokecolor=:white, markerstrokewidth=1.5)

plot!(p_leg, [NaN], [NaN]; label="admissible region",
      color=:steelblue, linewidth=8, alpha=0.35)

plot!(p_leg, [NaN], [NaN]; label="", color=:white, linewidth=0)
plot!(p_leg, [NaN], [NaN]; label="noise level (\u03c3)", color=:black, linewidth=0)

for i in eachindex(σ_grad)
    # show only exact powers of 10 (odd indices in the half-decade sweep)
    i % 2 == 1 || continue
    exp_int = round(Int, log10(σ_grad[i]))
    lbl = exp_int == 0 ? latexstring("\\sigma = 1") :
                         latexstring("\\sigma = 10^{$(exp_int)}")
    scatter!(p_leg, [NaN], [NaN]; label=lbl,
             markercolor=σ_colors[i], markerstrokecolor=σ_colors[i],
             markersize=5, markershape=:circle)
end

plot!(p_leg, [NaN], [NaN]; label="", color=:white, linewidth=0)
plot!(p_leg, [NaN], [NaN]; label="signal", color=:black, linewidth=0)

for (marker, lbl) in [(:diamond,   "H\u00e9non"),
                       (:pentagon,  "Lorenz"),
                       (:circle,    "Ornstein\u2013Uhlenbeck"),
                       (:utriangle, "Stoch. harm. osc.")]
    scatter!(p_leg, [NaN], [NaN]; label=lbl,
             markersize=6, markershape=marker,
             markercolor=:gray60, markerstrokecolor=:gray60)
end

# ------------------------------------------------------------
# Assemble and save
# ------------------------------------------------------------

fig = plot(p1, p2, p3, p4, p_leg;
           layout = @layout([grid(2, 2) c{0.15w}]),
           size   = (1050, 750), dpi=300)

savefig(fig, joinpath(figure_folder, "CE_plane_fourpanel.png"))
println("Saved to $(figure_folder)/CE_plane_fourpanel.png")
