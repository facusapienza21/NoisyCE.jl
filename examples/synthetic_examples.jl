# ============================================================
# Generates:
#   - Figure 3 (synthetic_majorization_combined),
#   - Figure 4 (CE_plane_fivepanel),
# Author : Facundo Sapienza
# Date   : 2026-07-20
# ============================================================

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
using Printf
using Random

# Packege not available to install
using Brownian

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

const figure_folder = joinpath(@__DIR__, "figures")

# const N = 10000
const N = 1000000

# Per-dataset embedding dimension d, lag τ, and saveat δt.
# Each continuous simulation runs from t=0 to N*δt (time-independent models).
# Hénon is discrete so δt = 1 iteration (dimensionless).
d_henon  = 5;   τ_henon  = 1;   δt_henon  = 1.0
d_logis  = 6;   τ_logis  = 1;   δt_logis  = 1.0
d_lorenz = 5;   τ_lorenz = 1;   δt_lorenz = 0.1
d_ou     = 5;   τ_ou     = 1;   δt_ou     = 0.1
d_fbm    = 4;   τ_fbm    = 1;   δt_fbm    = 1.0 / N

# White noise amplitude sweep (log-spaced)
# σ_grad   = 10.0 .^ collect(-5.0:0.5:0.0)
SNR_grad = 10.0 .^ collect(-3.0:0.2:0.0)
n_SNR_grad = length(SNR_grad)
SNR_colors = [cgrad(:plasma)[x] for x in range(0.95, 0.05; length=n_SNR_grad)]

# Correlated noise: fix σ, sweep lag-1 autocorrelation ρ ∈ [0, 1]
SNR_corr_fix = 0.2
n_ρ_grad = 11
ρ_grad     = collect(range(0.0, 1.0; length=n_ρ_grad))
ρ_colors   = [cgrad(:viridis)[x] for x in range(0.95, 0.05; length=n_ρ_grad)]

const λ_fix = 0.0

# ------------------------------------------------------------
# Signal generation
# ------------------------------------------------------------

# Hénon map
x_henon, y_henon, σ_henon = let a = 1.4, b = 0.3, n_burn = 10_000
    x, y = 0.0, 0.0
    for _ in 1:n_burn
        x, y = 1 - a*x^2 + y, b*x
    end
    xs = zeros(N); ys = zeros(N)
    xs[1] = x; ys[1] = y
    for i in 2:N
        xs[i] = 1 - a*xs[i-1]^2 + ys[i-1]
        ys[i] = b*xs[i-1]
    end
    # (xs .- mean(xs)) ./ std(xs), ys
    xs, ys, std(xs)
end

# Logistic map
x_logis, σ_logis = let r = 4.0, n_burn = 10_000
    x =  0.1
    for _ in 1:n_burn
        x = r * x * (1.0 - x)
    end
    xs = zeros(N)
    xs[1] = x
    for i in 2:N
        xs[i] = r * xs[i-1] * (1 - xs[i-1])
    end
    xs, std(xs)
end

# Lorenz system
x_lorenz_norm, σ_lorenz = let t_burn = 100
    function lorenz!(du, u, p, _)
        σ_l, ρ, β = p
        du[1] =  σ_l * (u[2] - u[1])
        du[2] =  u[1] * (ρ - u[3]) - u[2]
        du[3] =  u[1] * u[2] - β * u[3]
    end
    u0 = [1.0, 1.0, 1.0]
    K_burn = trunc(Int, t_burn / δt_lorenz)
    sol = solve(ODEProblem(lorenz!, u0, (0.0, (K_burn + N -1) * δt_lorenz), (10.0, 28.0, 8/3)),
                Tsit5(); reltol=1e-12, abstol=1e-12, saveat=δt_lorenz, maxiters=1_000*N)
    x_raw = Array(sol)[1, (K_burn+1):end]
    @assert length(x_raw) == N
    # (x_raw .- mean(x_raw)) ./ std(x_raw)
    x_raw, std(x_raw)
end

# Ornstein–Uhlenbeck Process
x_ou, σ_ou = let
    sol = solve(SDEProblem(
        (u,p,_) -> -p*u,
        (_,p,_) -> 1.0,
        0.0,
        (0.0, (N-1) * δt_ou),
        1.0),
        EM();
        dt = δt_ou,
        saveat=δt_ou,
        seed=1234)
    x_raw = sol.u
    # (x_raw .- mean(x_raw)) ./ std(x_raw)
    x_raw, std(x_raw)
end

# Fractional Brownian Motion
H_fbm = 0.91   # Hurst exponent: H < 0.5 antipersistent, H = 0.5 Brownian, H > 0.5 persistent
x_fbm, σ_fbm = let
    # p = FBM(0.0:(1/N):1.0, H_fbm)
    p = FBM(0.0:1.0:(N), H_fbm)
    # x_raw = rand(p, fbm=false)
    x_raw = rand(p)[begin:end - 1]
    x_raw, 0.01 * std(x_raw)
end

let
    t_full = 0.0:δt_fbm:(N - 1) * δt_fbm
    p_fbm  = plot(t_full, x_fbm;
                  linewidth=0.6, color=:black, label="",
                  xlabel="Step", ylabel="x",
                  title="fBm (H = $H_fbm), N = $N",
                  framestyle=:box, grid=:none,
                  size=(800, 300), dpi=300, margin=5Plots.mm)
    savefig(p_fbm, joinpath(figure_folder, "fbm_full.png"))
    println("Saved fbm_full.png")
end

# ------------------------------------------------------------
# Ordinal pattern probabilities (per-dataset τ)
# ------------------------------------------------------------

_ord_probs = (x, d, τ) -> vec(first(allprobabilities_and_outcomes(OrdinalPatterns(; m=d, τ=τ), x)))
_n_pat     = (d, τ)   -> N - (d - 1) * τ   # effective ordinal window count

prob_henon_clean  = _ord_probs(x_henon,        d_henon,  τ_henon)
prob_logis_clean  = _ord_probs(x_logis,        d_logis,  τ_logis)
prob_lorenz_clean = _ord_probs(x_lorenz_norm,  d_lorenz, τ_lorenz)
prob_ou_clean     = _ord_probs(x_ou,           d_ou,     τ_ou)
prob_fbm_clean    = _ord_probs(x_fbm,          d_fbm,    τ_fbm)

N_pat_henon  = _n_pat(d_henon,  τ_henon)
N_pat_logis  = _n_pat(d_logis,  τ_logis)
N_pat_lorenz = _n_pat(d_lorenz, τ_lorenz)
N_pat_ou     = _n_pat(d_ou,     τ_ou)
N_pat_fbm    = _n_pat(d_fbm,    τ_fbm)

# ------------------------------------------------------------
# White noise sweep: fixed noise realization, varying σ
# ------------------------------------------------------------

rng_white = MersenneTwister(2025)
ε_henon  = randn(rng_white, N)
ε_logis  = randn(rng_white, N)
ε_lorenz = randn(rng_white, N)
ε_ou     = randn(rng_white, N)
ε_fbm    = randn(rng_white, N)

probs_henon_σgrad  = [_ord_probs(x_henon        .+ snr .* σ_henon .* ε_henon,  d_henon,  τ_henon)  for snr in SNR_grad]
probs_logis_σgrad  = [_ord_probs(x_logis        .+ snr .* σ_logis .* ε_logis,  d_logis,  τ_logis)  for snr in SNR_grad]
probs_lorenz_σgrad = [_ord_probs(x_lorenz_norm  .+ snr .* σ_lorenz .* ε_lorenz, d_lorenz, τ_lorenz) for snr in SNR_grad]
probs_ou_σgrad     = [_ord_probs(x_ou           .+ snr .* σ_ou  .* ε_ou,  d_ou,  τ_ou)  for snr in SNR_grad]
probs_fbm_σgrad    = [_ord_probs(x_fbm          .+ snr .* σ_fbm .* ε_fbm, d_fbm, τ_fbm) for snr in SNR_grad]

# ------------------------------------------------------------
# Correlated noise sweep: AR(1) with fixed σ, varying ρ
# This generates a signal such that 
# E[ϵ_i] = 0
# V[ϵ_i] = σ^2
# E[ϵ_(i+k) ϵ_i] = ρ^k σ^2
# ------------------------------------------------------------

function ar1_noise(n, σ, ρ; seed)
    rng     = MersenneTwister(seed)
    σ_innov = σ * sqrt(max(1 - ρ^2, eps()))
    ε       = zeros(n)
    ε[1]    = σ * randn(rng)
    for i in 2:n
        ε[i] = ρ * ε[i-1] + σ_innov * randn(rng)
    end
    return ε
end

probs_henon_ρgrad  = [_ord_probs(x_henon        .+ ar1_noise(N, σ_henon  * SNR_corr_fix, ρ_grad[i]; seed=2000+i), d_henon,  τ_henon)  for i in eachindex(ρ_grad)]
probs_logis_ρgrad  = [_ord_probs(x_logis        .+ ar1_noise(N, σ_logis  * SNR_corr_fix, ρ_grad[i]; seed=2000+i), d_logis,  τ_logis)  for i in eachindex(ρ_grad)]
probs_lorenz_ρgrad = [_ord_probs(x_lorenz_norm  .+ ar1_noise(N, σ_lorenz * SNR_corr_fix, ρ_grad[i]; seed=3000+i), d_lorenz, τ_lorenz) for i in eachindex(ρ_grad)]
probs_ou_ρgrad     = [_ord_probs(x_ou  .+ ar1_noise(N, σ_ou  * SNR_corr_fix, ρ_grad[i]; seed=4000+i), d_ou,  τ_ou)  for i in eachindex(ρ_grad)]
probs_fbm_ρgrad    = [_ord_probs(x_fbm .+ ar1_noise(N, σ_fbm * SNR_corr_fix, ρ_grad[i]; seed=5000+i), d_fbm, τ_fbm) for i in eachindex(ρ_grad)]

# ------------------------------------------------------------
# Majorization test p-values
# H₀: p_clean ≻ p_noisy
# ------------------------------------------------------------

println("Computing majorization test p-values (σ sweep)...")
pvals_henon  = [majorization_test(prob_henon_clean,  probs_henon_σgrad[i],  N_pat_henon,  N_pat_henon;  B=1000).p_value for i in eachindex(SNR_grad)]
pvals_logis  = [majorization_test(prob_logis_clean,  probs_logis_σgrad[i],  N_pat_logis,  N_pat_logis;  B=1000).p_value for i in eachindex(SNR_grad)]
pvals_lorenz = [majorization_test(prob_lorenz_clean, probs_lorenz_σgrad[i], N_pat_lorenz, N_pat_lorenz; B=1000).p_value for i in eachindex(SNR_grad)]
pvals_ou     = [majorization_test(prob_ou_clean,  probs_ou_σgrad[i],  N_pat_ou,  N_pat_ou;  B=1000).p_value for i in eachindex(SNR_grad)]
pvals_fbm    = [majorization_test(prob_fbm_clean, probs_fbm_σgrad[i], N_pat_fbm, N_pat_fbm; B=1000).p_value for i in eachindex(SNR_grad)]

println("Computing majorization test p-values (ρ sweep)...")
pvals_henon_ρ  = [majorization_test(prob_henon_clean,  probs_henon_ρgrad[i],  N_pat_henon,  N_pat_henon;  B=1000).p_value for i in eachindex(ρ_grad)]
pvals_logis_ρ  = [majorization_test(prob_logis_clean,  probs_logis_ρgrad[i],  N_pat_logis,  N_pat_logis;  B=1000).p_value for i in eachindex(ρ_grad)]
pvals_lorenz_ρ = [majorization_test(prob_lorenz_clean, probs_lorenz_ρgrad[i], N_pat_lorenz, N_pat_lorenz; B=1000).p_value for i in eachindex(ρ_grad)]
pvals_ou_ρ     = [majorization_test(prob_ou_clean,  probs_ou_ρgrad[i],  N_pat_ou,  N_pat_ou;  B=1000).p_value for i in eachindex(ρ_grad)]
pvals_fbm_ρ    = [majorization_test(prob_fbm_clean, probs_fbm_ρgrad[i], N_pat_fbm, N_pat_fbm; B=1000).p_value for i in eachindex(ρ_grad)]
println("Done.")

# ------------------------------------------------------------
# Figure: 4 rows × 4 columns + legend column
#   col 1: time series
#   col 2: majorization curves, white noise (σ sweep)
#   col 3: majorization curves, correlated noise (ρ sweep)
#   col 4: p-value panel (dual x-axis: σ bottom, ρ top)
#   col 5: legend
# ------------------------------------------------------------

maj_ext(probs) = [0.0; majorization_curve(probs)]

N_plot       = 200
plot_idx_disc     = 9800:9900
plot_idx_logis    = 9800:9859
plot_idx_cont     = (N - N_plot + 1):N |> collect
# Inset: consecutive original samples so individual data points are visible.
n_inset      = 20           # number of consecutive original samples to show
inset_start  = N - N_plot + 10           # starting original index (late in signal)
inset_orig   = inset_start:(inset_start + n_inset - 1)
α_level     = 0.05

x_norm   = collect(range(0.0, 1.0; length=n_SNR_grad))
x_norm_ρ = collect(range(0.0, 1.0; length=n_ρ_grad))

tick_idx    = 1:2:n_SNR_grad
tick_idx_ρ  = 1:2:n_ρ_grad
SNR_tick_vals = x_norm[tick_idx]
SNR_tick_lbls = [L"10^{%$(round(Int, log10(SNR_grad[i])))}" for i in tick_idx]
ρ_tick_vals   = x_norm_ρ[tick_idx_ρ]
ρ_tick_lbls   = ["$(round(ρ_grad[i], digits=2))" for i in tick_idx_ρ]

lay = @layout [grid(5, 4); b{0.07h}]
fig = plot(layout=lay, size=(1150, 1350), dpi=600,
           guidefontsize=8, tickfontsize=7, legendfontsize=7,
           framestyle=:axes, grid=:none,
           left_margin=0Plots.mm, right_margin=0Plots.mm,
           top_margin=2Plots.mm, bottom_margin=2Plots.mm,
           legend=false)

function draw_row!(row, x_clean, ε_white, d_ds, τ_ds, Δt_ds, t_axis, xlabel_ts,
                   probs_clean,
                   probs_σnoisy, pvals_σ,
                   probs_ρnoisy, pvals_ρ,
                   plot_idx,
                   row_label; first_row=false, ylims_ts=nothing, σ=nothing, discrete=false,
                   snr_xlabel="Noise level σ₀", snr_ticks=(SNR_tick_vals, SNR_tick_lbls))
    n_pat_ds   = factorial(d_ds)
    ranks_norm = collect(0:n_pat_ds) ./ n_pat_ds
    uniform_norm = ranks_norm

    sp_ts    = (row - 1) * 4 + 1
    sp_maj_σ = (row - 1) * 4 + 2
    sp_maj_ρ = (row - 1) * 4 + 3
    sp_pval  = (row - 1) * 4 + 4
    sp_inset = 21 + row

    ts_title    = first_row ? "Time series"        : ""
    maj_σ_title = first_row ? "White noise (σ)"   : ""
    maj_ρ_title = first_row ? L"Colored noise $(\rho)$" : ""
    pval_title  = first_row ? "Majorization test"  : ""

    # --- Time series ---
    noisy_ts = [x_clean .+ snr .* σ .* ε_white for snr in SNR_grad]
    for (i, ts) in enumerate(noisy_ts)
        plot!(fig, t_axis, ts[plot_idx]; subplot=sp_ts, color=SNR_colors[i], linewidth=0.7, alpha=0.3, label="")
        discrete && scatter!(fig, t_axis, ts[plot_idx]; subplot=sp_ts, markercolor=SNR_colors[i],
                             markerstrokecolor=SNR_colors[i], markersize=2, alpha=0.3, label="")
    end
    ts_ylims = isnothing(ylims_ts) ? () : (ylims=ylims_ts,)
    plot!(fig, t_axis, x_clean[plot_idx]; subplot=sp_ts, color=:black, linewidth=1.0, label="",
          xlabel=xlabel_ts, ylabel=row_label, title=ts_title, ts_ylims...)
    discrete && scatter!(fig, t_axis, x_clean[plot_idx]; subplot=sp_ts, markercolor=:black,
                         markerstrokecolor=:black, markersize=3, label="")

    # Zoom box on main panel: inset coords relative to plot_idx[1] to match 0-based t_axis
    t_inset_orig = (inset_orig .- plot_idx[1]) .* Δt_ds
    x_lo = t_inset_orig[1]
    x_hi = t_inset_orig[end]
    all_inset_y = vcat([ts[inset_orig] for ts in noisy_ts]..., x_clean[inset_orig])
    y_lo, y_hi = extrema(all_inset_y)
    buf = 0.08 * (y_hi - y_lo)
    plot!(fig, Shape([x_lo, x_hi, x_hi, x_lo], [y_lo-buf, y_lo-buf, y_hi+buf, y_hi+buf]);
          subplot=sp_ts, linecolor=:black, linewidth=1.2, fillalpha=0.0, label="")

    # Inset panel: consecutive original samples with lines + scatter
    plot!(fig; inset=(sp_ts, bbox(0.54, 0.50, 0.43, 0.45)), subplot=sp_inset,
          framestyle=:box, grid=false, xticks=nothing, yticks=nothing,
          background_color_inside=RGBA(1,1,1,0.85))
    for (i, ts) in enumerate(noisy_ts)
        plot!(fig, t_inset_orig, ts[inset_orig];
              subplot=sp_inset, color=SNR_colors[i], linewidth=1.0, alpha=0.3, label="")
        scatter!(fig, t_inset_orig, ts[inset_orig];
                 subplot=sp_inset, markercolor=SNR_colors[i], markerstrokecolor=SNR_colors[i],
                 markersize=2, alpha=0.3, label="")
    end
    plot!(fig, t_inset_orig, x_clean[inset_orig];
          subplot=sp_inset, color=:black, linewidth=2.0, label="")
    scatter!(fig, t_inset_orig, x_clean[inset_orig];
             subplot=sp_inset, markercolor=:black, markerstrokecolor=:black,
             markersize=3, label="")


    # --- Majorization curves: white noise σ sweep ---
    for i in eachindex(SNR_grad)
        plot!(fig, ranks_norm, maj_ext(probs_σnoisy[i]);
              subplot=sp_maj_σ, color=SNR_colors[i], linewidth=1.2, label="")
    end
    plot!(fig, ranks_norm, maj_ext(probs_clean);
          subplot=sp_maj_σ, color=:black, linewidth=2.2, label="",
          xlabel="rank", ylabel="Accumulated prob.", title=maj_σ_title,
          xlims=(0.0, 1.0), ylims=(0.0, 1.0), aspect_ratio=1)
    plot!(fig, ranks_norm, uniform_norm;
          subplot=sp_maj_σ, color=:gray, linewidth=0.8, linestyle=:dash, label="")
    annotate!(fig, 0.97, 0.05, text("d=$(d_ds), τ=$(τ_ds)", 6, :black, :right); subplot=sp_maj_σ)

    # --- Majorization curves: correlated noise ρ sweep ---
    for i in eachindex(ρ_grad)
        plot!(fig, ranks_norm, maj_ext(probs_ρnoisy[i]);
              subplot=sp_maj_ρ, color=ρ_colors[i], linewidth=1.2, label="")
    end
    plot!(fig, ranks_norm, maj_ext(probs_clean);
          subplot=sp_maj_ρ, color=:black, linewidth=2.2, label="",
          xlabel="rank", ylabel="Accumulated prob.", title=maj_ρ_title,
          xlims=(0.0, 1.0), ylims=(0.0, 1.0), aspect_ratio=1)
    plot!(fig, ranks_norm, uniform_norm;
          subplot=sp_maj_ρ, color=:gray, linewidth=0.8, linestyle=:dash, label="")

    # --- p-value panel with dual x-axis ---
    plot!(fig, x_norm, pvals_σ;
          subplot=sp_pval, color=:black, linewidth=1.8, label="",
          title=pval_title, ylabel="p-value",
          xlims=(-0.05, 1.05), ylims=(0.0, 1.25),
          yticks=0.0:0.25:1.0,
          xticks=snr_ticks,
          xlabel=snr_xlabel)
    scatter!(fig, x_norm, pvals_σ;
             subplot=sp_pval, markercolor=SNR_colors, markerstrokecolor=SNR_colors,
             markersize=4, label="")

    plot!(fig, x_norm_ρ, pvals_ρ;
          subplot=sp_pval, color=:steelblue, linewidth=1.8, label="")
    scatter!(fig, x_norm_ρ, pvals_ρ;
             subplot=sp_pval, markercolor=ρ_colors, markerstrokecolor=ρ_colors,
             markersize=4, label="")

    plot!(fig, [-0.05, 1.05], [α_level, α_level];
          subplot=sp_pval, color=:gray, linewidth=1.0, label="")
    plot!(fig, Shape([-0.05, 1.05, 1.05, -0.05], [0.0, 0.0, α_level, α_level]);
          subplot=sp_pval, fillcolor=:gray, fillalpha=0.2, linewidth=0, label="")
    annotate!(fig, 1.03, α_level + 0.03,
              text(L"\alpha=0.05", 6, :gray, :right); subplot=sp_pval)

    for j in eachindex(tick_idx_ρ)
        annotate!(fig, ρ_tick_vals[j], 1.06,
                  text(ρ_tick_lbls[j], 6, :steelblue, :center); subplot=sp_pval)
    end
    annotate!(fig, 0.5, 1.18, text("ρ", 7, :steelblue, :center); subplot=sp_pval)

end

# Per-model time axes — all start from 0 for display
t_axis_henon  = float.(0:length(plot_idx_disc)-1)
t_axis_logis  = float.(0:length(plot_idx_logis)-1)
t_axis_lorenz = (0:length(plot_idx_cont)-1) .* δt_lorenz
t_axis_ou     = (0:length(plot_idx_cont)-1) .* δt_ou
t_axis_fbm    = (0:length(plot_idx_cont)-1) .* δt_fbm

draw_row!(1, x_henon,       ε_henon,  d_henon,  τ_henon,  δt_henon,  t_axis_henon,  "Iteration",
          prob_henon_clean,  probs_henon_σgrad,  pvals_henon,  probs_henon_ρgrad,  pvals_henon_ρ, plot_idx_disc, "Hénon";       first_row=true, ylims_ts=(-2.0, 2.0), σ=σ_henon, discrete=true)
draw_row!(2, x_logis,       ε_logis,  d_logis,  τ_logis,  δt_logis,  t_axis_logis,  "Iteration",
          prob_logis_clean,  probs_logis_σgrad,  pvals_logis,  probs_logis_ρgrad,  pvals_logis_ρ, plot_idx_logis, "Logistic Map"; first_row=false, ylims_ts=(0.0, 1.0), σ=σ_logis, discrete=true)
draw_row!(3, x_lorenz_norm, ε_lorenz, d_lorenz, τ_lorenz, δt_lorenz, t_axis_lorenz, "Time",
          prob_lorenz_clean, probs_lorenz_σgrad, pvals_lorenz, probs_lorenz_ρgrad, pvals_lorenz_ρ, plot_idx_cont, "Lorenz"; σ=σ_lorenz)
draw_row!(4, x_ou,          ε_ou,     d_ou,     τ_ou,     δt_ou,     t_axis_ou,     "Time",
          prob_ou_clean,     probs_ou_σgrad,     pvals_ou,     probs_ou_ρgrad,     pvals_ou_ρ,     plot_idx_cont, "Ornstein-Uhlenbeck"; σ=σ_ou)
draw_row!(5, x_fbm,         ε_fbm,    d_fbm,    τ_fbm,    δt_fbm,    t_axis_fbm,    "Time",
          prob_fbm_clean,    probs_fbm_σgrad,    pvals_fbm,    probs_fbm_ρgrad,    pvals_fbm_ρ,    plot_idx_cont, "fBm (H=$H_fbm)"; σ=σ_fbm)

# ------------------------------------------------------------
# Legend panel at the bottom (subplot 21)
# ------------------------------------------------------------

legend_idx = 21

plot!(fig; subplot=legend_idx, framestyle=:none, xticks=nothing, yticks=nothing,
      xlims=(0,1), ylims=(0,1), legend=:inside, legendfontsize=7,
      legendcolumns=3, background_color_inside=:transparent,
      top_margin=-4Plots.mm, bottom_margin=-4Plots.mm)

# Entries are interleaved left-to-right so legendcolumns=3 produces:
#   col 1: reference lines  |  col 2: white noise (SNR)  |  col 3: colored noise (ρ)
# Pick 4 representative SNR and ρ levels (min, two intermediates, max).
i_snr = [1,
         round(Int, 1 + (n_SNR_grad - 1) / 3),
         round(Int, 1 + 2 * (n_SNR_grad - 1) / 3),
         n_SNR_grad]
i_ρ   = [1,
         round(Int, 1 + (n_ρ_grad - 1) / 3),
         round(Int, 1 + 2 * (n_ρ_grad - 1) / 3),
         n_ρ_grad]

# Row 1: column headers (reference is blank; white-noise and ρ headers are gray text lines)
plot!(fig, [NaN], [NaN]; subplot=legend_idx, label=" ",                         color=:white,   linewidth=0)
plot!(fig, [NaN], [NaN]; subplot=legend_idx, label="── white noise (SNR) ──",   color=:dimgray, linewidth=1.0)
plot!(fig, [NaN], [NaN]; subplot=legend_idx, label="── colored noise (ρ) ──",   color=:dimgray, linewidth=1.0)

# Rows 2–5: one reference entry, one SNR entry, one ρ entry per row
ref_entries = [
    () -> plot!(fig, [NaN], [NaN]; subplot=legend_idx, label="noiseless", color=:black, linewidth=2.2),
    () -> plot!(fig, [NaN], [NaN]; subplot=legend_idx, label="uniform",   color=:gray,  linewidth=0.8, linestyle=:dash),
    () -> plot!(fig, [NaN], [NaN]; subplot=legend_idx, label=L"\alpha=0.05", color=:gray, linewidth=1.0),
    () -> plot!(fig, [NaN], [NaN]; subplot=legend_idx, label=" ", color=:white, linewidth=0),
]
for k in 1:4
    ref_entries[k]()
    i = i_snr[k]
    snr_lbl = k == 1 ? "σ₀ = $(round(SNR_grad[i], sigdigits=2)) (min)" :
              k == 4 ? "σ₀ = $(round(SNR_grad[i], sigdigits=2)) (max)" :
                       "σ₀ = $(round(SNR_grad[i], sigdigits=2))"
    plot!(fig, [NaN], [NaN]; subplot=legend_idx, label=snr_lbl, color=SNR_colors[i], linewidth=1.5)
    j = i_ρ[k]
    ρ_lbl = k == 1 ? "ρ = $(round(ρ_grad[j], digits=2)) (min)" :
            k == 4 ? "ρ = $(round(ρ_grad[j], digits=2)) (max)" :
                     "ρ = $(round(ρ_grad[j], digits=2))"
    plot!(fig, [NaN], [NaN]; subplot=legend_idx, label=ρ_lbl, color=ρ_colors[j], linewidth=1.5)
end

savefig(fig, joinpath(figure_folder, "synthetic_majorization_combined.png"))
savefig(fig, joinpath(figure_folder, "synthetic_majorization_combined.pdf"))
println("Saved to $(figure_folder)/synthetic_majorization_combined.{png,pdf}")

# ============================================================
# CE plane figure: 2×3 panel (5 signals + legend)
# All signals reuse the already-computed x_* and σ_* from above
# ============================================================

println("Computing CH boundary curves...")
_ch_curves = (d, τ) -> let
    o_ec = OrdinalPatterns(; m=d, τ=τ)
    c_ec = StatisticalComplexity(; o=o_ec)
    min_raw, max_raw = entropy_complexity_curves(c_ec; num_max=1, num_min=1000)
    ord_max = sortperm([Float64(pt[1]) for pt in max_raw])
    ord_min = sortperm([Float64(pt[1]) for pt in min_raw])
    ([Float64(pt[1]) for pt in max_raw][ord_max], [Float64(pt[2]) for pt in max_raw][ord_max],
     [Float64(pt[1]) for pt in min_raw][ord_min], [Float64(pt[2]) for pt in min_raw][ord_min])
end

CH_henon_curves  = _ch_curves(d_henon,  τ_henon)
CH_logis_curves  = _ch_curves(d_logis,  τ_logis)
CH_lorenz_curves = _ch_curves(d_lorenz, τ_lorenz)
CH_ou_curves     = _ch_curves(d_ou,     τ_ou)
CH_fbm_curves    = _ch_curves(d_fbm,    τ_fbm)
println("Done.")

H_henon_c,  C_henon_c  = ordinal_entropy_complexity(x_henon;        d=d_henon,  τ=τ_henon)
H_logis_c,  C_logis_c  = ordinal_entropy_complexity(x_logis;        d=d_logis,  τ=τ_logis)
H_lorenz_c, C_lorenz_c = ordinal_entropy_complexity(x_lorenz_norm;  d=d_lorenz, τ=τ_lorenz)
H_ou_c,     C_ou_c     = ordinal_entropy_complexity(x_ou;           d=d_ou,     τ=τ_ou)
H_fbm_c,    C_fbm_c    = ordinal_entropy_complexity(x_fbm;          d=d_fbm,    τ=τ_fbm)

HC_henon_snr  = [ordinal_entropy_complexity(x_henon       .+ snr .* σ_henon  .* ε_henon;  d=d_henon,  τ=τ_henon)  for snr in SNR_grad]
HC_logis_snr  = [ordinal_entropy_complexity(x_logis       .+ snr .* σ_logis  .* ε_logis;  d=d_logis,  τ=τ_logis)  for snr in SNR_grad]
HC_lorenz_snr = [ordinal_entropy_complexity(x_lorenz_norm .+ snr .* σ_lorenz .* ε_lorenz; d=d_lorenz, τ=τ_lorenz) for snr in SNR_grad]
HC_ou_snr     = [ordinal_entropy_complexity(x_ou          .+ snr .* σ_ou     .* ε_ou;     d=d_ou,     τ=τ_ou)     for snr in SNR_grad]
HC_fbm_snr    = [ordinal_entropy_complexity(x_fbm         .+ snr .* σ_fbm    .* ε_fbm;    d=d_fbm,    τ=τ_fbm)    for snr in SNR_grad]

# Marker shape encodes SNR decade range; color encodes exact SNR value.
# Range is determined from the actual SNR value, not the step size.
_snr_exp_lo   = round(Int, log10(SNR_grad[1]))
_snr_exp_hi   = round(Int, log10(SNR_grad[end]))
_snr_n_ranges = _snr_exp_hi - _snr_exp_lo
_snr_all_markers  = [:circle, :diamond, :square, :utriangle, :dtriangle]
_snr_range_markers = _snr_all_markers[1:_snr_n_ranges]
_snr_marker = i -> _snr_range_markers[clamp(floor(Int, log10(SNR_grad[i])) - _snr_exp_lo + 1, 1, _snr_n_ranges)]

function draw_CH_panel(H0, C0, hc_snr, curves, title_str;
                       show_ylabel=true, show_xlabel=true)
    Hmax, Cmax, Hmin, Cmin = curves
    p = plot(dpi=300, legend=false,
             guidefontsize=10, tickfontsize=9,
             framestyle=:box, grid=:none,
             xlims=(0.0, 1.0), ylims=(0.0, 0.5),
             xlabel = show_xlabel ? L"H" : "",
             ylabel = show_ylabel ? L"C" : "",
             title  = title_str,
             left_margin   = show_ylabel ? 10Plots.mm : 1Plots.mm,
             right_margin  = 1Plots.mm,
             top_margin    = 4Plots.mm,
             bottom_margin = show_xlabel ? 8Plots.mm : 1Plots.mm)

    plot!(p, Hmax, Cmax; color=:black, linewidth=0.9, linestyle=:dash, label="")
    plot!(p, Hmin, Cmin; color=:black, linewidth=0.9, linestyle=:dash, label="")
    shade_admissible_noise!(p, H0, C0, Hmax, Cmax, Hmin, Cmin; color=:steelblue, alpha=0.20)

    Hs = [first(hc) for hc in hc_snr]
    Cs = [last(hc)  for hc in hc_snr]
    plot!(p, [H0; Hs], [C0; Cs]; color=:gray, linewidth=0.8, alpha=0.5, label="")
    for i in eachindex(SNR_grad)
        scatter!(p, [Hs[i]], [Cs[i]];
                 markercolor=SNR_colors[i], markerstrokecolor=SNR_colors[i],
                 markersize=5, markershape=_snr_marker(i), label="")
    end
    scatter!(p, [H0], [C0];
             markersize=10, markershape=:circle,
             markercolor=:black, markerstrokecolor=:white, markerstrokewidth=1.5, label="")
    return p
end

cp1 = draw_CH_panel(H_henon_c,  C_henon_c,  HC_henon_snr,  CH_henon_curves,  "H\u00e9non";              show_ylabel=true,  show_xlabel=false)
cp2 = draw_CH_panel(H_logis_c,  C_logis_c,  HC_logis_snr,  CH_logis_curves,  "Logistic Map";            show_ylabel=false, show_xlabel=false)
cp3 = draw_CH_panel(H_lorenz_c, C_lorenz_c, HC_lorenz_snr, CH_lorenz_curves, "Lorenz";                  show_ylabel=false, show_xlabel=false)
cp4 = draw_CH_panel(H_ou_c,     C_ou_c,     HC_ou_snr,     CH_ou_curves,     "Ornstein\u2013Uhlenbeck"; show_ylabel=true,  show_xlabel=true)
cp5 = draw_CH_panel(H_fbm_c,    C_fbm_c,    HC_fbm_snr,    CH_fbm_curves,    "fBm (H=$H_fbm)";          show_ylabel=false, show_xlabel=true)

# Zoom panel: OU process near H ~ 1, C ~ 0
cp_zoom = let
    Hmax, Cmax, Hmin, Cmin = CH_ou_curves
    Hs = [first(hc) for hc in HC_ou_snr]
    Cs = [last(hc)  for hc in HC_ou_snr]

    p = plot(dpi=300, legend=false,
             guidefontsize=10, tickfontsize=8,
             framestyle=:box, grid=:none,
             xlims=(0.92, 1.0), ylims=(0.0, 0.15),
             xlabel=L"H", ylabel="",
             title="Zoom OU",
             left_margin=1Plots.mm, right_margin=1Plots.mm,
             top_margin=4Plots.mm, bottom_margin=8Plots.mm)

    plot!(p, Hmax, Cmax; color=:black, linewidth=0.9, linestyle=:dash, label="")
    plot!(p, Hmin, Cmin; color=:black, linewidth=0.9, linestyle=:dash, label="")

    plot!(p, [H_ou_c; Hs], [C_ou_c; Cs]; color=:gray, linewidth=0.8, alpha=0.5, label="")
    for i in eachindex(SNR_grad)
        scatter!(p, [Hs[i]], [Cs[i]];
                 markercolor=SNR_colors[i], markerstrokecolor=SNR_colors[i],
                 markersize=5, markershape=_snr_marker(i), label="")
    end
    scatter!(p, [H_ou_c], [C_ou_c];
             markersize=10, markershape=:circle,
             markercolor=:black, markerstrokecolor=:white, markerstrokewidth=1.5, label="")
    p
end

cp_leg = plot(; framestyle=:none, xticks=nothing, yticks=nothing,
                xlims=(0,1), ylims=(0,1), legend=:inside, legendfontsize=8,
                background_color_inside=:transparent, dpi=300)
scatter!(cp_leg, [NaN], [NaN]; label="noiseless", markersize=9, markershape=:circle,
         markercolor=:black, markerstrokecolor=:white, markerstrokewidth=1.5)
plot!(cp_leg, [NaN], [NaN]; label="admissible region", color=:steelblue, linewidth=8, alpha=0.35)
plot!(cp_leg, [NaN], [NaN]; label="", color=:white, linewidth=0)
plot!(cp_leg, [NaN], [NaN]; label=L"$\sigma_0$ level", color=:black, linewidth=0)
for i in eachindex(SNR_grad)
    i % 3 == 1 || continue
    exp_int = round(Int, log10(SNR_grad[i]))
    lbl = latexstring("10^{$(exp_int)}")
    scatter!(cp_leg, [NaN], [NaN]; label=lbl, markercolor=SNR_colors[i],
             markerstrokecolor=SNR_colors[i], markersize=5, markershape=:circle)
end
plot!(cp_leg, [NaN], [NaN]; label="", color=:white, linewidth=0)
plot!(cp_leg, [NaN], [NaN]; label=L"$\sigma_0$ range", color=:black, linewidth=0)
for r in 1:_snr_n_ranges
    exp_lo = _snr_exp_lo + r - 1
    exp_hi = _snr_exp_lo + r
    lbl = latexstring("[10^{$(exp_lo)},\\,10^{$(exp_hi)}]")
    mid_idx = min((r - 1) * 3 + 2, n_SNR_grad)
    scatter!(cp_leg, [NaN], [NaN]; label=lbl, markersize=6,
             markershape=_snr_range_markers[r],
             markercolor=SNR_colors[mid_idx], markerstrokecolor=SNR_colors[mid_idx])
end

fig_ce = plot(cp1, cp2, cp3, cp4, cp_zoom, cp5, cp_leg;
              layout = @layout([grid(2, 3) c{0.15w}]),
              size   = (1050, 600), dpi=600)

savefig(fig_ce, joinpath(figure_folder, "CE_plane_fivepanel.png"))
savefig(fig_ce, joinpath(figure_folder, "CE_plane_fivepanel.pdf"))
println("Saved to $(figure_folder)/CE_plane_fivepanel.{png,pdf}")

# ------------------------------------------------------------
# H and C vs SNR — one line per signal
# ------------------------------------------------------------

signals_ce = [
    ("H\u00e9non",              H_henon_c,  C_henon_c,  HC_henon_snr),
    ("Logistic Map",            H_logis_c,  C_logis_c,  HC_logis_snr),
    ("Lorenz",                  H_lorenz_c, C_lorenz_c, HC_lorenz_snr),
    ("Ornstein\u2013Uhlenbeck", H_ou_c,     C_ou_c,     HC_ou_snr),
    ("fBm (H=$H_fbm)",          H_fbm_c,    C_fbm_c,    HC_fbm_snr),
]
sig_colors = [:crimson, :royalblue, :darkorange, :seagreen, :purple]

# Log-scale x-axis ticks: major at powers of 10 (labelled), minor at 2×–9× multiples (unlabelled)
_log_major_vals = [10.0^e for e in _snr_exp_lo:_snr_exp_hi]
_log_minor_vals = sort(vcat([Float64(k) * 10.0^e
                              for e in _snr_exp_lo:(_snr_exp_hi - 1)
                              for k in 2:9]...))
_log_all_vals   = sort(vcat(_log_major_vals, _log_minor_vals))
_log_all_lbls   = [v in _log_major_vals ?
                   latexstring("10^{$(round(Int, log10(v)))}") : ""
                   for v in _log_all_vals]
_snr_xticks = (_log_all_vals, _log_all_lbls)

p_H = plot(dpi=300, legend=false, framestyle=:box, grid=:none,
           xlabel=L"Noise level ($\sigma_0$)", ylabel=L"H",
           guidefontsize=11, tickfontsize=9, legendfontsize=8,
           xscale=:log10, ylims=(0.5, 1.0), xticks=_snr_xticks, margin=5Plots.mm)
p_C = plot(dpi=300, legend=false, framestyle=:box, grid=:none,
           xlabel=L"Noise level ($\sigma_0$)", ylabel=L"C",
           guidefontsize=11, tickfontsize=9, legendfontsize=8,
           xscale=:log10, ylims=(0.0, 0.5), xticks=_snr_xticks, margin=5Plots.mm)

for ((lbl, H0, C0, hc_snr), col) in zip(signals_ce, sig_colors)
    Hs = [first(hc) for hc in hc_snr]
    Cs = [last(hc)  for hc in hc_snr]
    plot!(p_H, SNR_grad, Hs; label=lbl, color=col, linewidth=1.8)
    scatter!(p_H, SNR_grad, Hs; markercolor=col, markerstrokecolor=col, markersize=4,
             markershape=[_snr_marker(i) for i in eachindex(SNR_grad)], label="")
    plot!(p_C, SNR_grad, Cs; label=lbl, color=col, linewidth=1.8)
    scatter!(p_C, SNR_grad, Cs; markercolor=col, markerstrokecolor=col, markersize=4,
             markershape=[_snr_marker(i) for i in eachindex(SNR_grad)], label="")
end

p_leg_HC = plot(; framestyle=:none, ticks=nothing,
                  xlims=(0,1), ylims=(0,1), legend=:inside, legendfontsize=9,
                  background_color_inside=:transparent, dpi=300)
for ((lbl, _, _, _), col) in zip(signals_ce, sig_colors)
    plot!(p_leg_HC, [NaN], [NaN]; label=lbl, color=col, linewidth=2)
end

fig_HC_snr = plot(p_H, p_C, p_leg_HC;
                  layout=@layout([grid(1, 2) c{0.18w}]),
                  size=(1200, 400), dpi=600)

savefig(fig_HC_snr, joinpath(figure_folder, "CE_vs_sigma.png"))
savefig(fig_HC_snr, joinpath(figure_folder, "CE_vs_sigma.pdf"))
println("Saved to $(figure_folder)/CE_vs_sigma.{png,pdf}")
