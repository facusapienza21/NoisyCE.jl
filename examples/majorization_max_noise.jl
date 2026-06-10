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
using Brownian

default(fontfamily = "Computer Modern")

# ------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------

const figure_folder = joinpath(@__DIR__, "figures")
mkpath(figure_folder)

# const B_boot = 10
# const N_rep  = 10    # repetitions for the crossing-SNR histograms
const B_boot = 1000
const N_rep  = 100    # repetitions for the crossing-SNR histograms

# Per-dataset embedding parameters (same as example_majorization.jl)
d_henon  = 5;  τ_henon  = 1;  δt_lorenz = 0.1;  δt_ou = 0.1
d_logis  = 6;  τ_logis  = 1
d_lorenz = 5;  τ_lorenz = 1
d_ou     = 5;  τ_ou     = 1
d_fbm    = 4;  τ_fbm    = 1

H_fbm = 0.91

# True noise level and SNR sweep
const SNR_obs   = 0.1
SNR_sweep       = collect(range(0.0, 0.20; length=41))   # row 1: fine grid
SNR_sweep_cross = collect(range(0.0, 0.20; length=21))   # row 2: coarser grid
n_SNR           = length(SNR_sweep)

α_level = 0.05

# N values to compare
N_values  = [2000, 4000, 8000]
N_labels  = ["N = 2 000", "N = 4 000", "N = 8 000"]
N_colors  = [RGB(0.161, 0.502, 0.725),   # blue
             RGB(0.898, 0.553, 0.000),   # orange
             RGB(0.753, 0.224, 0.169)]   # red
N_lstyles = [:dot, :dash, :solid]
N_lwidths = [2.2, 2.5, 3.0]

# ------------------------------------------------------------
# Ordinal utilities
# ------------------------------------------------------------

_ord_probs(x, d, τ) = vec(first(allprobabilities_and_outcomes(OrdinalPatterns(; m=d, τ=τ), x)))
_n_pat(N, d, τ)     = N - (d - 1) * τ

# ------------------------------------------------------------
# Signal generators (parameterised by N and an offset/seed for different realisations)
# ------------------------------------------------------------

function gen_henon(N_i, rep=0; a=1.4, b=0.3)
    rng_ic = MersenneTwister(rep)
    x = (2 * rand(rng_ic) - 1) + 1e-6 * randn(rng_ic)   # base ∈ (-1, 1)
    y = (rand(rng_ic) - 0.5)   + 1e-6 * randn(rng_ic)   # base ∈ (-0.5, 0.5)
    for _ in 1:10_000; x, y = 1 - a*x^2 + y, b*x; end
    xs = zeros(N_i); ys = zeros(N_i)
    xs[1] = x; ys[1] = y
    for i in 2:N_i
        xs[i] = 1 - a*xs[i-1]^2 + ys[i-1]
        ys[i] = b*xs[i-1]
    end
    xs, std(xs)
end

function gen_logis(N_i, rep=0; r=4.0)
    rng_ic = MersenneTwister(rep)
    x      = rand(rng_ic) + 1e-6 * randn(rng_ic)   # base ∈ (0, 1)
    x      = clamp(x, 1e-10, 1 - 1e-10)
    for _ in 1:10_000; x = r * x * (1 - x); end
    xs = zeros(N_i); xs[1] = x
    for i in 2:N_i; xs[i] = r * xs[i-1] * (1 - xs[i-1]); end
    xs, std(xs)
end

function gen_lorenz(N_i, rep=0)
    function lorenz!(du, u, p, _)
        σ_l, ρ, β = p
        du[1] = σ_l * (u[2] - u[1])
        du[2] = u[1] * (ρ - u[3]) - u[2]
        du[3] = u[1] * u[2] - β * u[3]
    end
    rng_ic = MersenneTwister(rep)
    u0     = [1.0, 1.0, 1.0] .+ 1e-6 .* randn(rng_ic, 3)
    K_burn = trunc(Int, 100.0 / δt_lorenz)
    sol    = solve(ODEProblem(lorenz!, u0,
                              (0.0, (K_burn + N_i - 1) * δt_lorenz), (10.0, 28.0, 8/3)),
                   Tsit5(); reltol=1e-12, abstol=1e-12, saveat=δt_lorenz, maxiters=10_000*N_i)
    x_raw = Array(sol)[1, (K_burn+1):end]
    @assert length(x_raw) == N_i
    x_raw, std(x_raw)
end

function gen_ou(N_i, rep=0)
    sol = solve(SDEProblem((u, p, _) -> -p * u, (_, p, _) -> 1.0,
                           0.0, (0.0, (N_i - 1) * δt_ou), 1.0),
                EM(); dt=δt_ou, saveat=δt_ou, seed=1234 + rep)
    sol.u, std(sol.u)
end

function gen_fbm(N_i, rep=0)
    Random.seed!(999 * (rep + 1))
    p     = FBM(0.0:1.0:(N_i), H_fbm)
    x_raw = rand(p)[begin:end-1]
    x_raw, 0.01 * std(x_raw)
end

# ------------------------------------------------------------
# Main computation loop
# pvals_by_N[ni][sig]      = p-value curve (row 1)
# crossing_by_N[ni][sig]   = crossing SNR per repetition (row 2)
# ------------------------------------------------------------

pvals_by_N   = Vector{Vector{Vector{Float64}}}(undef, length(N_values))
crossing_by_N = [[fill(NaN, N_rep) for _ in 1:5] for _ in eachindex(N_values)]

for (ni, N_i) in enumerate(N_values)
    println("\n── N = $N_i ──────────────────────────────────────────")

    # ---- single-realization p-value sweep (rep=0, row 1) ----
    x_h, σ_h = gen_henon(N_i, 0)
    x_q, σ_q = gen_logis(N_i, 0)
    x_l, σ_l = gen_lorenz(N_i, 0)
    x_o, σ_o = gen_ou(N_i, 0)
    x_f, σ_f = gen_fbm(N_i, 0)

    rng_obs = MersenneTwister(123)
    prob_h_obs = _ord_probs(x_h .+ SNR_obs .* σ_h .* randn(rng_obs, N_i), d_henon,  τ_henon)
    prob_q_obs = _ord_probs(x_q .+ SNR_obs .* σ_q .* randn(rng_obs, N_i), d_logis,  τ_logis)
    prob_l_obs = _ord_probs(x_l .+ SNR_obs .* σ_l .* randn(rng_obs, N_i), d_lorenz, τ_lorenz)
    prob_o_obs = _ord_probs(x_o .+ SNR_obs .* σ_o .* randn(rng_obs, N_i), d_ou,     τ_ou)
    prob_f_obs = _ord_probs(x_f .+ SNR_obs .* σ_f .* randn(rng_obs, N_i), d_fbm,    τ_fbm)

    n_h = _n_pat(N_i, d_henon,  τ_henon);  n_q = _n_pat(N_i, d_logis,  τ_logis)
    n_l = _n_pat(N_i, d_lorenz, τ_lorenz); n_o = _n_pat(N_i, d_ou,     τ_ou)
    n_f = _n_pat(N_i, d_fbm,    τ_fbm)

    rng_sw = MersenneTwister(42)
    ε_h = randn(rng_sw, N_i); ε_q = randn(rng_sw, N_i); ε_l = randn(rng_sw, N_i)
    ε_o = randn(rng_sw, N_i); ε_f = randn(rng_sw, N_i)

    pv_h = Vector{Float64}(undef, n_SNR); pv_q = Vector{Float64}(undef, n_SNR)
    pv_l = Vector{Float64}(undef, n_SNR); pv_o = Vector{Float64}(undef, n_SNR)
    pv_f = Vector{Float64}(undef, n_SNR)

    print("  p-value sweep: ")
    for (j, snr) in enumerate(SNR_sweep)
        ph = _ord_probs(x_h .+ snr .* σ_h .* ε_h, d_henon,  τ_henon)
        pq = _ord_probs(x_q .+ snr .* σ_q .* ε_q, d_logis,  τ_logis)
        pl = _ord_probs(x_l .+ snr .* σ_l .* ε_l, d_lorenz, τ_lorenz)
        po = _ord_probs(x_o .+ snr .* σ_o .* ε_o, d_ou,     τ_ou)
        pf = _ord_probs(x_f .+ snr .* σ_f .* ε_f, d_fbm,    τ_fbm)
        pv_h[j] = majorization_test(ph, prob_h_obs, n_h, n_h; B=B_boot).p_value
        pv_q[j] = majorization_test(pq, prob_q_obs, n_q, n_q; B=B_boot).p_value
        pv_l[j] = majorization_test(pl, prob_l_obs, n_l, n_l; B=B_boot).p_value
        pv_o[j] = majorization_test(po, prob_o_obs, n_o, n_o; B=B_boot).p_value
        pv_f[j] = majorization_test(pf, prob_f_obs, n_f, n_f; B=B_boot).p_value
        print("\r  p-value sweep: $(j)/$(n_SNR)")
    end
    println("  done.")
    pvals_by_N[ni] = [pv_h, pv_q, pv_l, pv_o, pv_f]

    # ---- crossing-SNR over N_rep independent realizations (row 2) ----
    println("  Crossing SNRs ($N_rep reps):")
    for rep in 1:N_rep
        xr_h, σr_h = gen_henon(N_i, rep)
        xr_q, σr_q = gen_logis(N_i, rep)
        xr_l, σr_l = gen_lorenz(N_i, rep)
        xr_o, σr_o = gen_ou(N_i, rep)
        xr_f, σr_f = gen_fbm(N_i, rep)

        rng_obs_r = MersenneTwister(123 + rep * 1000)
        pr_h_obs = _ord_probs(xr_h .+ SNR_obs .* σr_h .* randn(rng_obs_r, N_i), d_henon,  τ_henon)
        pr_q_obs = _ord_probs(xr_q .+ SNR_obs .* σr_q .* randn(rng_obs_r, N_i), d_logis,  τ_logis)
        pr_l_obs = _ord_probs(xr_l .+ SNR_obs .* σr_l .* randn(rng_obs_r, N_i), d_lorenz, τ_lorenz)
        pr_o_obs = _ord_probs(xr_o .+ SNR_obs .* σr_o .* randn(rng_obs_r, N_i), d_ou,     τ_ou)
        pr_f_obs = _ord_probs(xr_f .+ SNR_obs .* σr_f .* randn(rng_obs_r, N_i), d_fbm,    τ_fbm)

        nr_h = _n_pat(N_i, d_henon,  τ_henon);  nr_q = _n_pat(N_i, d_logis,  τ_logis)
        nr_l = _n_pat(N_i, d_lorenz, τ_lorenz); nr_o = _n_pat(N_i, d_ou,     τ_ou)
        nr_f = _n_pat(N_i, d_fbm,    τ_fbm)

        rng_sw_r = MersenneTwister(42 + rep * 1000)
        εr_h = randn(rng_sw_r, N_i); εr_q = randn(rng_sw_r, N_i); εr_l = randn(rng_sw_r, N_i)
        εr_o = randn(rng_sw_r, N_i); εr_f = randn(rng_sw_r, N_i)

        sigs = [(xr_h, σr_h, εr_h, pr_h_obs, nr_h, d_henon,  τ_henon),
                (xr_q, σr_q, εr_q, pr_q_obs, nr_q, d_logis,  τ_logis),
                (xr_l, σr_l, εr_l, pr_l_obs, nr_l, d_lorenz, τ_lorenz),
                (xr_o, σr_o, εr_o, pr_o_obs, nr_o, d_ou,     τ_ou),
                (xr_f, σr_f, εr_f, pr_f_obs, nr_f, d_fbm,    τ_fbm)]

        for (si, (xc, σc, εc, p_obs, nc, dc, τc)) in enumerate(sigs)
            pv_rep = [majorization_test(_ord_probs(xc .+ snr .* σc .* εc, dc, τc),
                                        p_obs, nc, nc; B=B_boot).p_value
                      for snr in SNR_sweep_cross[2:end]]   # skip snr=0 (trivially passes)
            rejected = SNR_sweep_cross[2:end][pv_rep .< α_level]
            crossing_by_N[ni][si][rep] = isempty(rejected) ? NaN : minimum(rejected)
        end
        print("\r  rep $(rep)/$(N_rep)")
    end
    println("  done.")
end

# ------------------------------------------------------------
# Figure: 2 rows × 5 panels + right legend column
#   row 1 (sp 1–5)  : p-value vs noise level
#   row 2 (sp 6–10) : histogram of crossing noise level
#   sp 11           : legend
# ------------------------------------------------------------

signal_titles = ["Hénon", "Logistic Map", "Lorenz", "Ornstein–Uhlenbeck", "fBm  (H=$H_fbm)"]
signal_emb    = ["d=$d_henon, τ=$τ_henon", "d=$d_logis, τ=$τ_logis",
                 "d=$d_lorenz, τ=$τ_lorenz", "d=$d_ou, τ=$τ_ou", "d=$d_fbm, τ=$τ_fbm"]

hist_bins = range(0.0, 0.20; step=0.01)

fig = plot(layout=@layout([[grid(1, 5){0.45h}; grid(3, 5)] b{0.10w}]),
           size=(1980, 850), dpi=300,
           guidefontsize=9, tickfontsize=8, legendfontsize=9,
           framestyle=:box, grid=:none,
           left_margin=6Plots.mm, right_margin=0Plots.mm,
           top_margin=2Plots.mm, bottom_margin=4Plots.mm,
           legend=false)

for sp in 1:5
    fp = sp == 1
    # ---- row 1: p-value curves (subplots 1–5) ----
    hline!(fig, [α_level]; subplot=sp, color=:gray, linewidth=1.0, label="")
    plot!(fig, Shape([-0.005, 0.21, 0.21, -0.005], [0.0, 0.0, α_level, α_level]);
          subplot=sp, fillcolor=:gray, fillalpha=0.2, linewidth=0, label="")
    vline!(fig, [SNR_obs]; subplot=sp, color=:black, linewidth=1.4, linestyle=:dot, label="")
    for (ni, N_i) in enumerate(N_values)
        plot!(fig, SNR_sweep, pvals_by_N[ni][sp];
              subplot=sp, color=N_colors[ni], linewidth=N_lwidths[ni],
              linestyle=N_lstyles[ni], label="")
    end
    annotate!(fig, 0.195, 0.97, text(signal_emb[sp], 7, :black, :right); subplot=sp)
    plot!(fig; subplot=sp,
          xlims=(0.0, 0.20), ylims=(-0.04, 1.04),
          xlabel="", ylabel=fp ? L"$p$-value" : "",
          yformatter=fp ? :auto : _->"",
          left_margin=fp ? 14Plots.mm : 0Plots.mm,
          right_margin=sp < 5 ? -10Plots.mm : 2Plots.mm,
          legend=false, title=signal_titles[sp])

    # ---- rows 2–4: one thin histogram per N value (subplots 6–20) ----
    # ni=1 → subplots 6–10, ni=2 → 11–15, ni=3 → 16–20
    for (ni, N_i) in enumerate(N_values)
        sp_h    = 5 * ni + sp
        bot_row = ni == length(N_values)
        top_row = ni == 1
        vals    = filter(!isnan, crossing_by_N[ni][sp])

        vline!(fig, [SNR_obs]; subplot=sp_h, color=:black, linewidth=1.4, linestyle=:dot, label="")
        if !isempty(vals)
            histogram!(fig, vals; subplot=sp_h, bins=hist_bins, normalize=:pdf,
                       fillcolor=N_colors[ni], linecolor=:white, fillalpha=0.6,
                       linewidth=0.5, label="")
            vline!(fig, [mean(vals)]; subplot=sp_h, color=N_colors[ni],
                   linewidth=4.0, linestyle=:solid, label="")
        end
        plot!(fig; subplot=sp_h,
              xlims=(0.0, 0.20),
              xlabel=bot_row ? L"Noise level ($\sigma_0$)" : "",
              ylabel=fp ? N_labels[ni] : "",
              xformatter=bot_row ? :auto : _->"",
              yformatter=fp ? :auto : _->"",
              left_margin=fp ? 14Plots.mm : 0Plots.mm,
              right_margin=sp < 5 ? -10Plots.mm : 2Plots.mm,
              top_margin=top_row ? 0Plots.mm : -4Plots.mm,
              bottom_margin=bot_row ? 6Plots.mm : -4Plots.mm,
              legend=false)
    end
end

# ---- legend panel (subplot 21) ----
plot!(fig; subplot=21, framestyle=:none, xticks=nothing, yticks=nothing,
      xlims=(0, 1), ylims=(0, 1), legend=:inside, legendfontsize=10,
      legendfontfamily="Computer Modern", background_color_inside=:transparent,
      left_margin=-8Plots.mm,
      legendtitle="Time series length", legendtitlefontsize=10,
      legendtitlefontfamily="Computer Modern")
for (ni, N_i) in enumerate(N_values)
    plot!(fig, [NaN], [NaN]; subplot=21,
          color=N_colors[ni], linewidth=N_lwidths[ni], linestyle=N_lstyles[ni],
          label=N_labels[ni])
end

savefig(fig, joinpath(figure_folder, "majorization_max_noise.png"))
savefig(fig, joinpath(figure_folder, "majorization_max_noise.pdf"))
println("Saved to $(figure_folder)/majorization_max_noise.{png,pdf}")
