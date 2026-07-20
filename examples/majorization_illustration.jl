# ============================================================
# Generates:
#   - Figure 1 (illustration),
# Author : Facundo Sapienza
# Date   : 2026-07-20
# ============================================================

ENV["GKSwstype"] = "100"

import Pkg
Pkg.activate(@__DIR__)

using NoisyCE
using Plots
using ComplexityMeasures
using Distributions
using Statistics
using Random

const figure_folder = joinpath(@__DIR__, "figures")
isdir(figure_folder) || mkpath(figure_folder)

const d = 5
const τ = 3
const N = 10_000

# Colors: black for noiseless, purple for noisy (matches col_sho in synthetic_CH)
const col_clean = :black
const col_noise = RGB(0.580, 0.404, 0.741)

# α for points outside the interior d-window
const α_outside = 0.20

# ------------------------------------------------------------
# Hénon map (same parameters as example_majorization.jl)
# ------------------------------------------------------------

x_henon, σ_henon = let a = 1.4, b = 0.3, n_burn = 10_000
    x, y = 0.0, 0.0
    for _ in 1:n_burn
        x, y = 1 - a*x^2 + y, b*x
    end
    xs = zeros(N)
    ys = zeros(N)
    xs[1] = x
    ys[1] = y
    for i in 2:N
        xs[i] = 1 - a*xs[i-1]^2 + ys[i-1]
        ys[i] = b*xs[i-1]
    end
    xs, std(xs)
end

rng   = MersenneTwister(2025)
ε_all = randn(rng, N)

SNR_illus = 0.10   # increased noise for clear visual contrast

# ------------------------------------------------------------
# Figure 1: time series illustration with τ=4
# With d=5, τ=3: one ordinal window uses d=5 points spaced τ=3 apart,
# spanning (d-1)*τ + 1 = 13 consecutive time steps.
# Show n_show = 5 + (d-1)*τ = 17 data points → 5 ordinal windows.
# Highlight the middle window (k=3): τ-spaced points at 3,6,9,12,15.
# These are connected with a line to make the sub-sampling structure visible.
# Remaining 12 points are shown faded (α_outside).
# ------------------------------------------------------------

let
    n_show    = 17
    start_idx = 115
    idx_show  = start_idx:(start_idx + n_show - 1)
    x_show    = x_henon[idx_show]
    x_noisy   = x_henon[idx_show] .+ SNR_illus .* σ_henon .* ε_all[idx_show]

    ts_idx  = 1:n_show
    int_idx = [3, 6, 9, 12, 15]   # middle τ=3 ordinal window (k=3, d=5)

    lw_clean = 2.0
    lw_noisy = 1.8
    ms_clean = 7
    ms_noisy = 6

    p = plot(
        size=(750, 280), dpi=300,
        guidefontsize=11, tickfontsize=8,
        framestyle=:box, grid=:none,
        margin=5Plots.mm, legend=false,
        xlabel="Iteration", ylabel="x",
        xticks=1:n_show,
        ylims=(-2.0, 2.0),
    )

    # Clean: full faded line + scatter, then interior τ-spaced points at full alpha
    plot!(p, ts_idx, x_show;
          color=col_clean, linewidth=lw_clean, alpha=α_outside, label="")
    scatter!(p, ts_idx, x_show;
             markercolor=col_clean, markerstrokecolor=col_clean,
             markersize=ms_clean, alpha=α_outside, label="")
    plot!(p, int_idx, x_show[int_idx];
          color=col_clean, linewidth=lw_clean, alpha=1.0, label="")
    scatter!(p, int_idx, x_show[int_idx];
             markercolor=col_clean, markerstrokecolor=col_clean,
             markersize=ms_clean, alpha=1.0, label="")

    # Noisy (purple): same layered approach
    plot!(p, ts_idx, x_noisy;
          color=col_noise, linewidth=lw_noisy, alpha=α_outside, label="")
    scatter!(p, ts_idx, x_noisy;
             markercolor=col_noise, markerstrokecolor=col_noise,
             markersize=ms_noisy, alpha=α_outside, label="")
    plot!(p, int_idx, x_noisy[int_idx];
          color=col_noise, linewidth=lw_noisy, alpha=1.0, label="")
    scatter!(p, int_idx, x_noisy[int_idx];
             markercolor=col_noise, markerstrokecolor=col_noise,
             markersize=ms_noisy, alpha=1.0, label="")

    # Rank annotations: ordinal ordering of the d=5 interior points.
    # sortperm(sortperm(v)) gives rank 1 (smallest) … d (largest) for each element.
    ranks_clean = sortperm(sortperm(x_show[int_idx]))
    ranks_noisy = sortperm(sortperm(x_noisy[int_idx]))

    Δy = 0.12 * 4.0   # 12% of fixed ylims range [-2, 2]
    for (k, i) in enumerate(int_idx)
        if x_show[i] >= x_noisy[i]
            annotate!(p, i, x_show[i]  + Δy, text(string(ranks_clean[k]), 10, :black,    :center))
            annotate!(p, i, x_noisy[i] - Δy, text(string(ranks_noisy[k]), 10, col_noise, :center))
        else
            annotate!(p, i, x_noisy[i] + Δy, text(string(ranks_noisy[k]), 10, col_noise, :center))
            annotate!(p, i, x_show[i]  - Δy, text(string(ranks_clean[k]), 10, :black,    :center))
        end
    end

    savefig(p, joinpath(figure_folder, "illustration_timeseries.pdf"))
    println("Saved illustration_timeseries.pdf")
end

# ------------------------------------------------------------
# Ordinal pattern probabilities from the full N=10000 series
# ------------------------------------------------------------

ord_est = OrdinalPatterns(; m=d, τ=τ)

prob_clean = vec(first(allprobabilities_and_outcomes(ord_est, x_henon)))

x_henon_noisy = x_henon .+ SNR_illus .* σ_henon .* ε_all
prob_noisy    = vec(first(allprobabilities_and_outcomes(ord_est, x_henon_noisy)))

n_pat      = factorial(d)          # 120
ranks_norm = collect(0:n_pat) ./ n_pat
maj_ext(p) = [0.0; majorization_curve(p)]

# ------------------------------------------------------------
# Figure 2: Majorization curves — noiseless (black) and noisy (purple)
# Style follows the second column of synthetic_majorization_combined.
# ------------------------------------------------------------

let
    p = plot(
        size=(420, 420), dpi=300,
        guidefontsize=11, tickfontsize=9,
        framestyle=:box, grid=:none,
        margin=5Plots.mm, legend=false,
        aspect_ratio=1,
        xlims=(0.0, 1.0), ylims=(0.0, 1.0),
        xlabel="Pattern rank", ylabel="Accumulated probability",
    )

    plot!(p, ranks_norm, ranks_norm;
          color=:gray, linewidth=0.9, linestyle=:dash, label="")
    plot!(p, ranks_norm, maj_ext(prob_noisy);
          color=col_noise, linewidth=2.0, label="")
    plot!(p, ranks_norm, maj_ext(prob_clean);
          color=col_clean, linewidth=2.5, label="")

    savefig(p, joinpath(figure_folder, "illustration_majorization.pdf"))
    println("Saved illustration_majorization.pdf")
end

# ------------------------------------------------------------
# CH boundary curves (d=5, τ=3) — computed once for Figure 3
# ------------------------------------------------------------

println("Computing CH boundary curves (d=$d, τ=$τ)...")
H_max_curve, C_max_curve, H_min_curve, C_min_curve = let
    o_ec = OrdinalPatterns(; m=d, τ=τ)
    c_ec = StatisticalComplexity(; o=o_ec)
    min_raw, max_raw = entropy_complexity_curves(c_ec; num_max=1, num_min=1000)
    ord_max = sortperm([Float64(pt[1]) for pt in max_raw])
    ord_min = sortperm([Float64(pt[1]) for pt in min_raw])
    ([Float64(pt[1]) for pt in max_raw][ord_max],
     [Float64(pt[2]) for pt in max_raw][ord_max],
     [Float64(pt[1]) for pt in min_raw][ord_min],
     [Float64(pt[2]) for pt in min_raw][ord_min])
end

# H_clean, C_clean = ordinal_entropy_complexity(x_henon;       d=d, τ=τ)
# H_noisy, C_noisy = ordinal_entropy_complexity(x_henon_noisy; d=d, τ=τ)
H_clean, C_clean = 0.6, 0.3
H_noisy, C_noisy = 0.70, 0.25
println("  noiseless : H=$(round(H_clean, digits=4))  C=$(round(C_clean, digits=4))")
println("  noisy     : H=$(round(H_noisy, digits=4))  C=$(round(C_noisy, digits=4))")

# ------------------------------------------------------------
# Figure 3: Complexity-entropy plane
# ------------------------------------------------------------

let
    p = plot(
        size=(420, 370), dpi=300,
        guidefontsize=11, tickfontsize=9,
        framestyle=:box, grid=:none,
        margin=5Plots.mm, legend=false,
        xlims=(0.0, 1.0), ylims=(0.0, 0.45),
        xlabel="Entropy", ylabel="Complexity",
    )

    plot!(p, H_max_curve, C_max_curve; color=:black, linewidth=0.9, linestyle=:dash, label="")
    plot!(p, H_min_curve, C_min_curve; color=:black, linewidth=0.9, linestyle=:dash, label="")

    shade_admissible_noise!(p, H_clean, C_clean,
                            H_max_curve, C_max_curve, H_min_curve, C_min_curve;
                            color=col_noise, alpha=0.25)

    scatter!(p, [H_noisy], [C_noisy];
             markercolor=col_noise, markerstrokecolor=col_noise,
             markersize=8, markershape=:circle, label="")
    scatter!(p, [H_clean], [C_clean];
             markercolor=col_clean, markerstrokecolor=:white,
             markersize=10, markershape=:circle, markerstrokewidth=1.5, label="")

    savefig(p, joinpath(figure_folder, "illustration_CH_plane.pdf"))
    println("Saved illustration_CH_plane.pdf")
end

# ------------------------------------------------------------
# Helper: multinomial draw (sequential conditional-Binomial)
# ------------------------------------------------------------

function _draw_multinomial(p::AbstractVector, n::Int)
    d = length(p)
    counts = zeros(Float64, d)
    rem = n
    cum = 0.0
    for i in 1:d-1
        rem == 0 && break
        pc = clamp(p[i] / (1.0 - cum), 0.0, 1.0)
        counts[i] = rand(Binomial(rem, pc))
        rem -= Int(counts[i])
        cum += p[i]
    end
    counts[d] = rem
    return counts
end

# ------------------------------------------------------------
# Figures 4a & 4b: Hypothesis test illustration
# Use N_test-point subseries so bootstrap variability is visible.
# ------------------------------------------------------------

let
    N_test = 1000

    x_sub       = x_henon[1:N_test]
    x_sub_noisy = x_sub .+ SNR_illus .* σ_henon .* ε_all[1:N_test]
    prob_ci = vec(first(allprobabilities_and_outcomes(ord_est, x_sub)))
    prob_ni = vec(first(allprobabilities_and_outcomes(ord_est, x_sub_noisy)))

    # ---- Figure 4a: a handful of bootstrap resamples of both majorization curves ----
    B_show = 10
    Random.seed!(42)

    # Pre-generate resamples so the same curves appear in the main plot and the inset
    clean_stars = [[0.0; majorization_curve(_draw_multinomial(prob_ci, N_test) ./ N_test)]
                   for _ in 1:B_show]
    noisy_stars = [[0.0; majorization_curve(_draw_multinomial(prob_ni, N_test) ./ N_test)]
                   for _ in 1:B_show]

    p4a = plot(
        size=(420, 420), dpi=300,
        guidefontsize=11, tickfontsize=9,
        framestyle=:box, grid=:none,
        margin=5Plots.mm, legend=false,
        aspect_ratio=1,
        xlims=(0.0, 1.0), ylims=(0.0, 1.0),
        xlabel="Pattern rank", ylabel="Accumulated probability",
    )
    plot!(p4a, ranks_norm, ranks_norm; color=:gray, linewidth=0.9, linestyle=:dash, label="")

    for maj_star in clean_stars
        plot!(p4a, ranks_norm, maj_star; color=col_clean, linewidth=1.2, alpha=0.4, label="")
    end
    for maj_star in noisy_stars
        plot!(p4a, ranks_norm, maj_star; color=col_noise, linewidth=1.2, alpha=0.4, label="")
    end

    plot!(p4a, ranks_norm, maj_ext(prob_ni); color=col_noise, linewidth=2.0, label="")
    plot!(p4a, ranks_norm, maj_ext(prob_ci); color=col_clean, linewidth=2.5, label="")

    # Zoom region shown in the inset
    x_z1, x_z2 = 0.30, 0.65
    y_z1, y_z2 = 0.65, 0.90

    # Dashed rectangle on the main plot indicating the zoomed region
    plot!(p4a, Shape([x_z1, x_z2, x_z2, x_z1], [y_z1, y_z1, y_z2, y_z2]);
          subplot=1, fillcolor=:transparent, linecolor=:gray40, linewidth=1.0, label="")

    # Inset subplot in the bottom-right corner
    plot!(p4a, Float64[], Float64[];
          inset=(1, bbox(0.05, 0.05, 0.40, 0.36, :right, :bottom)),
          subplot=2,
          xlims=(x_z1, x_z2), ylims=(y_z1, y_z2),
          framestyle=:box, grid=:none,
          xticks=false, yticks=false,
          background_color_inside=:white,
          label="")

    for maj_star in clean_stars
        plot!(p4a, ranks_norm, maj_star; subplot=2, color=col_clean, linewidth=1.5, alpha=0.4, label="")
    end
    for maj_star in noisy_stars
        plot!(p4a, ranks_norm, maj_star; subplot=2, color=col_noise, linewidth=1.5, alpha=0.4, label="")
    end
    plot!(p4a, ranks_norm, maj_ext(prob_ni); subplot=2, color=col_noise, linewidth=2.0, label="")
    plot!(p4a, ranks_norm, maj_ext(prob_ci); subplot=2, color=col_clean, linewidth=2.5, label="")

    savefig(p4a, joinpath(figure_folder, "illustration_resampling.pdf"))
    println("Saved illustration_resampling.pdf")

    # ---- Figure 4b: GMS bootstrap distribution and p-value ----
    # Replicates the actual majorization_test statistic:
    #   T = min_k  (S_k(p̂) - S_k(q̂)) / σ̂_k  over GMS-selected valid constraints.
    # T_boot[b] is the same statistic on multinomial resamples, centred at 0.
    # p-value = fraction of T_boot ≤ T_obs.
    n1 = n2 = N_test
    B  = 5_000
    Random.seed!(123)

    d_len = length(prob_ci)
    Sp    = majorization_curve(prob_ci)
    Sq    = majorization_curve(prob_ni)
    gaps  = Sp[1:d_len-1] .- Sq[1:d_len-1]
    n_eff = n1 * n2 / (n1 + n2)
    V̂     = Sp[1:d_len-1] .* (1 .- Sp[1:d_len-1]) ./ n1 .+
             Sq[1:d_len-1] .* (1 .- Sq[1:d_len-1]) ./ n2
    σ̂     = max.(sqrt.(max.(V̂, 0.0)), 1.0 / n_eff)
    thresh    = 1.0 - 1.0 / n_eff
    valid_idx = findall(.!((Sp[1:d_len-1] .>= thresh) .& (Sq[1:d_len-1] .>= thresh)))

    gaps_v  = gaps[valid_idx]
    σ̂_v     = σ̂[valid_idx]
    z_v     = gaps_v ./ σ̂_v
    T_obs_v = minimum(z_v)
    κ_n     = sqrt(log(n_eff))
    sel     = let s = findall(z_v .<= κ_n); isempty(s) ? collect(eachindex(z_v)) : s end

    T_boot = Vector{Float64}(undef, B)
    for b in 1:B
        cp_star = _draw_multinomial(prob_ci, n1) ./ n1
        cq_star = _draw_multinomial(prob_ni, n2) ./ n2
        g_star  = majorization_curve(cp_star)[1:d_len-1] .-
                  majorization_curve(cq_star)[1:d_len-1]
        z_star  = (g_star[valid_idx] .- gaps_v) ./ σ̂_v
        T_boot[b] = minimum(z_star[sel])
    end
    p_val = mean(T_boot .<= T_obs_v)
    println("GMS bootstrap p-value (H₀: clean ≻ noisy, N=$N_test): $(round(p_val, digits=4))")

    # Axis covers the T_boot distribution only.
    # T_obs may fall outside when H₀ is clearly true (>> T_boot) or false (<< T_boot).
    span = maximum(T_boot) - minimum(T_boot)
    t_lo = minimum(T_boot) - 0.10 * span
    t_hi = maximum(T_boot) + 0.10 * span

    # Estimate y_max from a coarse density so we can size the rejection overlay
    bw_est  = (t_hi - t_lo) / 40
    cts_est = zeros(Int, 40)
    for x in T_boot
        i = clamp(floor(Int, (x - t_lo) / bw_est) + 1, 1, 40)
        cts_est[i] += 1
    end
    y_max = maximum(cts_est) / (B * bw_est)

    p4b = plot(
        size=(480, 320), dpi=300,
        guidefontsize=11, tickfontsize=9,
        framestyle=:box, grid=:none,
        margin=5Plots.mm, legend=false,
        xlabel="Test statistic  T", ylabel="Density",
        ylims=(0.0, y_max * 1.18),
        xlims=(t_lo, t_hi),
    )

    # Bootstrap distribution as a native histogram (renders reliably)
    histogram!(p4b, T_boot;
               bins=range(-4.0, 1.0, length=61),
               normalize=:pdf,
               fillcolor=:gray80, linewidth=0, label="")

    # Rejection region: semi-transparent purple overlay up to T_obs_v
    # T_shade = clamp(T_obs_v, t_lo, t_hi)
    # if T_shade > t_lo
    #     plot!(p4b, Shape([t_lo, T_shade, T_shade, t_lo],
    #                      [0.0, 0.0, y_max * 1.15, y_max * 1.15]);
    #           fillcolor=col_noise, alpha=0.30, linewidth=0, label="")
    # end

    # T_obs: vline if inside axis, edge annotation otherwise
    if t_lo <= T_obs_v <= t_hi
        vline!(p4b, [T_obs_v]; color=col_noise, linewidth=2.5, label="")
        annotate!(p4b, T_obs_v, y_max * 1.09, text("T_obs", 9, col_noise, :center))
    elseif T_obs_v > t_hi
        annotate!(p4b, t_hi - 0.01*(t_hi - t_lo), y_max * 0.55,
                  text("T_obs →", 9, col_noise, :right))
    else
        annotate!(p4b, t_lo + 0.01*(t_hi - t_lo), y_max * 0.55,
                  text("← T_obs", 9, col_noise, :left))
    end

    x_ann = T_obs_v >= (t_lo + t_hi) / 2 ? t_lo + 0.22*(t_hi-t_lo) : t_hi - 0.22*(t_hi-t_lo)
    annotate!(p4b, x_ann, y_max * 0.78,
              text("p = $(round(p_val, digits=3))", 11, :black, :center))

    savefig(p4b, joinpath(figure_folder, "illustration_pvalue.pdf"))
    println("Saved illustration_pvalue.pdf")
end
