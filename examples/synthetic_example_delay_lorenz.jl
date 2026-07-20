ENV["GKSwstype"] = "100"

import Pkg
Pkg.activate(@__DIR__)

using Revise
using NoisyCE
using Plots
using LaTeXStrings
using ComplexityMeasures
using DifferentialEquations
using Statistics

default(fontfamily = "Computer Modern")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

const figure_folder = joinpath(@__DIR__, "figures")
mkpath(figure_folder)

const d    = 5          # fixed embedding dimension
const N    = 500_000     # number of Lorenz samples
const δt   = 0.001        # saveat for Lorenz ODE

τ_vals = 1:1:300      # integer delays; physical delay = τ * δt, ranging 0.001 → 2.0

# ------------------------------------------------------------
# Generate Lorenz X-component
# ------------------------------------------------------------

x_lorenz = let
    function lorenz!(du, u, p, _)
        σ_l, ρ, β = p
        du[1] =  σ_l * (u[2] - u[1])
        du[2] =  u[1] * (ρ - u[3]) - u[2]
        du[3] =  u[1] * u[2] - β * u[3]
    end
    sol = solve(ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, (N-1)*δt), (10.0, 28.0, 8/3)),
                Tsit5(); reltol=1e-10, abstol=1e-12, saveat=δt, maxiters=10*N)
    x_raw = Array(sol)[1, :]
    (x_raw .- mean(x_raw)) ./ std(x_raw)
end

println("Lorenz signal: $(length(x_lorenz)) points, δt = $δt")

# ------------------------------------------------------------
# Sweep τ: compute (H, C) and majorization gap for each delay
# ------------------------------------------------------------

_ord_probs = (x, τ) -> vec(first(allprobabilities_and_outcomes(OrdinalPatterns(; m=d, τ=τ), x)))

prob_ref = _ord_probs(x_lorenz, τ_vals[1])   # reference: smallest τ

HC_τ     = [ordinal_entropy_complexity(x_lorenz; d=d, τ=τ) for τ in τ_vals]
H_τ      = [hc[1] for hc in HC_τ]
C_τ      = [hc[2] for hc in HC_τ]

# min_gap: minimum of S_k(p_ref) - S_k(p_τ); negative means majorization is violated
gaps_τ   = [maj_stats(prob_ref, _ord_probs(x_lorenz, τ))[4] for τ in τ_vals]

T_delay = collect(τ_vals) .* δt   # physical time delay for each τ

println("δt·τ=$(round(T_delay[1],   digits=4)):  H=$(round(H_τ[1],   digits=4))  C=$(round(C_τ[1],   digits=4))  gap=$(round(gaps_τ[1],   digits=4))")
println("δt·τ=$(round(T_delay[end], digits=4)):  H=$(round(H_τ[end], digits=4))  C=$(round(C_τ[end], digits=4))  gap=$(round(gaps_τ[end], digits=4))")

# ------------------------------------------------------------
# CH boundary curves (τ-independent in theory; use τ=1)
# ------------------------------------------------------------

H_max_curve, C_max_curve, H_min_curve, C_min_curve = let
    o_ec = OrdinalPatterns(; m=d, τ=1)
    c_ec = StatisticalComplexity(; o=o_ec)
    min_raw, max_raw = entropy_complexity_curves(c_ec; num_max=1, num_min=1000)
    ord_max = sortperm([Float64(pt[1]) for pt in max_raw])
    ord_min = sortperm([Float64(pt[1]) for pt in min_raw])
    ([Float64(pt[1]) for pt in max_raw][ord_max],
     [Float64(pt[2]) for pt in max_raw][ord_max],
     [Float64(pt[1]) for pt in min_raw][ord_min],
     [Float64(pt[2]) for pt in min_raw][ord_min])
end

# ------------------------------------------------------------
# Color scale: min_gap mapped to RdYlGn colormap
# (green = large positive gap → p_ref majorizes; red = negative → violated)
# ------------------------------------------------------------

gap_min_val = minimum(gaps_τ)
gap_max_val = maximum(gaps_τ)
_gnorm = g -> (g - gap_min_val) / (gap_max_val - gap_min_val + 1e-12)
gap_colors  = [cgrad(:RdYlGn)[_gnorm(g)] for g in gaps_τ]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

p = plot(size=(700, 500), dpi=300, legend=:outerright,
         guidefontsize=11, tickfontsize=9, legendfontsize=8,
         framestyle=:box, grid=:none,
         xlims=(0.0, 1.0), ylims=(0.0, 0.5),
         xlabel=L"H", ylabel=L"C",
         title="Lorenz X — delay sweep (d = $d, δt = $δt)")

# CH boundaries
plot!(p, H_max_curve, C_max_curve; color=:black, linewidth=0.9, linestyle=:dash, label="")
plot!(p, H_min_curve, C_min_curve; color=:black, linewidth=0.9, linestyle=:dash, label="")

# Line colored by gap: draw segment-by-segment
n_τ = length(τ_vals)
for i in 1:n_τ-1
    plot!(p, [H_τ[i], H_τ[i+1]], [C_τ[i], C_τ[i+1]];
          color=gap_colors[i], linewidth=2.5, label="")
end

# Markers + labels at selected physical delays
T_marks = [0.01, 0.05, 0.1, 0.15, 0.30]
for T_ann in T_marks
    idx = argmin(abs.(T_delay .- T_ann))
    scatter!(p, [H_τ[idx]], [C_τ[idx]];
             markercolor=gap_colors[idx], markerstrokecolor=:black,
             markerstrokewidth=1.0, markersize=7, label="")
    annotate!(p, H_τ[idx] + 0.018, C_τ[idx],
              text("$(T_ann)", 7, :gray20, :left))
end

# Legend: gap color swatches
for i_lbl in [1, n_τ÷4, n_τ÷2, n_τ]
    g_lbl = round(gaps_τ[i_lbl], digits=4)
    T_lbl = round(T_delay[i_lbl], digits=3)
    plot!(p, [NaN, NaN], [NaN, NaN];
          color=gap_colors[i_lbl], linewidth=2.5, label="gap=$g_lbl (δt·τ=$T_lbl)")
end

savefig(p, joinpath(figure_folder, "delay_lorenz_CH.png"))
println("Saved to $(figure_folder)/delay_lorenz_CH.png")
