# ============================================================
# Generates:
#   - Figure 2 (pvalue_qqplot),
# Author : Facundo Sapienza
# Date   : 2026-07-20
# ============================================================


import Pkg
Pkg.activate(@__DIR__)

using Revise
using NoisyCE
using Distributions
using CairoMakie
using LaTeXStrings
using Random
using Statistics

# ------------------------------------------------------------
# Four distributions:
#   p1: strongly concentrated   (highest curve)
#   p2: moderately concentrated (middle curve)
#   p3: mildly concentrated     (lowest curve, p1 ≻ p2 ≻ p3)
#   p4: crosses all curves      (bimodal shape)
#
# Four tests:
#   Test 1 (boundary):   H₀: p1 ≻ p1  →  p-values ~ Uniform
#   Test 2 (strict H₀):  H₀: p1 ≻ p2  →  conservative (p-values large)
#   Test 3 (reject H₀):  H₀: p1 ≻ p3  →  false, p-values small
#   Test 4 (crossing):   H₀: p1 ≻ p4  →  false (curves cross), p-values small
# ------------------------------------------------------------

const N_sim = 5000   # number of simulation repetitions
const n     = 4000   # sample size for each draw
const d     = 120    # number of bins
const B     = 1000   # bootstrap replicates per test
const seed  = 666
const lw    = 1.5    # line width for majorization curves
const lw_pv = 2.5    # line width for QQ and CDF plots

rng = MersenneTwister(seed)

p1 = let w = exp.(-3.0 .* (0:d-1) ./ d); w ./= sum(w) end    # strongly concentrated   (highest curve)
p2 = let w = exp.(-2.85 .* (0:d-1) ./ d); w ./= sum(w) end    # moderately concentrated (middle curve)
p3 = let w = exp.(-3.15 .* (0:d-1) ./ d); w ./= sum(w) end    # mildly concentrated     (lowest curve)
p4 = let
    # Majorization curve of p4 crosses p1 at rank 0.30:
    # above p1 for rank < 0.30, below p1 for rank > 0.30.
    # Top block (first k bins): steeper exponential than p1, same total mass as p1's top block.
    # Bottom block (remaining bins): uniform, so partial sums grow slowly → below p1.
    p_sorted = sort(p1; rev=true)
    k        = round(Int, 0.30 * d)        # crossing at rank 0.30
    top_mass = sum(p_sorted[1:k])
    bot_mass = 1.0 - top_mass
    top_block = exp.(-1.3 .* (0:k-1) ./ k);  top_block .*= top_mass / sum(top_block)
    bot_block = fill(bot_mass / (d - k), d - k)   # uniform tail
    vcat(top_block, bot_block)
end

# Ordering: p1 ≻ p2 ≻ p3; p4 crosses p1 at rank 0.30
@assert all(NoisyCE.majorization_curve(p1) .>= NoisyCE.majorization_curve(p2) .- 1e-12) "p1 ≻ p2 failed"
@assert any(NoisyCE.majorization_curve(p1) .< NoisyCE.majorization_curve(p4) .- 1e-12)  "p4 should be above p1 before rank 0.30"
println("Distributions confirmed: p1 ≻ p2 ≻ p3, p4 crosses p1 at rank 0.30")

bar_width = 40

function run_test(label, dist_p, dist_q)
    println("\nRunning $N_sim simulations: $label...")
    pvals = Vector{Float64}(undef, N_sim)
    for i in 1:N_sim
        p̂ = rand(rng, Multinomial(n, dist_p)) ./ n
        q̂ = rand(rng, Multinomial(n, dist_q)) ./ n
        pvals[i] = majorization_test(p̂, q̂, n, n; B=B).p_value
        filled = round(Int, bar_width * i / N_sim)
        print("\r[$("█"^filled)$("░"^(bar_width-filled))] $i/$N_sim")
    end
    println()
    println("  Mean p-value:     $(round(mean(pvals); digits=3))")
    println("  Rejection @ 0.05: $(round(mean(pvals .< 0.05); digits=3))")
    return pvals
end

pvals_boundary = run_test("H₀: p1 ≻ p1 (boundary)", p1, p1)
pvals_strict   = run_test("H₀: p1 ≻ p2 (strict)",   p1, p2)
pvals_power    = run_test("H₀: p1 ≻ p3 (false)",    p1, p3)
pvals_cross    = run_test("H₀: p1 ≻ p4 (crossing)",  p1, p4)

# ------------------------------------------------------------
# Plots (CairoMakie — native row-spanning for spine alignment)
# ------------------------------------------------------------

c1, c2, c3, c4 = :steelblue, :crimson, :seagreen, :darkorange

ranks = collect(0:d) ./ d
S1 = [0.0; NoisyCE.majorization_curve(p1)]
S2 = [0.0; NoisyCE.majorization_curve(p2)]
S3 = [0.0; NoisyCE.majorization_curve(p3)]
S4 = [0.0; NoisyCE.majorization_curve(p4)]

uniform_quantiles = (1:N_sim) ./ (N_sim + 1)

const zoom_x = (0.25, 0.35)
const n_resample = 10

# Shared y-range for zoom panels
let
    global zoom_yl
    idx = findall(x -> zoom_x[1] <= x <= zoom_x[2], ranks)
    all_y = vcat(S1[idx], S2[idx], S3[idx], S4[idx])
    buf = 0.03 * (maximum(all_y) - minimum(all_y) + 1e-6)
    zoom_yl = (max(0.0, minimum(all_y) - buf), min(1.0, maximum(all_y) + buf))
end

fig = Figure(size=(1500, 500))

# ---- Left panel: majorization curves (spans both rows) ----
ax_maj = Axis(fig[1:2, 1];
    xlabel="Pattern rank", ylabel="Accumulated prob.",
    title="Majorization curves",
    limits=((0, 1), (0, 1)),
    aspect=AxisAspect(1))

lines!(ax_maj, ranks, S1; color=c1, linewidth=lw, label=L"p_1")
lines!(ax_maj, ranks, S2; color=c2, linewidth=lw, label=L"p_2")
lines!(ax_maj, ranks, S3; color=c3, linewidth=lw, label=L"p_3")
lines!(ax_maj, ranks, S4; color=c4, linewidth=lw, label=L"p_4")
axislegend(ax_maj; position=:rb, title="Curves")

# ---- Right panel: QQ plot (spans both rows) ----
ax_qq = Axis(fig[1:2, 4];
    xlabel="Uniform(0,1) quantiles", ylabel="Empirical p-value quantiles",
    title="QQ plot",
    limits=((0, 1), (0, 1)),
    aspect=AxisAspect(1))

# α=0.05 shaded region drawn first so it sits under all curves
band!(ax_qq, [0, 1], [0, 0], [0.05, 0.05]; color=(:gray, 0.2))
lines!(ax_qq, [0, 1], [0.05, 0.05]; color=:gray, linewidth=1.0, label=L"\alpha=0.05")
lines!(ax_qq, [0, 1], [0, 1];                                      color=:black, linewidth=lw_pv, linestyle=:dash, label="Uniform")
lines!(ax_qq, uniform_quantiles, sort(pvals_boundary); color=c1,   linewidth=lw_pv, label=L"p_1 \succ p_1")
lines!(ax_qq, uniform_quantiles, sort(pvals_strict);   color=c2,   linewidth=lw_pv, label=L"p_1 \succ p_2")
lines!(ax_qq, uniform_quantiles, sort(pvals_power);    color=c3,   linewidth=lw_pv, label=L"p_1 \nsucc p_3")
lines!(ax_qq, uniform_quantiles, sort(pvals_cross);    color=c4,   linewidth=lw_pv, label=L"p_1 \nsucc p_4")
axislegend(ax_qq; position=:lt)

# ---- Middle 2×2 zoom panels ----
function make_zoom_axis!(fig, row, col, pA, pB, SA, SB, colA, colB, lA, lB, title_str)
    ax = Axis(fig[row, col];
        title=title_str,
        limits=(zoom_x, zoom_yl),
        xticks=([0.25, 0.30, 0.35], ["0.25", "0.30", "0.35"]),
        titlesize=11,
        aspect=AxisAspect(1))

    for p_true in (pA, pB)
        col_line = p_true === pA ? colA : colB
        for _ in 1:n_resample
            p̂ = rand(rng, Multinomial(n, p_true)) ./ n
            S = [0.0; NoisyCE.majorization_curve(p̂)]
            lines!(ax, ranks, S; color=col_line, linewidth=1.2, alpha=0.25)
        end
    end
    lines!(ax, ranks, SA; color=colA, linewidth=2.5, label=lA)
    lines!(ax, ranks, SB; color=colB, linewidth=2.5, label=lB)
    axislegend(ax; labelsize=8, padding=(4, 4, 4, 4), position=:lt)
    return ax
end

make_zoom_axis!(fig, 1, 2, p1, p1, S1, S1, c1, c1, L"p_1", L"p_1", L"p_1 \succ p_1")
make_zoom_axis!(fig, 1, 3, p1, p2, S1, S2, c1, c2, L"p_1", L"p_2", L"p_1 \succ p_2")
make_zoom_axis!(fig, 2, 2, p1, p3, S1, S3, c1, c3, L"p_1", L"p_3", L"p_1 \nsucc p_3")
make_zoom_axis!(fig, 2, 3, p1, p4, S1, S4, c1, c4, L"p_1", L"p_4", L"p_1 \nsucc p_4")

# Panel labels a)–f) in top-left of each panel
for (label, pos) in zip(["a)", "b)", "c)", "d)", "e)", "f)"],
                         [(1:2, 1), (1, 2), (1, 3), (2, 2), (2, 3), (1:2, 4)])
    Label(fig[pos..., TopLeft()], label; fontsize=14, font=:bold, halign=:right, padding=(0, 3, 3, 0))
end

# Equal column widths for outer panels; middle columns take the rest
colsize!(fig.layout, 1, Relative(0.333))
colsize!(fig.layout, 4, Relative(0.333))

# Remove row gap between the two zoom rows
rowgap!(fig.layout, 1, 0)

save(joinpath(@__DIR__, "figures", "pvalue_qqplot.png"), fig; px_per_unit=4)
save(joinpath(@__DIR__, "figures", "pvalue_qqplot.pdf"), fig)
println("\nSaved figures/pvalue_qqplot.png and .pdf")
