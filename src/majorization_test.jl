export majorization_test

"""
    majorization_test(p̂, q̂, n1, n2; B=10000, α=0.05) -> NamedTuple

GMS bootstrap test of H₀: p ≻ q (p majorizes q) vs H₁: p ⊁ q.

# Arguments
- `p̂`, `q̂`: empirical probability vectors in Δ^{d-1} (length d, sum ≈ 1)
- `n1`, `n2`: sample sizes used to estimate `p̂` and `q̂`
- `B`: number of bootstrap replicates (default 10000)
- `α`: significance level (default 0.05)

# Returns a NamedTuple with fields:
- `T_obs`: observed test statistic min_k (S_k(p̂) - S_k(q̂)) over valid constraints
- `p_value`: bootstrap p-value
- `reject`: true if H₀ is rejected at level α
- `selected`: global indices k in the GMS-selected near-binding set K̂
- `gaps`: vector of S_k(p̂) - S_k(q̂) for k = 1, …, d-1 (all constraints)
- `σ̂`: delta-method std estimate for each gap (all constraints)
- `thresh`: near-unity threshold used to filter constraints

# Details
The test statistic is T = min_k (S_k(p̂) - S_k(q̂)) over k = 1, …, d-1,
where S_k(p) is the k-th partial sum of p sorted descending (majorization curve).
Rejection for sufficiently negative T.

Constraints where both S_k(p̂) and S_k(q̂) exceed `thresh = 1 - 1/√n_eff` are
excluded before selection and testing: the gap there is a difference of two
near-cancelling numbers and σ̂_k → 0, making the standardised ratio unreliable.

Constraint selection on the remaining valid set follows the Generalized Moment
Selection (GMS) principle: only near-binding constraints (gap / σ̂_k ≤ κ_n) enter
the bootstrap minimum. The GMS threshold is κ_n = √log(n_eff), where
n_eff = n1·n2/(n1+n2) is the harmonic-mean effective sample size.

# References
- Andrews, D. W. K. & Guggenberger, P. (2010). Asymptotic size and a problem with
  subsampling and with the m out of n bootstrap. *Econometric Theory*, 26(2), 426–468.
  https://doi.org/10.1017/S0266466609990423
- Linton, O., Maasoumi, E. & Whang, Y.-J. (2005). Consistent testing for stochastic
  dominance under general sampling schemes. *Review of Economic Studies*, 72(3), 735–765.
  https://doi.org/10.1111/j.1467-937X.2005.00350.x
"""
function majorization_test(
    p̂,
    q̂,
    n1::Int,
    n2::Int;
    B::Int=10000,
    α::Real=0.05,
    c::Real=1.0,
    )

    d = length(p̂)
    @assert length(q̂) == d "p̂ and q̂ must have the same length"
    @assert d >= 2          "need at least d=2"

    Sp   = majorization_curve(p̂)
    Sq   = majorization_curve(q̂)
    gaps = Sp[1:d-1] .- Sq[1:d-1]

    n_eff  = n1 * n2 / (n1 + n2)

    # Delta-method variance: Var[S_k(p̂)] ≈ S_k(p)(1-S_k(p))/n; independent samples add.
    V̂ = Sp[1:d-1] .* (1 .- Sp[1:d-1]) ./ n1 .+ Sq[1:d-1] .* (1 .- Sq[1:d-1]) ./ n2
    # Floor σ̂ at the degenerate-tail scale (~1/n_eff) so the standardised ratio is
    # never inflated by σ̂ → 0; the floor sits well below ordinary mid-k σ̂.
    σ_floor = 1.0 / n_eff
    σ̂ = max.(sqrt.(max.(V̂, 0.0)), σ_floor)

    # Exclude indices to close to 1.0
    thresh = 1.0 - 1.0 / n_eff
    valid_idx = findall(.!((Sp[1:d-1] .>= thresh) .& (Sq[1:d-1] .>= thresh)))

    # All constraints excluded → both curves are nearly uniform; test is uninformative.
    # Return p_value = 1.0 (fail to reject H₀) as a conservative fallback.
    if isempty(valid_idx)
        return (T_obs=0.0, p_value=1.0, reject=false,
                selected=Int[], gaps=gaps, σ̂=σ̂, thresh=thresh)
    end

    gaps_valid = gaps[valid_idx]
    σ̂_valid   = σ̂[valid_idx]
    # Student of the GMS scale
    z_valid    = gaps_valid ./ σ̂_valid
    T_obs  = minimum(z_valid)

    # GMS selection within valid constraints
    sel_local = _gms_select(z_valid, n_eff, c)
    selected  = valid_idx[sel_local]   # global indices (for inspection)

    # Bootstrap: single pass over valid constraints only
    T_boot = Vector{Float64}(undef, B)
    for b in 1:B
        cp_star = _multinomial_draw(p̂, n1) ./ n1
        cq_star = _multinomial_draw(q̂, n2) ./ n2
        g_star  = majorization_curve(cp_star)[1:d-1] .-
                  majorization_curve(cq_star)[1:d-1]
        z_star  = (g_star[valid_idx] .- gaps_valid) ./ σ̂_valid   # divide by the FIXED sample σ̂
        T_boot[b] = minimum(z_star[sel_local])
    end

    p_value = mean(T_boot .<= T_obs)

    return (
        T_obs    = T_obs,
        p_value  = p_value,
        reject   = p_value < α,
        selected = selected,
        gaps     = gaps,
        σ̂        = σ̂,
        thresh   = thresh,
    )
end

function _gms_select(z, n_eff, c)
    # GMS threshold κ_n must satisfy κ_n → ∞ and κ_n/√n → 0 (Andrews & Guggenberger 2010).
    # The original single-sample proposal uses κ_n = √log(n).  In the two-sample setting the
    # variance of each gap estimator is σ₁²/n1 + σ₂²/n2, so the quantity governing sampling
    # uncertainty is the harmonic mean n_eff = n1·n2/(n1+n2).  When n1 >> n2 (e.g. a long
    # model simulation vs a short data record), n_eff ≈ n2: the weaker sample controls
    # inference and κ_n should reflect that.  Using n1+n2 instead inflates κ_n, selecting
    # nearly all constraints and biasing T_boot negative (order-statistics artifact), which
    # destroys test power without any gain in size control.
    κ_n      = c * sqrt(log(n_eff))
    selected = findall(z .<= κ_n)
    isempty(selected) ? collect(eachindex(z)) : selected
end

"""
    _multinomial_draw(p, n) -> Vector{Float64}

Draw a single multinomial(n, p) sample via the sequential conditional-Binomial method.
"""
function _multinomial_draw(p::AbstractVector, n::Int)
    d      = length(p)
    counts = zeros(Float64, d)
    remaining = n
    cum_p     = 0.0
    for i in 1:d-1
        remaining == 0 && break
        p_cond    = clamp(p[i] / (1.0 - cum_p), 0.0, 1.0)
        counts[i] = rand(Binomial(remaining, p_cond))
        remaining -= Int(counts[i])
        cum_p     += p[i]
    end
    counts[d] = remaining
    return counts
end
