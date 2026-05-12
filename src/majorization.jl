export majorization_curve, maj_stats, transition_matrix, ds_stats

"""
    majorization_curve(probs)

Sort probabilities descending and return their cumulative sum.
"""
majorization_curve(probs) = cumsum(sort(probs; rev=true))

"""
    maj_stats(p, q) -> (n_pass, n_fail, max_gap, min_gap)

Test whether p majorizes q (p ≻ q).  Returns the number of positions that pass/fail,
the maximum margin (strength of dominance), and the minimum gap (negative = violation).
The last element (total mass ≈ 1) is excluded to avoid float noise.
"""
function maj_stats(p, q)
    cp   = majorization_curve(p)
    cq   = majorization_curve(q)
    n    = something(findlast(x -> x < 1.0 - 1e-12, cp), length(cp))
    gaps = cp[1:n] .- cq[1:n]
    pass = count(>=(0), gaps)
    return pass, n - pass, maximum(gaps), minimum(gaps)
end

"""
    transition_matrix(x_clean, x_noisy, est, n_pat) -> Matrix{Float64}

Build the empirical row-stochastic transition matrix T where
T[i, j] = P(noisy pattern = j | clean pattern = i),
estimated from the aligned ordinal-pattern sequences of `x_clean` and `x_noisy`.
`est` is a ComplexityMeasures ordinal-pattern estimator; `n_pat` is the number of patterns (d!).
"""
function transition_matrix(x_clean, x_noisy, est, n_pat)
    codes_c = codify(est, x_clean)
    codes_n = codify(est, x_noisy)
    C = zeros(Int, n_pat, n_pat)
    for (ci, ni) in zip(codes_c, codes_n)
        C[ci, ni] += 1
    end
    row_s = vec(sum(C, dims=2))
    T = zeros(Float64, n_pat, n_pat)
    for i in 1:n_pat
        row_s[i] > 0 && (T[i, :] = C[i, :] ./ row_s[i])
    end
    return T
end

"""
    ds_stats(T) -> (n_active, max_row_dev, max_col_dev, active, T_sub, row_sums, col_sums)

Compute doubly-stochastic diagnostics for a transition matrix T.
Restricts to the active sub-matrix (patterns present in the clean signal) and returns
the max column-sum deviation from 1 as the primary test statistic.
"""
function ds_stats(T)
    row_s  = vec(sum(T, dims=2))
    active = findall(row_s .> 0)
    T_sub  = T[active, active]
    row_s_sub = vec(sum(T_sub, dims=2))
    col_s_sub = vec(sum(T_sub, dims=1))
    max_row_dev = maximum(abs.(row_s_sub .- 1.0))
    max_col_dev = maximum(abs.(col_s_sub .- 1.0))
    return length(active), max_row_dev, max_col_dev, active, T_sub, row_s_sub, col_s_sub
end
