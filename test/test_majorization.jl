function test_majorization_no_reject()
    # p majorizes q → H₀ should NOT be rejected
    p = [0.5, 0.3, 0.2]
    q = [0.4, 0.35, 0.25]
    r = majorization_test(p, q, 500, 500; B=999)
    @test r.T_obs >= 0
    @test !r.reject
    @test length(r.gaps) == length(p) - 1
    @test length(r.selected) >= 1
end

function test_majorization_reject()
    # q ≻ p, so p ⊁ q → H₀ should be rejected
    p = [0.5, 0.3, 0.2]
    q = [0.4, 0.35, 0.25]
    r = majorization_test(q, p, 500, 500; B=999)
    @test r.T_obs < 0
    @test r.reject
end

function test_majorization_equal()
    # Uniform vs uniform: p = q, conservative test should not reject
    u = [1/3, 1/3, 1/3]
    r = majorization_test(u, u, 200, 200; B=999)
    @test !r.reject
end

function test_majorization_output_sanity()
    p = [0.5, 0.3, 0.2]
    q = [0.4, 0.35, 0.25]
    r = majorization_test(p, q, 500, 500; B=999)
    @test r.p_value >= 0 && r.p_value <= 1
    @test all(1 .<= r.selected .<= length(p) - 1)
end
