using NoisyCE
using Test
using Random

include("test_majorization.jl")

@testset verbose=true "NoisyCE.jl" begin

    @testset verbose=true "Majorization Test" begin
        Random.seed!(42)
        test_majorization_no_reject()
        test_majorization_reject()
        test_majorization_equal()
        test_majorization_output_sanity()
    end

end
