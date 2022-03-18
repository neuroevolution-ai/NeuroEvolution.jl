using Plots
using Test

include("../tools/linear_congruential_generator.jl")

function main()
    seed = 6

    number_randoms = 10000

    lower_bound = 0

    upper_bound = 20

    states = fill(seed, 20)

    randoms = fill(0, number_randoms)
    randoms_compare = fill(0, number_randoms)

    for i = 1:number_randoms
        seed, randoms[i] = lgc_random(seed, lower_bound, upper_bound)
        randoms_compare[i] = lgc_random(states, 1, lower_bound, upper_bound)
    end
    
    gr()
    display(histogram(randoms, bins=(upper_bound-lower_bound + 1), alpha=0.5))

    @test randoms == randoms_compare
end

main()