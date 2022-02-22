using Plots

include("../tools/linear_congruential_generator.jl")

function main()
    seed = 5

    number_randoms = 10

    lower_bound = 0

    upper_bound = 5

    states = fill(seed, 10)

    randoms = fill(0, number_randoms)
    randoms_compare = fill(0, number_randoms)

    for i = 1:number_randoms
        seed, randoms[i] = lgc_random(seed, lower_bound, upper_bound)
        randoms_compare[i] = lgc_random(states, 1, 0, 5)
    end
    
    gr()
    display(histogram(randoms, bins=(upper_bound-lower_bound + 1), alpha=0.5, label="5"))

    println(randoms == randoms_compare)
end

main()