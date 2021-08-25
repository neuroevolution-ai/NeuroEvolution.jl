using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics

include("optimizers/optimizer.jl")
include("brains/brain.jl")


function kernel_eval_fitness()
end









function main()
    configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json")
    number_generations = configuration["number_generations"]
    number_validation_runs = configuration["number_validation_runs"]
    number_rounds = configuration["number_rounds"]
    maximum_env_seed = configuration["maximum_env_seed"]
    environment = configuration["environment"]
    brain = configuration["brain"]
    optimizer = configuration["optimizer"]

    maze_columns = environment["maze_columns"]
    maze_rows = environment["maze_rows"]
    number_inputs = 6
    number_outputs = 2
    number_neurons = brain["number_neurons"]
    number_individuals = optimizer["population_size"]
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(number_inputs,number_outputs,brain,brain_state)
    optimizer = inititalize_optimizer(free_parameters,optimizer)

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    for generation in 1:number_generations
        env_seed = Random.rand((number_validation_runs:maximum_env_seed), 1)

        individuals = fill(0.0f0,number_individuals,free_parameters) # number_individuals, free_parameters
        genomes = convert(Array{Array{Float32}},ask(optimizer))
        println(generation)


        rewards = rand(Float32,112)
        tell(optimizer,rewards)
    end

end




main()