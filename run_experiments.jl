using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics

include("optimizers/optimizer.jl")


function kernel_eval_fitness()
end









function main()
    configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json")
    println("Konfiguration:",configuration)
    number_generations = configuration["number_generations"]
    println("number_generations:",number_generations)
    number_validation_runs = configuration["number_validation_runs"]
    println("number_validation_runs:",number_validation_runs)
    number_rounds = configuration["number_rounds"]
    println("number_rounds:",number_rounds)
    maximum_env_seed = configuration["maximum_env_seed"]
    println("maximum_env_seed:",maximum_env_seed)
    environment = configuration["environment"]
    println("environment:",environment)
    brain = configuration["brain"]
    println("brain:",brain)
    optimizer = configuration["optimizer"]
    println("optimizer:",optimizer)

    maze_columns = environment["maze_columns"]
    println("maze_columns:",maze_columns)
    maze_rows = environment["maze_rows"]
    println("maze_rows:",maze_rows)
    number_inputs = 6
    println("number_inputs:",number_inputs)
    number_outputs = 2
    println("number_outputs:",number_outputs)
    number_neurons = brain["number_neurons"]
    println("number_neurons:",number_neurons)
    number_individuals = optimizer["population_size"]
    println("number_individuals:",number_individuals)
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    println("brain_state:",brain_state)
    free_parameters = get_individual_size(number_inputs,number_outputs,brain,brain_state)
    println("free_parameters:",free_parameters)
    optimizer = inititalize_optimizer(free_parameters,optimizer)
    println("optimizer:",optimizer)

    best_genome_overall = nothing
    println("best_genome_overall:",best_genome_overall)
    best_reward_overall = typemin(Int32)
    println("best_reward_overall:",best_reward_overall)
end




main()