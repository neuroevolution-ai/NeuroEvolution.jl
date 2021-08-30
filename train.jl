using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics

include("environments/collect_points_env.jl")
include("brains/continuous_time_rnn.jl")
include("optimizers/optimizer.jl")
include("tools/episode_runner.jl")

function main()
    configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json")

    number_generations = configuration["number_generations"]
    number_validation_runs = configuration["number_validation_runs"]
    number_rounds = configuration["number_rounds"]
    maximum_env_seed = configuration["maximum_env_seed"]
    environment = configuration["environment"]
    brain = configuration["brain"]
    optimizer = configuration["optimizer"]

    maze_columns = 5
    maze_rows = 5
    number_inputs = 10
    number_outputs = 2
    number_neurons = brain["number_neurons"]
    number_individuals = optimizer["population_size"]
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(number_inputs,number_outputs,brain,brain_state)

    optimizer = inititalize_optimizer(free_parameters,optimizer)

    #get start time of training and Date

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    for generation in 1:number_generations
        env_seed = Random.rand((number_validation_runs:maximum_env_seed), 1)

        individuals = fill(0.0f0,number_individuals,free_parameters)
        genomes = convert(Array{Array{Float32}},ask(optimizer))
        for i in 1:number_individuals
            for j in 1:free_parameters
                individuals[i,j] = (genomes[i])[j]
            end
        end

        input = CUDA.fill(1.0f0,10)
        rewards_training = CUDA.fill(0f0,number_individuals)
        individuals_gpu = CuArray(individuals) 
        action = CUDA.fill(1.0f0,2)
        fitness_results = CUDA.fill(0f0,112)
        rounds = CUDA.fill(number_rounds,1)
        println("start Generation:",generation)
        @cuda threads=number_neurons blocks=number_individuals shmem=sizeof(Float32)*(number_neurons*(number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 10) kernel_eval_fitness(individuals_gpu,fitness_results,CuArray(env_seed),rounds)
        CUDA.synchronize()
        println("finished Generation:",generation)
        rewards_training = Array(fitness_results)

        tell(optimizer,rewards_training)

        best_genome_current_generation = genomes[(findmax(rewards_training))[2]]


        env_seeds = Array(1:number_validation_runs)
        number_validation_rounds = 1
        rewards_validation = CUDA.fill(0f0,number_validation_runs)
        validation_individuals = fill(0.0f0,number_validation_runs,free_parameters)
        for i in 1:number_validation_runs
            for j in 1:free_parameters
                validation_individuals[i,j] = best_genome_current_generation[j]
            end
        end
        rounds = CUDA.fill(1,1)
        println("started Validation Generation:",generation)
        @cuda threads=number_neurons blocks=number_validation_runs shmem=sizeof(Float32)*(number_neurons*(number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 16) kernel_eval_fitness(CuArray(validation_individuals),rewards_validation, CuArray(env_seeds), rounds)
        CUDA.synchronize()
        println("finished Validation Generation:",generation)
        rewards_validation_cpu = Array(rewards_validation)
        display(mean(rewards_validation_cpu))

        
        best_reward_current_generation = mean(rewards_validation)
        if best_reward_current_generation > best_reward_overall
            best_genome_overall = best_genome_current_generation
            best_reward_overall = best_reward_current_generation
        end
        
        
    end
    display(best_reward_overall)
end

main()
