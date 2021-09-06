using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics
using Logging
using Dates

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
    brain_cfg = CTRNN_Cfg(brain["delta_t"],brain["number_neurons"],brain["clipping_range_min"],brain["clipping_range_max"],brain["alpha"])
    number_inputs = get_number_inputs()
    number_outputs = get_number_outputs()
    environment_cfg = Collect_Points_Env_Cfg(environment["maze_columns"],environment["maze_rows"],environment["maze_cell_size"],environment["agent_radius"],environment["point_radius"],environment["agent_movement_range"],environment["reward_per_collected_positive_point"],environment["reward_per_collected_negative_point"],environment["number_time_steps"],number_inputs,number_outputs)
    number_individuals = optimizer["population_size"]
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(brain_state)

    optimizer = inititalize_optimizer(free_parameters,optimizer)

    #get start time of training and Date

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)
    required_shared_memory = get_memory_requirements(number_inputs,number_outputs, brain_cfg) + get_memory_requirements(environment_cfg)
    for generation in 1:number_generations

        env_seed = Random.rand(number_validation_runs:maximum_env_seed)
        individuals = fill(0.0f0,number_individuals,free_parameters)
        genomes = convert(Array{Array{Float32}},ask(optimizer))

        #the optimizer generates the genomes as an Array of Arrays, which the GPU cannot work with, so the genomes need to be reshaped into a MxN matrix.
        for i in 1:number_individuals
            for j in 1:free_parameters
                individuals[i,j] = (genomes[i])[j]
            end
        end

        individuals_gpu = CuArray(individuals) 
        fitness_results = CUDA.fill(0f0,number_individuals)
        @cuda threads=brain_cfg.number_neurons blocks=number_individuals shmem=required_shared_memory kernel_eval_fitness(individuals_gpu,fitness_results,env_seed,number_rounds,brain_cfg,environment_cfg)
        CUDA.synchronize()
        rewards_training = Array(fitness_results)
        display(rewards_training)
        tell(optimizer,rewards_training)
        best_genome_current_generation = genomes[(findmax(rewards_training))[2]]


        env_seeds = Array(1:number_validation_runs)
        rewards_validation = CUDA.fill(0f0,number_validation_runs)
        validation_individuals = fill(0.0f0,number_validation_runs,free_parameters)
        for i in 1:number_validation_runs
            for j in 1:free_parameters
                validation_individuals[i,j] = best_genome_current_generation[j]
            end
        end
        @cuda threads=brain_cfg.number_neurons blocks=number_validation_runs shmem=required_shared_memory kernel_eval_validation(CuArray(validation_individuals),rewards_validation,brain_cfg,environment_cfg)
        CUDA.synchronize()
        rewards_validation_cpu = Array(rewards_validation)

        
        best_reward_current_generation = mean(rewards_validation)
        if best_reward_current_generation > best_reward_overall
            best_genome_overall = best_genome_current_generation
            best_reward_overall = best_reward_current_generation
        end
        
        #println("Generation:",generation," result:",rewards_training[1])
        #println("Generation:",generation," best_reward_current_generation:",best_reward_current_generation," Highest Reward overall:",best_reward_overall)
    end

end

main()
