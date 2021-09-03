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
    brain_cfg = CTRNN_Cfg(brain["delta_t"],brain["number_neurons"],brain["clipping_range_min"],brain["clipping_range_max"],brain["alpha"])
    environment_cfg = Collect_Points_Env_Cfg(environment["maze_columns"],environment["maze_rows"],environment["maze_cell_size"],environment["agent_radius"],environment["point_radius"],environment["agent_movement_range"],environment["reward_per_collected_positive_point"],environment["reward_per_collected_negative_point"],environment["number_time_steps"])
    maze_columns = 5
    maze_rows = 5
    number_inputs = 10
    number_outputs = 2
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

        individuals_gpu = CuArray(individuals) 
        fitness_results = CUDA.fill(0f0,number_individuals)
        println("start Generation:",generation)
        @cuda threads=brain_cfg.number_neurons blocks=number_individuals shmem=sizeof(Float32)*(brain_cfg.number_neurons*(brain_cfg.number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 10) kernel_eval_fitness(individuals_gpu,fitness_results,CuArray(env_seed),number_rounds,brain_cfg,environment_cfg)
        CUDA.synchronize()
        println("finished Generation:",generation)
        rewards_training = Array(fitness_results)

        tell(optimizer,rewards_training)
        display((findmax(rewards_training))[1])
        best_genome_current_generation = genomes[(findmax(rewards_training))[2]]


        env_seeds = Array(1:number_validation_runs)
        rewards_validation = CUDA.fill(0f0,number_validation_runs)
        validation_individuals = fill(0.0f0,number_validation_runs,free_parameters)
        for i in 1:number_validation_runs
            for j in 1:free_parameters
                validation_individuals[i,j] = best_genome_current_generation[j]
            end
        end
        println("started Validation Generation:",generation)
        @cuda threads=brain_cfg.number_neurons blocks=number_validation_runs shmem=sizeof(Float32)*(brain_cfg.number_neurons*(brain_cfg.number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 16) kernel_eval_fitness(CuArray(validation_individuals),rewards_validation, CuArray(env_seeds), 1,brain_cfg,environment_cfg)
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
