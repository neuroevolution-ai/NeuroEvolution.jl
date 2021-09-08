using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics
using Logging
using Dates
using DataStructures

include("environments/collect_points_env.jl")
include("brains/continuous_time_rnn.jl")
include("optimizers/optimizer.jl")
include("tools/episode_runner.jl")
include("tools/write_results.jl")

function main()
    configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json",dicttype=DataStructures.OrderedDict)

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
    start_time_training = now()


    form = DateFormat("yyyy-mm-dd_HH-MM-SS")
    a = floor(start_time_training,Second)

    log = OrderedDict()
 
    for generation in 1:number_generations
        start_time_generation = now()
        env_seed = Random.rand(number_validation_runs:maximum_env_seed)
        env_seeds_gpu = CUDA.fill(env_seed,number_individuals)
        individuals = fill(0.0f0,number_individuals,free_parameters)
        genomes = ask(optimizer)

        #the optimizer generates the genomes as an Array of Arrays, which the GPU cannot work with, so the genomes need to be reshaped into a MxN matrix.
        for i in 1:number_individuals
            for j in 1:free_parameters
                individuals[i,j] = (genomes[i])[j]
            end
        end
        individuals_gpu = CuArray(individuals) 
        fitness_results = CUDA.fill(0.0f0,number_individuals)
        #@cuda threads=brain_cfg.number_neurons blocks=number_individuals shmem=required_shared_memory kernel_eval_fitness(individuals_gpu,fitness_results,env_seeds_gpu,number_rounds,brain_cfg,environment_cfg)
        CUDA.synchronize()
        rewards_training = Array(fitness_results)
        #tell(optimizer,rewards_training)
        best_genome_current_generation = genomes[(findmax(rewards_training))[2]]
        rewards_validation = CUDA.fill(0.0f0,number_validation_runs)
        validation_individuals = fill(0.0f0,number_validation_runs,free_parameters)
        for i in 1:number_validation_runs
            for j in 1:free_parameters
                validation_individuals[i,j] = best_genome_current_generation[j]
            end
        end
        env_seeds_validation = 1:number_validation_runs

        #@cuda threads=brain_cfg.number_neurons blocks=number_validation_runs shmem=required_shared_memory kernel_eval_fitness(CuArray(validation_individuals),rewards_validation,CuArray(env_seeds_validation),1,brain_cfg,environment_cfg)
        CUDA.synchronize()
        rewards_validation_cpu = Array(rewards_validation)

        
        best_reward_current_generation = mean(rewards_validation_cpu)
        if best_reward_current_generation > best_reward_overall
            best_genome_overall = best_genome_current_generation
            best_reward_overall = best_reward_current_generation
        end
        
        elapsed_time_current_generation = now() - start_time_generation
        elapsed_time_current_generation = string(Second(floor(elapsed_time_current_generation,Second))) * string(elapsed_time_current_generation % 1000)
        log_line = OrderedDict()
        log_line["gen"] = generation
        log_line["min"] = minimum(rewards_training)
        log_line["mean"] = mean(rewards_training)
        log_line["max"] = maximum(rewards_training)
        log_line["best"] = best_reward_overall
        log_line["elapsed_time"] = elapsed_time_current_generation
        log[generation] = log_line
        println("Generation:",generation," Min:",findmin(rewards_training)[1]," Mean:",mean(rewards_training)," Max:",findmax(rewards_training)[1]," Best:",best_reward_overall," elapsed time (s):",elapsed_time_current_generation)
    end
    result_directory = "Simulation_results/"*Dates.format(a,form)

    mkdir(result_directory)
    write_results_to_textfile(result_directory*"/Log.txt",configuration,log,number_inputs,number_outputs,number_individuals,free_parameters,now()-start_time_training)
end

main()
