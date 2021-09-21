using JSON
using Random
using CUDA
using Statistics
using Dates
using DataStructures

include("environments/collect_points.jl")
include("brains/continuous_time_rnn.jl")
include("optimizers/cma_es.jl")
include("tools/episode_runner.jl")
include("tools/write_results.jl")

struct TrainingCfg
    number_generations::Int
    number_validation_runs::Int
    number_rounds::Int
    maximum_env_seed::Int
    environment::OrderedDict
    brain::OrderedDict
    optimizer::OrderedDict
    experiment_id::Int

    function TrainingCfg(configuration::OrderedDict)
        new(
            configuration["number_generations"],
            configuration["number_validation_runs"],
            configuration["number_rounds"],
            configuration["maximum_env_seed"],
            configuration["environment"],
            configuration["brain"],
            configuration["optimizer"],
            get(configuration, "experiment_id", -1),
        )
    end
end


function main()

    configuration_file = "configurations/CMA_ES_Deap_CTRNN_Dense.json"

    # Load configuration file
    configuration = JSON.parsefile(configuration_file, dicttype = OrderedDict)

    config = TrainingCfg(configuration)

    # Get environment type from configuration TODO
    environment_type = CollectPoints

    # Get brain type from configuration TODO
    brain_type = ContinuousTimeRNN

    # Get optimizer type from configuration TODO
    optimizer_type = OptimizerCmaEs

    number_individuals = config.optimizer["population_size"]

    # Initialize environments
    environments = environment_type(config.environment, number_individuals)

    number_inputs = get_number_inputs(environments)
    number_outputs = get_number_outputs(environments)

    # Initialize brains 
    brains = brain_type(config.brain, number_inputs, number_individuals)

    brain_state = generate_brain_state(number_inputs, number_outputs, config.brain)
    free_parameters = get_individual_size(brain_state)

    # Initialize optimizer
    optimizer = optimizer_type(free_parameters, config.optimizer)

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    required_shared_memory =
        sizeof(Int32) +
        get_memory_requirements(number_inputs, number_outputs, brains) +
        get_memory_requirements(environments)

    # Get start time of training and Date
    start_time_training = now()

    log = OrderedDict()

    # Run evolutionary training for given number of generations
    for generation = 1:config.number_generations

        start_time_generation = now()
        env_seed = Random.rand(config.number_validation_runs:config.maximum_env_seed)
        env_seeds_gpu = CUDA.fill(env_seed, number_individuals)
        individuals = ask(optimizer)

        individuals_gpu = CuArray(individuals)
        fitness_results = CUDA.fill(0.0f0, number_individuals)
        @cuda threads = brains.number_neurons blocks = number_individuals shmem =
            required_shared_memory kernel_eval_fitness(
            individuals_gpu,
            fitness_results,
            env_seeds_gpu,
            config.number_rounds,
            brains,
            environments,
        )
        CUDA.synchronize()
        rewards_training = Array(fitness_results)
        tell(optimizer, rewards_training)
        best_genome_current_generation = individuals[(findmax(rewards_training))[2], :]
        rewards_validation = CUDA.fill(0.0f0, config.number_validation_runs)
        validation_individuals = fill(0.0f0, config.number_validation_runs, free_parameters)
        for i = 1:config.number_validation_runs
            for j = 1:free_parameters
                validation_individuals[i, j] = best_genome_current_generation[j]
            end
        end
        env_seeds_validation = 1:config.number_validation_runs

        @cuda threads = brains.number_neurons blocks = config.number_validation_runs shmem =
            required_shared_memory kernel_eval_fitness(
            CuArray(validation_individuals),
            rewards_validation,
            CuArray(env_seeds_validation),
            1,
            brains,
            environments,
        )
        CUDA.synchronize()
        rewards_validation_cpu = Array(rewards_validation)


        best_reward_current_generation = mean(rewards_validation_cpu)
        if best_reward_current_generation > best_reward_overall
            best_genome_overall = best_genome_current_generation
            best_reward_overall = best_reward_current_generation
        end

        elapsed_time_current_generation = now() - start_time_generation
        elapsed_time_current_generation =
            string(Second(floor(elapsed_time_current_generation, Second))) *
            string(elapsed_time_current_generation % 1000)
        log_line = OrderedDict()
        log_line["gen"] = generation
        log_line["min"] = minimum(rewards_training)
        log_line["mean"] = mean(rewards_training)
        log_line["max"] = maximum(rewards_training)
        log_line["best"] = best_reward_overall
        log_line["elapsed_time"] = elapsed_time_current_generation
        log[generation] = log_line
        println(
            "Generation:",
            generation,
            " Min:",
            findmin(rewards_training)[1],
            " Mean:",
            mean(rewards_training),
            " Max:",
            findmax(rewards_training)[1],
            " Best:",
            best_reward_overall,
            " elapsed time (s):",
            elapsed_time_current_generation,
        )
    end

    result_directory =
        "Simulation_results/" *
        Dates.format(floor(start_time_training, Second), DateFormat("yyyy-mm-dd_HH-MM-SS"))

    mkdir(result_directory)
    write_results_to_textfile(
        result_directory * "/Log.txt",
        configuration,
        log,
        number_inputs,
        number_outputs,
        number_individuals,
        free_parameters,
        now() - start_time_training,
    )
end

main()
