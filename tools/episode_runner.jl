using CUDA
using Random

function training_runs(individuals, number_individuals, brains, environments; env_seed::Int, number_rounds::Int, threads::Int, blocks::Int, shared_memory::Int)
    
    rewards = CUDA.fill(0.0f0, number_individuals)
    individuals_gpu = CUDA.CuArray(individuals)
    environment_seeds = CUDA.fill(env_seed, number_individuals)
    CUDA.@cuda threads = threads blocks = blocks shmem = shared_memory kernel_eval_fitness(
        individuals_gpu ,
        rewards,
        environment_seeds,
        number_rounds,
        brains,
        environments
    )

    CUDA.synchronize()
    
    return Array(rewards)

end

function validation_runs(individual, individual_size, number_validation_runs, brains, environments; threads::Int, blocks::Int, shared_memory::Int)
    
    rewards = CUDA.fill(0.0f0, number_validation_runs)
    
    individuals = fill(0.0f0, number_validation_runs, individual_size)

    for i = 1:number_validation_runs
        for j = 1:individual_size
            individuals[i, j] = individual[j]
        end
    end
    individuals_gpu = CuArray(individuals)
    environment_seeds = CuArray(1:number_validation_runs)
    number_rounds = 1
    CUDA.@cuda threads = threads blocks = blocks shmem = shared_memory kernel_eval_fitness(
        individuals_gpu,
        rewards,
        environment_seeds,
        number_rounds,
        brains,
        environments
    )

    CUDA.synchronize()

    return Array(rewards)

end

function kernel_eval_fitness(individuals, rewards, environment_seeds, number_rounds, brains::ContinuousTimeRNN, environments::CollectPoints)

    tx = threadIdx().x
    bx = blockIdx().x

    fitness_total = 0

    initialize(tx, bx, individuals, brains)
    offset = get_memory_requirements(brains)

    sync_threads()

    input = @cuDynamicSharedMem(Float32, environments.number_inputs, offset)
    offset += sizeof(input)

    sync_threads()

    action = @cuDynamicSharedMem(Float32, environments.number_outputs, offset)
    offset += sizeof(action)

    maze = @cuDynamicSharedMem(Int32, (environments.maze_columns, environments.maze_rows, 4), offset)
    offset += sizeof(maze)
    
    environment_config_array = @cuDynamicSharedMem(Int32, 6, offset)
    offset += sizeof(environment_config_array)

    sync_threads()

    for j = 1:number_rounds

        if threadIdx().x == 1
            @inbounds Random.seed!(Random.default_rng(), environment_seeds[blockIdx().x] + j)
        end
        
        sync_threads()
        
        reset(tx, bx, brains)
        fitness_current = 0

        if tx == 1
            create_maze(maze, environments, offset)
        end

        #Place agent, positive_point, negative_point randomly in maze
        if tx == 1
            @inbounds environment_config_array[1], environment_config_array[2] =
                place_agent_randomly_in_maze(environments)
        end
        if tx == 2
            @inbounds environment_config_array[3], environment_config_array[4] =
                place_agent_randomly_in_maze(environments)
        end
        if tx == 3
            @inbounds environment_config_array[5], environment_config_array[6] =
                place_agent_randomly_in_maze(environments)
        end
        sync_threads()

        if tx == 1
            get_observation(maze, input, environment_config_array, environments)
        end

        sync_threads()


        #Loop through Timesteps
        #################################################
        for index = 1:environments.number_time_steps
            step(tx, bx, input, action, brains)

            sync_threads()
            if tx == 1
                rew = env_step(maze, action, input, environment_config_array, environments)
            
                fitness_current += rew
            end
            sync_threads()
        end

        ####################################################
        #end of Timestep
        if tx == 1
            fitness_total += fitness_current
        end
        sync_threads()

    end
    ######################################################
    #end of Round
    if tx == 1
        @inbounds rewards[blockIdx().x] = fitness_total / number_rounds
    end
    sync_threads()
    return
end