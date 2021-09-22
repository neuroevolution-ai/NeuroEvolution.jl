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
    fitness_total = 0

    V = @cuDynamicSharedMem(
        Float32,
        (brains.number_neurons, environments.number_inputs)
    )
    W = @cuDynamicSharedMem(
        Float32,
        (brains.number_neurons, brains.number_neurons),
        sizeof(V)
    )
    T = @cuDynamicSharedMem(
        Float32,
        (environments.number_outputs, brains.number_neurons),
        sizeof(V) + sizeof(W)
    )

    input = @cuDynamicSharedMem(
        Float32,
        environments.number_inputs,
        sizeof(V) + sizeof(W) + sizeof(T)
    )

    sync_threads()

    brain_initialize(tx, blockIdx().x, V, W, T, individuals,brains)

    sync_threads()

    x = @cuDynamicSharedMem(
        Float32,
        brains.number_neurons,
        sizeof(V) + sizeof(W) + sizeof(T) + sizeof(input)
    )


    temp_V = @cuDynamicSharedMem(
        Float32,
        brains.number_neurons,
        sizeof(V) + sizeof(W) + sizeof(T) + sizeof(input) + sizeof(x)
    )
    action = @cuDynamicSharedMem(
        Float32,
        environments.number_outputs,
        sizeof(V) + sizeof(W) + sizeof(T) + sizeof(input) + sizeof(temp_V) + sizeof(x)
    )

    maze = @cuDynamicSharedMem(
        Int32,
        (environments.maze_columns, environments.maze_rows, 4),
        sizeof(V) +
        sizeof(W) +
        sizeof(T) +
        sizeof(input) +
        sizeof(temp_V) +
        sizeof(x) +
        sizeof(action)
    )
    environment_config_array = @cuDynamicSharedMem(
        Int32,
        6,
        sizeof(V) +
        sizeof(W) +
        sizeof(T) +
        sizeof(input) +
        sizeof(temp_V) +
        sizeof(x) +
        sizeof(action) +
        sizeof(maze)
    )
    offset =
        sizeof(V) +
        sizeof(W) +
        sizeof(T) +
        sizeof(input) +
        sizeof(temp_V) +
        sizeof(x) +
        sizeof(action) +
        sizeof(maze) +
        sizeof(environment_config_array)

    sync_threads()
    for j = 1:number_rounds
        if threadIdx().x == 1
            @inbounds Random.seed!(Random.default_rng(), environment_seeds[blockIdx().x] + j)
        end
        sync_threads()
        #reset brain
        @inbounds x[tx] = 0.0f0
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
            brain_step(tx,blockIdx().x,temp_V, V, W, T, x, input, action, brains)
            sync_threads()
            if tx == 1
                rew =
                    env_step(maze, action, input, environment_config_array, environments)

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