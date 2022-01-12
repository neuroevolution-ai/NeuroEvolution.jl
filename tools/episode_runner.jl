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

function kernel_eval_fitness(individuals, rewards, environment_seeds, number_rounds, brains, environments)

    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_shared_memory = 0

    ob = @cuDynamicSharedMem(Float32, environments.number_inputs, offset_shared_memory)
    offset_shared_memory += sizeof(ob)

    sync_threads()

    action = @cuDynamicSharedMem(Float32, environments.number_outputs, offset_shared_memory)
    offset_shared_memory += sizeof(action)

    sync_threads()

    initialize(brains, individuals)
    initialize(environments, ob, environment_seeds[blockID], offset_shared_memory)

    fitness_current = 0
    done = false

    for time_step in 1:environments.number_time_steps

        step(brains, ob, action, offset_shared_memory)
        rew, done = step(environments, action, ob, time_step, offset_shared_memory)
        fitness_current += rew

        if done == true
            break
        end

    end

    if threadID == 1
        rewards[blockID] = fitness_current
    end

    return
end