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

    # Use same individual for each validation run 
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
