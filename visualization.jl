using Luxor

include("environments/collect_points.jl")
include("tools/play.jl")
include("tools/linear_congruential_generator.jl")

function kernel_test_initialize(environments, env_seed, number_individuals)

    tx = threadIdx().x
    bx = blockIdx().x

    rng_states = @cuDynamicSharedMem(Int64, number_individuals)
    fill!(rng_states, env_seed)
    
    offset = sizeof(rng_states)
    sync_threads()

    initialize(tx, bx, environments, offset, env_seed, rng_states)

    return
end

function kernel_test_step(actions, observations, rewards, environments, env_seed, number_individuals)

    tx = threadIdx().x
    bx = blockIdx().x

    input = @cuDynamicSharedMem(Float32, environments.number_inputs)
    input[1] = actions[1, bx]
    input[2] = actions[2, bx]
    offset = sizeof(input)
    sync_threads()
    
    rng_states = @cuDynamicSharedMem(Int64, number_individuals)
    offset += sizeof(rng_states)
    sync_threads()

    step(tx, bx, input, observations, rewards, offset, environments, rng_states)

    return
end


function main()

    config_environment = OrderedDict()
    config_environment["maze_columns"] = 5
    config_environment["maze_rows"] = 5
    config_environment["maze_cell_size"] = 80
    config_environment["agent_radius"] = 12
    config_environment["point_radius"] = 8
    config_environment["agent_movement_range"] = 10.0
    config_environment["use_sensors"] = true
    config_environment["reward_per_collected_positive_point"] = 500.0
    config_environment["reward_per_collected_negative_point"] = -700.0
    config_environment["number_time_steps"] = 1000
    config_environment["number_sensors"] = 10

    env_seed = rand(1:1000)
    
    number_individuals = 10

    environments = CollectPoints(config_environment, number_individuals)

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs + sizeof(Int64) * number_individuals

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed, number_individuals)

    CUDA.synchronize()

    number_iterations = 1000
    width = environments.maze_rows * environments.maze_cell_size
    height = environments.maze_columns * environments.maze_cell_size
    observations = CUDA.fill(0.0f0, (environments.number_outputs, number_individuals))
    rewards = CUDA.fill(0, number_individuals)

    @play width height number_iterations begin

        actions = CuArray(rand(Uniform(-1, 1),(2, number_individuals)))
        CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(actions, observations, rewards, environments, env_seed, number_individuals)

        CUDA.synchronize()

        render(environments, 1)
    end
end


main()