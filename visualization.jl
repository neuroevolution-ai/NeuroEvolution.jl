using Luxor

include("environments/collect_points.jl")
include("tools/play.jl")

function kernel_test_initialize(environments, env_seed)

    tx = threadIdx().x
    bx = blockIdx().x

    input = @cuDynamicSharedMem(Float32, environments.number_inputs)
    offset = sizeof(input)

    sync_threads()
    
    initialize(tx, bx, environments, offset, env_seed)

    return
end

function kernel_test_collect_points(action, observations, environments, env_seed)

    tx = threadIdx().x
    bx = blockIdx().x

    input = @cuDynamicSharedMem(Float32, environments.number_inputs)
    offset = sizeof(input)
    sync_threads()
    step(tx, bx, action, observations, offset, environments)

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

    env_seed = rand(1:1000)

    environments = CollectPoints(config_environment, number_individuals)

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed)

    CUDA.synchronize()

    number_iterations = 1000
    width = environments.maze_rows * environments.maze_cell_size
    height = environments.maze_columns * environments.maze_cell_size

    observations = CUDA.fill(0.0f0, (10, number_individuals))

    @play width height number_iterations begin

        action = CuArray(rand(Uniform(-1.0, 1.0), 2))
        CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_collect_points(action, observations, environments, env_seed)

        CUDA.synchronize()

        render(environments, 1)
    end
end


main()