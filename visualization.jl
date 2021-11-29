using Luxor

include("environments/collect_points.jl")
include("tools/play.jl")

function kernel_test_initialize(environments, env_seed)

    tx = threadIdx().x
    bx = blockIdx().x

    input = @cuDynamicSharedMem(Float32, environments.number_inputs)
    offset = sizeof(input)

    sync_threads()

    initialize(tx, bx, input, environments, offset, env_seed)

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
    number_individuals = 100

    environments = CollectPoints(config_environment, number_individuals)

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed)

    CUDA.synchronize()

    println("AGENT POSITION: ", environments.agents_positions[:, 1])
    println("POSITIVE POINTS: ", environments.positive_points_positions[:, 1])
    println("NEGATIVE POINTS: ", environments.negative_points_positions[:, 1])


    width = environments.maze_cell_size * environments.maze_columns
    height = environments.maze_cell_size * environments.maze_rows
    number_iterations = 1000

    @play width height number_iterations begin
        step(1, environments)
        render(environments)
    end

end


main()