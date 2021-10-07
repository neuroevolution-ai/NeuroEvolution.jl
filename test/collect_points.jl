using Test
using Random


include("../environments/collect_points.jl")
include("environments/create_maze.jl")


function kernel_test_initialize(environments, env_seed)

    tx = threadIdx().x
    bx = blockIdx().x

    input = @cuDynamicSharedMem(Float32, environments.number_inputs)
    offset = sizeof(input)

    sync_threads()

    initialize(tx, bx, input, environments, offset, env_seed)

    return
end


@testset "Collect Points" begin

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

    maze_walls_west_cpu = Array(environments.maze_walls_west)
    maze_walls_east_cpu = Array(environments.maze_walls_east)
    maze_walls_south_cpu = Array(environments.maze_walls_south)
    maze_walls_north_cpu = Array(environments.maze_walls_north)

    for i = 1:number_individuals

        maze = make_maze(environments.maze_columns, environments.maze_rows, Array(environments.maze_randoms[:, i]))

        for x = 1:environments.maze_columns
            for y = 1:environments.maze_rows

                @test maze[x, y].walls[West] == maze_walls_west_cpu[x, y, i]
                @test maze[x, y].walls[East] == maze_walls_east_cpu[x, y, i]
                @test maze[x, y].walls[South] == maze_walls_south_cpu[x, y, i]
                @test maze[x, y].walls[North] == maze_walls_north_cpu[x, y, i]

            end
        end

    end

end

println("Finished")