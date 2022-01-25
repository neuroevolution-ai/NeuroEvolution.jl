using Test
using Random


include("../environments/collect_points.jl")
include("environments/create_maze.jl")


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

function kernel_test_step(actions, action, observations, environments, env_seed, number_individuals)

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

    step(tx, bx, input, observations, offset, environments, rng_states)

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

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs + sizeof(Int64) * number_individuals

    #---------------------------------------------------------------------------------------------------------------
    #Maze initialization tests
    #---------------------------------------------------------------------------------------------------------------

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed, number_individuals)

    CUDA.synchronize()

    maze_walls_north_cpu = Array(environments.maze_walls_north)
    maze_walls_south_cpu = Array(environments.maze_walls_south)
    maze_walls_east_cpu = Array(environments.maze_walls_east)
    maze_walls_west_cpu = Array(environments.maze_walls_west)

    agent_positions_cpu = Array(environments.agents_positions)
    positive_points_position_cpu = Array(environments.positive_points_positions)
    negativ_points_position_cpu = Array(environments.negative_points_positions)

    #Comparing GPU- to CPU maze initialization
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

    #Tests wether environment is initialized identical for each individual
    for i = 2:number_individuals
        @test maze_walls_north_cpu[:,:,i] == maze_walls_north_cpu[:,:,1]
        @test maze_walls_south_cpu[:,:,i] == maze_walls_south_cpu[:,:,1]
        @test maze_walls_east_cpu[:,:,i] == maze_walls_east_cpu[:,:,1]
        @test maze_walls_west_cpu[:,:,i] == maze_walls_west_cpu[:,:,1]

        @test agent_positions_cpu[:, i] == agent_positions_cpu[:, 1]
        @test positive_points_position_cpu[:, i] == positive_points_position_cpu[:, 1]
        @test negativ_points_position_cpu[:, i] == negativ_points_position_cpu[:, 1]
    end

    #Further maze initialization for seed testing
    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed, number_individuals)

    CUDA.synchronize()

    #Test wether environment is initialized identical for same seed
    @test Array(environments.maze_walls_north) == maze_walls_north_cpu
    @test Array(environments.maze_walls_south) == maze_walls_south_cpu
    @test Array(environments.maze_walls_west) == maze_walls_west_cpu
    @test Array(environments.maze_walls_east) == maze_walls_east_cpu

    env_seed_second = rand(1:1000)

    while(env_seed == env_seed_second)
        env_seed_second = rand(1:1000)
    end

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed_second, number_individuals)

    CUDA.synchronize()

    #Test wether environment is initialized different for new seed
    @test Array(environments.maze_walls_north) != maze_walls_north_cpu
    @test Array(environments.maze_walls_south) != maze_walls_south_cpu
    @test Array(environments.maze_walls_west) != maze_walls_west_cpu
    @test Array(environments.maze_walls_east) != maze_walls_east_cpu

    #---------------------------------------------------------------------------------------------------------------
    #Step function tests
    #---------------------------------------------------------------------------------------------------------------

    observations = CUDA.fill(0.0f0, (10, number_individuals))


    mr = environments.agent_movement_range

    environments.agents_positions .= environments.maze_cell_size / 2

    original_agents_positions = Array(environments.agents_positions)

    action = CUDA.fill(0.0f0, 2)
    actions = rand(Uniform(-1, 1),(2, number_individuals))

    new_agents_position = Array(environments.agents_positions) 

    for i = 1:number_individuals
        new_agents_position[1, i] += convert(Int, round(clamp(actions[1, i] * mr, -mr, mr)))
        new_agents_position[2, i] += convert(Int, round(clamp(actions[2, i] * mr, -mr, mr)))
    end  

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), action, observations, environments, env_seed, number_individuals)

    CUDA.synchronize()

    @test Array(environments.agents_positions) == new_agents_position
    



    
end
