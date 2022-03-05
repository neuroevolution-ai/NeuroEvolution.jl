using Test
using Random


include("../environments/collect_points.jl")
include("environments/create_maze.jl")


function kernel_test_initialize(environments, env_seed, number_individuals)

    tx = threadIdx().x
    bx = blockIdx().x

    rng_states = @cuDynamicSharedMem(Int64, 1)
    rng_states[1] = env_seed    
    offset = sizeof(rng_states)
    sync_threads()

    initialize(tx, bx, environments, offset, rng_states)

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
    
    rng_states = @cuDynamicSharedMem(Int64, 1, offset)
    offset += sizeof(rng_states)
    sync_threads()

    step(tx, bx, input, observations, rewards, offset, environments, rng_states)

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
    config_environment["number_sensors"] = 2

    env_seed = rand(1:1000)
    number_individuals = 100

    environments = CollectPoints(config_environment, number_individuals)

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs + sizeof(Int64) * number_individuals

    #---------------------------------------------------------------------------------------------------------------
    # Maze initialization tests
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
    # Step function tests
    #---------------------------------------------------------------------------------------------------------------

    rewards = CUDA.fill(0, number_individuals)
    observations = CUDA.fill(0.0f0, (environments.number_outputs, number_individuals))


    mr = environments.agent_movement_range

    environments.agents_positions .= environments.maze_cell_size / 2

    original_agents_positions = Array(environments.agents_positions)

    #Testing agent movements
    actions = rand(Uniform(-1, 1),(2, number_individuals))

    new_agents_position = Array(environments.agents_positions) 

    for i = 1:number_individuals
        new_agents_position[1, i] += convert(Int, round(clamp(actions[1, i] * mr, -mr, mr)))
        new_agents_position[2, i] += convert(Int, round(clamp(actions[2, i] * mr, -mr, mr)))
    end  

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, rewards, environments, env_seed, number_individuals)

    CUDA.synchronize()

    @test Array(environments.agents_positions) == new_agents_position

    #Tests wether points are collected properly
    rewards = CUDA.fill(0, number_individuals)

    #positive points
    copyto!(environments.agents_positions, original_agents_positions)

    pos_point_positions = original_agents_positions .+= environments.agent_movement_range
    copyto!(environments.positive_points_positions, pos_point_positions)

    fill!(actions, 1)

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, rewards, environments, env_seed, number_individuals)

    @test pos_point_positions != Array(environments.positive_points_positions)

    @test Array(rewards) == fill(environments.reward_per_collected_positive_point, number_individuals)

    #negative points
    copyto!(environments.agents_positions, original_agents_positions)

    neg_point_positions = original_agents_positions .-= environments.agent_movement_range
    copyto!(environments.negative_points_positions, neg_point_positions)

    fill!(actions, -1)

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, rewards, environments, env_seed, number_individuals)

    @test neg_point_positions != Array(environments.negative_points_positions)

    @test Array(rewards) == fill(environments.reward_per_collected_positive_point + environments.reward_per_collected_negative_point, number_individuals)

    #---------------------------------------------------------------------------------------------------------------
    # Sensor tests
    #---------------------------------------------------------------------------------------------------------------

    #Test ray initialization
    cpu_ray_directions = Array(environments.ray_directions)
    cpu_ray_distances = Array(environments.ray_cell_distances)

    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            if cpu_ray_directions[1, sensor_number, individual] == 0.0
                @test isinf(cpu_ray_distances[1, sensor_number, individual])
            else
                x_distance =  sqrt(1 + (cpu_ray_directions[2, sensor_number, individual] / cpu_ray_directions[1, sensor_number, individual]) * (cpu_ray_directions[2, sensor_number, individual] / cpu_ray_directions[1, sensor_number, individual]))
                x_distance *= environments.maze_cell_size
                @test cpu_ray_distances[1, sensor_number, individual] == x_distance
            end

            if cpu_ray_directions[2, sensor_number, individual] == 0.0
                @test isinf(cpu_ray_distances[2, sensor_number, individual])
            else
                y_distance =  sqrt(1 + (cpu_ray_directions[1, sensor_number, individual] / cpu_ray_directions[2, sensor_number, individual]) * (cpu_ray_directions[1, sensor_number, individual] / cpu_ray_directions[2, sensor_number, individual]))
                y_distance *= environments.maze_cell_size
                @test cpu_ray_distances[1, sensor_number, individual] == y_distance
            end
        end
    end
    
    #Sensor informations after agent Step

    actions = rand(Uniform(-1, 1),(2, number_individuals))

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, rewards, environments, env_seed, number_individuals)
    
    normalization = max(environments.maze_cell_size * environments.maze_rows, environments.maze_cell_size * environments.maze_columns)
    new_agents_position = Array(environments.agents_positions)
    new_pos_position = Array(environments.positive_points_positions) 
    new_neg_position = Array(environments.negative_points_positions)
    cpu_observations = Array(observations)

    inital_sensor_distances = Array(environments.sensor_distances)

    for individual = 1:number_individuals
        @test cpu_observations[1, individual] == convert(Float32, new_agents_position[1, individual] / normalization)
        @test cpu_observations[2, individual] == convert(Float32, new_agents_position[2, individual] / normalization)

        @test cpu_observations[3, individual] == convert(Float32, new_pos_position[1, individual] / normalization)
        @test cpu_observations[4, individual] == convert(Float32, new_pos_position[2, individual] / normalization)

        @test cpu_observations[5, individual] == convert(Float32, new_neg_position[1, individual] / normalization)
        @test cpu_observations[6, individual] == convert(Float32, new_neg_position[2, individual] / normalization)

        for sensor_number = 1:environments.number_sensors
            @test cpu_observations[sensor_number + 6, individual] == convert(Float32, inital_sensor_distances[sensor_number, individual] / normalization)
        end    
    end

    #One step in x direction
    actions .= [1, 0]
    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, rewards, environments, env_seed, number_individuals)
    new_sensor_distances = Array(environments.sensor_distances)

    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            if cpu_ray_directions[1, sensor_number, individual] > 0
                @test inital_sensor_distances[sensor_number, individual] > new_sensor_distances[sensor_number, individual]
            elseif cpu_ray_directions[1, sensor_number, individual] < 0
                @test inital_sensor_distances[sensor_number, individual] < new_sensor_distances[sensor_number, individual]
            end    
        end    
    end
    
    #One step in y direction
    actions .= [0, 1]
    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, rewards, environments, env_seed, number_individuals)
    new_sensor_distances = Array(environments.sensor_distances)

    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            if cpu_ray_directions[2, sensor_number, individual] > 0
                @test inital_sensor_distances[sensor_number, individual] > new_sensor_distances[sensor_number, individual]
            elseif cpu_ray_directions[2, sensor_number, individual] < 0
                @test inital_sensor_distances[sensor_number, individual] < new_sensor_distances[sensor_number, individual]
            end    
        end    
    end
end
