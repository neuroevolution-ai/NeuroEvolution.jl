using Test
using Random


include("../environments/collect_points.jl")
include("environments/create_maze.jl")


function kernel_test_initialize(environments, env_seed, test_sensor_distances, test_ray_directions, test_ray_cell_distances)

    tx = threadIdx().x
    bx = blockIdx().x

    rng_states = @cuDynamicSharedMem(Int64, 1)
    rng_states[1] = env_seed    
    offset = sizeof(rng_states)
    sync_threads()

    sensor_distances = @cuDynamicSharedMem(Float32, environments.number_sensors, offset)
    offset += sizeof(sensor_distances)

    sync_threads()

    ray_directions = @cuDynamicSharedMem(Float32, (2, environments.number_sensors), offset)
    offset += sizeof(ray_directions)

    sync_threads()

    ray_cell_distances = @cuDynamicSharedMem(Float32, (2, environments.number_sensors), offset)
    offset += sizeof(ray_cell_distances)

    sync_threads()

    initialize(tx, bx, sensor_distances, ray_directions, ray_cell_distances, environments, offset, rng_states)
    sync_threads()

    #Copy data from shared memory to global memory
    if tx <= environments.number_sensors
        test_sensor_distances[tx, bx] = sensor_distances[tx]

        test_ray_directions[1, tx, bx] = ray_directions[1, tx]
        test_ray_directions[2, tx, bx] = ray_directions[2, tx]

        test_ray_cell_distances[1, tx, bx] = ray_cell_distances[1, tx]
        test_ray_cell_distances[2, tx, bx] = ray_cell_distances[2, tx]
    end 

    return
end

function kernel_test_step(actions, observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)

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

    observation = @cuDynamicSharedMem(Float32, environments.number_outputs, offset)
    offset += sizeof(observation)
    sync_threads()

    sensor_distances = @cuDynamicSharedMem(Float32, environments.number_sensors, offset)
    offset += sizeof(sensor_distances)

    sync_threads()

    ray_directions = @cuDynamicSharedMem(Float32, (2, environments.number_sensors), offset)
    offset += sizeof(ray_directions)

    sync_threads()

    ray_cell_distances = @cuDynamicSharedMem(Float32, (2, environments.number_sensors), offset)
    offset += sizeof(ray_cell_distances)

    sync_threads()

    angle_diff_per_ray = 360 / environments.number_sensors

    if tx <= environments.number_sensors
        init_ray(tx, 1, 0, angle_diff_per_ray, ray_directions, ray_cell_distances, environments)
    end
    sync_threads()

    if tx <= environments.number_sensors
        calculate_ray_distance(tx, bx, sensor_distances, ray_directions, ray_cell_distances, environments)
    end
    sync_threads()

    step(tx, bx, input, observation, sensor_distances, ray_directions, ray_cell_distances, rewards, environments, rng_states)
    sync_threads()

    #Copy data from shared memory to global memory
    if tx <= environments.number_outputs
        observations[tx, bx] = observation[tx]
    end
    
    if tx <= environments.number_sensors
        test_sensor_distances[tx, bx] = sensor_distances[tx]
        
        test_ray_directions[1, tx, bx] = ray_directions[1, tx]
        test_ray_directions[2, tx, bx] = ray_directions[2, tx]

        test_ray_cell_distances[1, tx, bx] = ray_cell_distances[1, tx]
        test_ray_cell_distances[2, tx, bx] = ray_cell_distances[2, tx]
    end     
    sync_threads()
    if tx == 1
        for i = (environments.number_sensors + 1):environments.number_outputs
            observations[i, bx] = observation[i]
        end    
    end    

    return
end

function DDA_cpu(environments::CollectPoints, sensor_distances,  ray_directions, ray_cell_distances, number_individuals)
    cpu_agent_position = Array(environments.agents_positions)
    cpu_ray_directions = Array(ray_directions)
    cpu_ray_cell_distances = Array(ray_cell_distances)

    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            current_cell_x = convert(Int, ceil(cpu_agent_position[1, individual] / environments.maze_cell_size)) 
            current_cell_y = convert(Int, ceil(cpu_agent_position[2, individual] / environments.maze_cell_size))
    
            cell_left, cell_right, cell_top, cell_bottom = get_coordinates_maze_cell(current_cell_x, current_cell_y, environments.maze_cell_size)
    
            if cpu_ray_directions[1, sensor_number, individual] < 0.0
                step_direction_x = -1
                ray_length_x = (cpu_agent_position[1, individual] - cell_left) / environments.maze_cell_size
                ray_length_x *= cpu_ray_cell_distances[1, sensor_number, individual]
                maze_walls_x = Array(environments.maze_walls_west)
            else
                step_direction_x = 1
                ray_length_x = 1 - ((cpu_agent_position[1, individual] - cell_left) / environments.maze_cell_size)
                ray_length_x *= cpu_ray_cell_distances[1, sensor_number, individual]
                maze_walls_x = Array(environments.maze_walls_east)
            end
            
            if cpu_ray_directions[2, sensor_number, individual] < 0.0
                step_direction_y = 1
                ray_length_y = 1 - ((cpu_agent_position[2, individual] - cell_top) / environments.maze_cell_size)
                ray_length_y *= cpu_ray_cell_distances[2, sensor_number, individual]
                maze_walls_y = Array(environments.maze_walls_south)
            else
                step_direction_y = -1
                ray_length_y = (cpu_agent_position[2, individual] - cell_top) / environments.maze_cell_size
                ray_length_y *= cpu_ray_cell_distances[2, sensor_number, individual]
                maze_walls_y = Array(environments.maze_walls_north)
            end
    
            #DDA
            hit = false
            current_distance = 0.0
            side = 0
    
            while !hit
                if ray_length_x < ray_length_y
                    current_distance = ray_length_x
                    ray_length_x += cpu_ray_cell_distances[1, sensor_number, individual]
                    side = 1
                    maze_walls = maze_walls_x   
                else
                    current_distance = ray_length_y 
                    ray_length_y += cpu_ray_cell_distances[2, sensor_number, individual]
                    side = 2
                    maze_walls = maze_walls_y
               end
            
                if maze_walls[current_cell_x, current_cell_y, individual]
                    hit = true
                else
                    side == 1 ? current_cell_x += step_direction_x : current_cell_y += step_direction_y 
                end      
            end
    
            current_distance -= environments.agent_radius
            sensor_distances[sensor_number, individual] = current_distance
        end    
    end
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
    config_environment["number_sensors"] = 10

    env_seed = rand(1:1000)
    number_individuals = 100

    environments = CollectPoints(config_environment, number_individuals)

    # In test code programm will jump out of gpu after every step
    #-> the in shared memory stored data has to be stored after every execution
    rewards = CUDA.fill(0, number_individuals)
    observations = CUDA.fill(0.0f0, (environments.number_outputs, number_individuals))
    test_sensor_distances = CUDA.fill(0.0f0, (convert(Int64, environments.number_sensors), number_individuals))
    test_ray_directions = CUDA.fill(0.0f0, (2, convert(Int64, environments.number_sensors), number_individuals))
    test_ray_cell_distances = CUDA.fill(0.0f0, (2, convert(Int64, environments.number_sensors), number_individuals))

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs + sizeof(Int64) * number_individuals

    thread_number = get_required_threads(environments)

    #---------------------------------------------------------------------------------------------------------------
    # Maze initialization tests
    #---------------------------------------------------------------------------------------------------------------

    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed, test_sensor_distances, test_ray_directions, test_ray_cell_distances)

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
    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed, test_sensor_distances, test_ray_directions, test_ray_cell_distances)

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

    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed_second, test_sensor_distances, test_ray_directions, test_ray_cell_distances)

    CUDA.synchronize()

    #Test wether environment is initialized different for new seed
    @test Array(environments.maze_walls_north) != maze_walls_north_cpu
    @test Array(environments.maze_walls_south) != maze_walls_south_cpu
    @test Array(environments.maze_walls_west) != maze_walls_west_cpu
    @test Array(environments.maze_walls_east) != maze_walls_east_cpu

    #---------------------------------------------------------------------------------------------------------------
    # Step function tests
    #---------------------------------------------------------------------------------------------------------------



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

    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)

    CUDA.synchronize()

    @test Array(environments.agents_positions) == new_agents_position

    #Tests wether points are collected properly
    rewards = CUDA.fill(0, number_individuals)

    #positive points
    copyto!(environments.agents_positions, original_agents_positions)

    pos_point_positions = original_agents_positions .+= environments.agent_movement_range
    copyto!(environments.positive_points_positions, pos_point_positions)

    fill!(actions, 1)

    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)

    CUDA.synchronize()

    @test pos_point_positions != Array(environments.positive_points_positions)

    @test Array(rewards) == fill(environments.reward_per_collected_positive_point, number_individuals)

    #negative points
    copyto!(environments.agents_positions, original_agents_positions)

    neg_point_positions = original_agents_positions .-= environments.agent_movement_range
    copyto!(environments.negative_points_positions, neg_point_positions)

    fill!(actions, -1)

    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)

    CUDA.synchronize()
    
    @test neg_point_positions != Array(environments.negative_points_positions)

    @test Array(rewards) == fill(environments.reward_per_collected_positive_point + environments.reward_per_collected_negative_point, number_individuals)

    #---------------------------------------------------------------------------------------------------------------
    # Sensor tests
    #---------------------------------------------------------------------------------------------------------------

    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed_second, test_sensor_distances, test_ray_directions, test_ray_cell_distances)

    CUDA.synchronize()

    #Test ray initialization
    cpu_ray_directions = Array(test_ray_directions)
    cpu_ray_distances = Array(test_ray_cell_distances)

    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            if cpu_ray_directions[1, sensor_number, individual] == 0.0
                @test isinf(cpu_ray_distances[1, sensor_number, individual])
            else
                x_distance =  sqrt(1 + (cpu_ray_directions[2, sensor_number, individual] / cpu_ray_directions[1, sensor_number, individual]) * (cpu_ray_directions[2, sensor_number, individual] / cpu_ray_directions[1, sensor_number, individual]))
                x_distance *= environments.maze_cell_size
                @test abs(cpu_ray_distances[1, sensor_number, individual]) - x_distance < 1e-4             
            end

            if cpu_ray_directions[2, sensor_number, individual] == 0.0
                @test isinf(cpu_ray_distances[2, sensor_number, individual])
            else
                y_distance =  sqrt(1 + (cpu_ray_directions[1, sensor_number, individual] / cpu_ray_directions[2, sensor_number, individual]) * (cpu_ray_directions[1, sensor_number, individual] / cpu_ray_directions[2, sensor_number, individual]))
                y_distance *= environments.maze_cell_size
                @test abs(cpu_ray_distances[2, sensor_number, individual]) - y_distance < 1e-4
            end
        end
    end
    
    #Sensor informations after agent Step

    normalization = max(environments.maze_cell_size * environments.maze_rows, environments.maze_cell_size * environments.maze_columns)

    #----------------------------
    #Test one step in x direction
    #----------------------------

    actions .= [1, 0]
    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)
    CUDA.synchronize()

    new_sensor_distances = Array(test_sensor_distances)
    new_agents_position = Array(environments.agents_positions)
    new_pos_position = Array(environments.positive_points_positions) 
    new_neg_position = Array(environments.negative_points_positions)
    cpu_observations = Array(observations)

    cpu_new_sensor_distances = fill(0.0f0, (environments.number_sensors, number_individuals))

    #Perform DDA on the cpu to calculate new sensor distances
    DDA_cpu(environments, cpu_new_sensor_distances, test_ray_directions, test_ray_cell_distances, number_individuals)

    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            @test new_sensor_distances[sensor_number, individual] == cpu_new_sensor_distances[sensor_number, individual]
        end
    end

    #Test Outputs (Observation)
    for individual = 1:number_individuals
        @test cpu_observations[1, individual] == convert(Float32, new_agents_position[1, individual] / normalization)
        @test cpu_observations[2, individual] == convert(Float32, new_agents_position[2, individual] / normalization)

        @test cpu_observations[3, individual] == convert(Float32, new_pos_position[1, individual] / normalization)
        @test cpu_observations[4, individual] == convert(Float32, new_pos_position[2, individual] / normalization)

        @test cpu_observations[5, individual] == convert(Float32, new_neg_position[1, individual] / normalization)
        @test cpu_observations[6, individual] == convert(Float32, new_neg_position[2, individual] / normalization)

        for sensor_number = 1:environments.number_sensors
            @test cpu_observations[sensor_number + 6, individual] == convert(Float32, new_sensor_distances[sensor_number, individual] / normalization)
        end    
    end



    #----------------------------
    #Test one step in y direction
    #----------------------------

    actions .= [0, 1]
    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)
    CUDA.synchronize()
    
    new_sensor_distances = Array(test_sensor_distances)
    new_agents_position = Array(environments.agents_positions)
    new_pos_position = Array(environments.positive_points_positions) 
    new_neg_position = Array(environments.negative_points_positions)
    cpu_observations = Array(observations)

    #distances
    DDA_cpu(environments, cpu_new_sensor_distances, cpu_ray_directions, cpu_ray_distances, number_individuals)
    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            @test new_sensor_distances[sensor_number, individual] == cpu_new_sensor_distances[sensor_number, individual]
        end
    end

    #observations
    #Test Outputs (Observation)
    for individual = 1:number_individuals
        @test cpu_observations[1, individual] == convert(Float32, new_agents_position[1, individual] / normalization)
        @test cpu_observations[2, individual] == convert(Float32, new_agents_position[2, individual] / normalization)

        @test cpu_observations[3, individual] == convert(Float32, new_pos_position[1, individual] / normalization)
        @test cpu_observations[4, individual] == convert(Float32, new_pos_position[2, individual] / normalization)

        @test cpu_observations[5, individual] == convert(Float32, new_neg_position[1, individual] / normalization)
        @test cpu_observations[6, individual] == convert(Float32, new_neg_position[2, individual] / normalization)

        for sensor_number = 1:environments.number_sensors
            @test cpu_observations[sensor_number + 6, individual] == convert(Float32, new_sensor_distances[sensor_number, individual] / normalization)
        end    
    end

    #----------------------------
    #Test random steps for each individual
    #----------------------------

    actions = rand(Uniform(-1, 1),(2, number_individuals))
    CUDA.@cuda threads = thread_number blocks = number_individuals shmem = shared_memory kernel_test_step(CuArray(actions), observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)
    CUDA.synchronize()
    
    new_sensor_distances = Array(test_sensor_distances)
    new_agents_position = Array(environments.agents_positions)
    new_pos_position = Array(environments.positive_points_positions) 
    new_neg_position = Array(environments.negative_points_positions)
    cpu_observations = Array(observations)

    #distances
    DDA_cpu(environments, cpu_new_sensor_distances, test_ray_directions, test_ray_cell_distances, number_individuals)
    for individual = 1:number_individuals
        for sensor_number = 1:environments.number_sensors
            @test new_sensor_distances[sensor_number, individual] == cpu_new_sensor_distances[sensor_number, individual]
        end
    end

    #observations
    for individual = 1:number_individuals
        @test cpu_observations[1, individual] == convert(Float32, new_agents_position[1, individual] / normalization)
        @test cpu_observations[2, individual] == convert(Float32, new_agents_position[2, individual] / normalization)

        @test cpu_observations[3, individual] == convert(Float32, new_pos_position[1, individual] / normalization)
        @test cpu_observations[4, individual] == convert(Float32, new_pos_position[2, individual] / normalization)

        @test cpu_observations[5, individual] == convert(Float32, new_neg_position[1, individual] / normalization)
        @test cpu_observations[6, individual] == convert(Float32, new_neg_position[2, individual] / normalization)

        for sensor_number = 1:environments.number_sensors
            @test cpu_observations[sensor_number + 6, individual] == convert(Float32, new_sensor_distances[sensor_number, individual] / normalization)
        end    
    end

end
