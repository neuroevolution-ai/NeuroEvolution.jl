using Luxor

include("environments/collect_points.jl")
include("tools/play.jl")
include("tools/linear_congruential_generator.jl")

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

    if tx == 1
        for i = (environments.number_sensors + 1):environments.number_outputs
            observations[i, bx] = observation[i]
        end    
    end    

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

    test_sensor_distances = CUDA.fill(0.0f0, (convert(Int64, environments.number_sensors), number_individuals))
    test_ray_directions = CUDA.fill(0.0f0, (2, convert(Int64, environments.number_sensors), number_individuals))
    test_ray_cell_distances = CUDA.fill(0.0f0, (2, convert(Int64, environments.number_sensors), number_individuals))

    shared_memory = get_memory_requirements(environments) + sizeof(Float32) * environments.number_inputs + sizeof(Int64) * number_individuals

    CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_initialize(environments, env_seed, test_sensor_distances, test_ray_directions, test_ray_cell_distances)

    CUDA.synchronize()

    number_iterations = 1000
    width = environments.maze_rows * environments.maze_cell_size
    height = environments.maze_columns * environments.maze_cell_size

    observations = CUDA.fill(0.0f0, (environments.number_outputs, number_individuals))

    rewards = CUDA.fill(0, number_individuals)
    

    @play width height number_iterations begin
        actions = CuArray(rand(Uniform(-1, 1),(2, number_individuals)))
        CUDA.@cuda threads = 10 blocks = number_individuals shmem = shared_memory kernel_test_step(actions, observations, test_sensor_distances, test_ray_directions, test_ray_cell_distances, rewards, environments)
        CUDA.synchronize()

        render(test_sensor_distances, environments, 1)
    end
end


main()