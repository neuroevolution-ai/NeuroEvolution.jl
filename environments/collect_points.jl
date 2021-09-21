using Adapt
using Random

struct CollectPoints{A,B}
    maze_columns::Int64
    maze_rows::Int64
    maze_cell_size::Int32
    agent_radius::Int64
    point_radius::Int32
    agent_movement_range::Float32
    reward_per_collected_positive_point::Float32
    reward_per_collected_negative_point::Float32
    number_time_steps::Int32
    number_inputs::Int64
    number_outputs::Int64
    mazes::A
    agents::B

end

function CollectPoints(configuration::OrderedDict, number_individuals::Int)

    CollectPoints(
        configuration["maze_columns"],
        configuration["maze_rows"],
        convert(Int32, configuration["maze_cell_size"]),
        configuration["agent_radius"],
        convert(Int32, configuration["point_radius"]),
        convert(Float32, configuration["agent_movement_range"]),
        convert(Float32, configuration["reward_per_collected_positive_point"]),
        convert(Float32, configuration["reward_per_collected_negative_point"]),
        convert(Int32, configuration["number_time_steps"]),
        10,
        2,
        CUDA.fill(
            1,
            (
                configuration["maze_rows"],
                configuration["maze_columns"],
                4,
                number_individuals,
            ),
        ),
        CUDA.fill(0, (6, number_individuals)),
    )
end

Adapt.@adapt_structure CollectPoints


function place_agent_randomly_in_maze(environment_cfg)
    x_coordinate = convert(
        Int32,
        (
            abs(rand(Int32)) %
            (environment_cfg.maze_cell_size - (2 * environment_cfg.agent_radius))
        ) +
        environment_cfg.agent_radius +
        ((abs(rand(Int32)) % environment_cfg.maze_columns)) *
        environment_cfg.maze_cell_size,
    )
    y_coordinate = convert(
        Int32,
        (
            abs(rand(Int32)) %
            (environment_cfg.maze_cell_size - (2 * environment_cfg.agent_radius))
        ) +
        environment_cfg.agent_radius +
        ((abs(rand(Int32)) % environment_cfg.maze_rows)) * environment_cfg.maze_cell_size,
    )
    return x_coordinate, y_coordinate
end

function get_memory_requirements(env_cfg::CollectPoints)
    return sizeof(Int32) * (env_cfg.maze_columns * env_cfg.maze_rows * 6 + 10)
end

function get_number_inputs(environments::CollectPoints)
    return environments.number_inputs
end

function get_number_outputs(environments::CollectPoints)
    return environments.number_outputs
end

function get_observation(maze, input, environment_config_array, env_cfg)
    cell_x =
        convert(Int32, ceil(@inbounds environment_config_array[1] / env_cfg.maze_cell_size))
    cell_y =
        convert(Int32, ceil(@inbounds environment_config_array[2] / env_cfg.maze_cell_size))
    screen_width = env_cfg.maze_cell_size * env_cfg.maze_columns
    screen_height = env_cfg.maze_cell_size * env_cfg.maze_rows

    x_left = env_cfg.maze_cell_size * (cell_x - 1)
    x_right = env_cfg.maze_cell_size * cell_x
    y_bottom = env_cfg.maze_cell_size * (cell_y - 1)
    y_top = env_cfg.maze_cell_size * cell_y

    sensor_north = begin
        sensor_distance =
            y_top - @inbounds environment_config_array[2] - env_cfg.agent_radius
        direction = 1
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y + 1) > env_cfg.maze_rows
                break
            end
            current_cell_y += 1
            if maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    sensor_east = begin
        sensor_distance =
            x_right - @inbounds environment_config_array[1] - env_cfg.agent_radius
        direction = 2
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x - 1) < 1
                break
            end
            current_cell_x -= 1
            if maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    sensor_south = begin
        sensor_distance =
            @inbounds environment_config_array[2] - y_bottom - env_cfg.agent_radius
        direction = 3
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y - 1) < 1
                break
            end
            current_cell_y -= 1
            if maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    sensor_west = begin
        sensor_distance =
            @inbounds environment_config_array[1] - x_left - env_cfg.agent_radius
        direction = 4
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x + 1) > env_cfg.maze_columns
                break
            end
            current_cell_x += 1
            if maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end

    @inbounds input[1] =
        convert(Float32, @inbounds environment_config_array[1] / screen_width)
    @inbounds input[2] =
        convert(Float32, @inbounds environment_config_array[2] / screen_height)
    @inbounds input[3] =
        convert(Float32, @inbounds environment_config_array[3] / screen_width)
    @inbounds input[4] =
        convert(Float32, @inbounds environment_config_array[4] / screen_height)
    @inbounds input[5] =
        convert(Float32, @inbounds environment_config_array[5] / screen_width)
    @inbounds input[6] =
        convert(Float32, @inbounds environment_config_array[6] / screen_height)
    @inbounds input[7] = convert(Float32, sensor_north / screen_height)
    @inbounds input[8] = convert(Float32, sensor_east / screen_width)
    @inbounds input[9] = convert(Float32, sensor_south / screen_height)
    @inbounds input[10] = convert(Float32, sensor_west / screen_height)
end

function env_step(maze, action, input, environment_config_array, env_cfg::CollectPoints)
    screen_width = env_cfg.maze_cell_size * env_cfg.maze_columns
    screen_height = env_cfg.maze_cell_size * env_cfg.maze_rows
    @inbounds agent_x_coordinate =
        environment_config_array[1] + clamp(
            floor(action[1] * env_cfg.agent_movement_range),
            -env_cfg.agent_movement_range,
            env_cfg.agent_movement_range,
        )
    @inbounds agent_y_coordinate =
        environment_config_array[2] + clamp(
            floor(action[2] * env_cfg.agent_movement_range),
            -env_cfg.agent_movement_range,
            env_cfg.agent_movement_range,
        )


    # Check agent collisions with outer walls
    agent_y_coordinate = max(agent_y_coordinate, env_cfg.agent_radius) # Upper border
    agent_y_coordinate = min(agent_y_coordinate, screen_height - env_cfg.agent_radius) # Lower bord.
    agent_x_coordinate = min(agent_x_coordinate, screen_width - env_cfg.agent_radius) # Right border
    agent_x_coordinate = max(agent_x_coordinate, env_cfg.agent_radius) # Left border

    # Get cell indizes of agents current position
    cell_x = convert(Int32, ceil(agent_x_coordinate / env_cfg.maze_cell_size))
    cell_y = convert(Int32, ceil(agent_y_coordinate / env_cfg.maze_cell_size))


    # Get coordinates of current cell
    x_left = env_cfg.maze_cell_size * (cell_x - 1)
    x_right = env_cfg.maze_cell_size * cell_x
    y_bottom = env_cfg.maze_cell_size * (cell_y - 1)
    y_top = env_cfg.maze_cell_size * cell_y
    # Check agent collisions with maze walls

    if maze[cell_y, cell_x, 1] == 0 #check for Northern Wall
        agent_y_coordinate = min(agent_y_coordinate, y_top - env_cfg.agent_radius)
    end
    if maze[cell_y, cell_x, 3] == 0 #check for Southern Wall
        agent_y_coordinate = max(agent_y_coordinate, y_bottom + env_cfg.agent_radius)
    end
    if maze[cell_y, cell_x, 2] == 0 #check for Eastern Wall
        agent_x_coordinate = min(agent_x_coordinate, x_right - env_cfg.agent_radius)
    end
    if maze[cell_y, cell_x, 4] == 0 #check for Western Wall
        agent_x_coordinate = max(agent_x_coordinate, x_left + env_cfg.agent_radius)
    end
    if (agent_x_coordinate - x_left < env_cfg.agent_radius) &&
       (y_top - agent_y_coordinate < env_cfg.agent_radius)
        agent_x_coordinate = x_left + env_cfg.agent_radius
        agent_y_coordinate = y_top - env_cfg.agent_radius
    end
    # Check agent collision with top-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < env_cfg.agent_radius) &&
       (y_top - agent_y_coordinate < env_cfg.agent_radius)
        agent_x_coordinate = x_right - env_cfg.agent_radius
        agent_y_coordinate = y_top - env_cfg.agent_radius
    end
    # Check agent collision with bottom-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < env_cfg.agent_radius) &&
       (agent_y_coordinate - y_bottom < env_cfg.agent_radius)
        agent_x_coordinate = x_right - env_cfg.agent_radius
        agent_y_coordinate = y_bottom + env_cfg.agent_radius
    end
    # Check agent collision with bottom-left edge (prevents sneaking through the edge)
    if (agent_x_coordinate - x_left < env_cfg.agent_radius) &&
       (agent_y_coordinate - y_bottom < env_cfg.agent_radius)
        agent_x_coordinate = x_left + env_cfg.agent_radius
        agent_y_coordinate = y_bottom + env_cfg.agent_radius
    end
    @inbounds environment_config_array[1] = agent_x_coordinate
    @inbounds environment_config_array[2] = agent_y_coordinate

    sensor_north = begin
        sensor_distance = y_top - agent_y_coordinate - env_cfg.agent_radius
        direction = 1
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y + 1) > env_cfg.maze_rows
                break
            end
            current_cell_y += 1
            if @inbounds maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    sensor_east = begin
        sensor_distance = x_right - agent_x_coordinate - env_cfg.agent_radius
        direction = 2
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x - 1) < 1
                break
            end
            current_cell_x -= 1
            if @inbounds maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    sensor_south = begin
        sensor_distance = agent_y_coordinate - y_bottom - env_cfg.agent_radius
        direction = 3
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y - 1) < 1
                break
            end
            current_cell_y -= 1
            if @inbounds maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    sensor_west = begin
        sensor_distance = agent_x_coordinate - x_left - env_cfg.agent_radius
        direction = 4
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x + 1) > env_cfg.maze_columns
                break
            end
            current_cell_x += 1
            if @inbounds maze[current_cell_y, current_cell_x, direction] == 0
                break
            else
                sensor_distance += env_cfg.maze_cell_size
            end
        end
        sensor_distance
    end
    rew = 0.0f0
    # Collect positive point in reach
    distance = sqrt(
        (environment_config_array[3] - agent_x_coordinate)^2 +
        (environment_config_array[4] - agent_y_coordinate)^2,
    )
    if distance <= env_cfg.point_radius + env_cfg.agent_radius
        #place new positive_point randomly in maze
        @inbounds environment_config_array[3], environment_config_array[4] =
            place_agent_randomly_in_maze(env_cfg)
        rew = env_cfg.reward_per_collected_positive_point
    end
    # Collect negative point in reach
    distance = sqrt(
        (environment_config_array[5] - agent_x_coordinate)^2 +
        (environment_config_array[6] - agent_y_coordinate)^2,
    )
    if distance <= env_cfg.point_radius + env_cfg.agent_radius
        #place new negative_point randomly in maze
        @inbounds environment_config_array[5], environment_config_array[6] =
            place_agent_randomly_in_maze(env_cfg)
        rew = env_cfg.reward_per_collected_negative_point
    end
    #get state of environment as Input for Brain
    #############################################

    @inbounds input[1] = convert(Float32, agent_x_coordinate / screen_width)
    @inbounds input[2] = convert(Float32, agent_y_coordinate / screen_height)
    @inbounds input[3] = convert(Float32, environment_config_array[3] / screen_width)
    @inbounds input[4] = convert(Float32, environment_config_array[4] / screen_height)
    @inbounds input[5] = convert(Float32, environment_config_array[5] / screen_width)
    @inbounds input[6] = convert(Float32, environment_config_array[6] / screen_height)
    @inbounds input[7] = convert(Float32, sensor_north / screen_height)
    @inbounds input[8] = convert(Float32, sensor_east / screen_width)
    @inbounds input[9] = convert(Float32, sensor_south / screen_height)
    @inbounds input[10] = convert(Float32, sensor_west / screen_width)
    return rew
end

function create_maze(maze, env_cfg::CollectPoints, offset)
    total_amount_of_cells = env_cfg.maze_columns * env_cfg.maze_rows

    x_coordinate_stack = @cuDynamicSharedMem(Int32, total_amount_of_cells, offset)
    y_coordinate_stack = @cuDynamicSharedMem(
        Int32,
        total_amount_of_cells,
        offset + sizeof(x_coordinate_stack)
    )
    neighbours = @cuDynamicSharedMem(
        Int32,
        4,
        offset + sizeof(x_coordinate_stack) + sizeof(y_coordinate_stack)
    )

    cell_x_coordinate = 1
    cell_y_coordinate = 1
    amount_of_cells_visited = 1
    cell_stack_index = 1
    for j = 1:4
        for k = 1:env_cfg.maze_rows
            for l = 1:env_cfg.maze_columns
                @inbounds maze[k, l, j] = convert(Int32, 0)
            end
        end
    end

    while amount_of_cells_visited < total_amount_of_cells
        for i = 1:4
            @inbounds neighbours[i] = 0
        end
        #step1: find all neighboring cells which have not been visited yet
        if (cell_x_coordinate + 1) <= env_cfg.maze_columns
            if @inbounds maze[cell_y_coordinate, cell_x_coordinate+1, 1] == 0 &&
                         maze[cell_y_coordinate, cell_x_coordinate+1, 2] == 0 &&
                         maze[cell_y_coordinate, cell_x_coordinate+1, 3] == 0 &&
                         maze[cell_y_coordinate, cell_x_coordinate+1, 4] == 0
                @inbounds neighbours[1] = 1
            else
                @inbounds neighbours[1] = 0
            end
        end
        if (cell_x_coordinate - 1) >= 1
            if @inbounds maze[cell_y_coordinate, cell_x_coordinate-1, 1] == 0 &&
                         maze[cell_y_coordinate, cell_x_coordinate-1, 2] == 0 &&
                         maze[cell_y_coordinate, cell_x_coordinate-1, 3] == 0 &&
                         maze[cell_y_coordinate, cell_x_coordinate-1, 4] == 0
                @inbounds neighbours[2] = 1
            else
                @inbounds neighbours[2] = 0
            end
        end
        if (cell_y_coordinate + 1) <= env_cfg.maze_rows
            if @inbounds maze[cell_y_coordinate+1, cell_x_coordinate, 1] == 0 &&
                         maze[cell_y_coordinate+1, cell_x_coordinate, 2] == 0 &&
                         maze[cell_y_coordinate+1, cell_x_coordinate, 3] == 0 &&
                         maze[cell_y_coordinate+1, cell_x_coordinate, 4] == 0
                @inbounds neighbours[3] = 1
            else
                @inbounds neighbours[3] = 0
            end
        end
        if (cell_y_coordinate - 1) >= 1
            if @inbounds maze[cell_y_coordinate-1, cell_x_coordinate, 1] == 0 &&
                         maze[cell_y_coordinate-1, cell_x_coordinate, 2] == 0 &&
                         maze[cell_y_coordinate-1, cell_x_coordinate, 3] == 0 &&
                         maze[cell_y_coordinate-1, cell_x_coordinate, 4] == 0
                @inbounds neighbours[4] = 1
            else
                @inbounds neighbours[4] = 0
            end
        end
        if @inbounds neighbours[1] == 0 &&
                     neighbours[2] == 0 &&
                     neighbours[3] == 0 &&
                     neighbours[4] == 0
            cell_stack_index = cell_stack_index - 1
            @inbounds cell_x_coordinate = x_coordinate_stack[cell_stack_index]
            @inbounds cell_y_coordinate = y_coordinate_stack[cell_stack_index]
            continue
        end

        move_x_coordinate = 0
        move_y_coordinate = 0

        #step3: choose random neighbor to move to
        rand_index = (abs(rand(Int32)) % 4)
        for i = 1:4
            index = ((rand_index + i) % 4) + 1
            if @inbounds neighbours[index] == 1
                if index == 3
                    move_y_coordinate = 1
                    break
                end
                if index == 1
                    move_x_coordinate = 1
                    break
                end
                if index == 4
                    move_y_coordinate = -1
                    break
                end
                if index == 2
                    move_x_coordinate = -1
                    break
                end
            end
        end

        #step4: knock down the wall between the cells for both cells   
        if move_x_coordinate == 1
            @inbounds maze[cell_y_coordinate, cell_x_coordinate, 2] = 1
            @inbounds maze[cell_y_coordinate, cell_x_coordinate+move_x_coordinate, 4] = 1
        end
        if move_x_coordinate == -1
            @inbounds maze[cell_y_coordinate, cell_x_coordinate, 4] = 1
            @inbounds maze[cell_y_coordinate, cell_x_coordinate+move_x_coordinate, 2] = 1
        end
        if move_y_coordinate == 1
            @inbounds maze[cell_y_coordinate, cell_x_coordinate, 1] = 1
            @inbounds maze[cell_y_coordinate+move_y_coordinate, cell_x_coordinate, 3] = 1
        end
        if move_y_coordinate == -1
            @inbounds maze[cell_y_coordinate, cell_x_coordinate, 3] = 1
            @inbounds maze[cell_y_coordinate+move_y_coordinate, cell_x_coordinate, 1] = 1
        end

        #step5: add origin cell to stack
        @inbounds x_coordinate_stack[cell_stack_index] = cell_x_coordinate
        @inbounds y_coordinate_stack[cell_stack_index] = cell_y_coordinate
        cell_stack_index = cell_stack_index + 1
        #step6: set coordinates to new cell
        cell_x_coordinate = cell_x_coordinate + move_x_coordinate
        cell_y_coordinate = cell_y_coordinate + move_y_coordinate
        amount_of_cells_visited = amount_of_cells_visited + 1

    end

end
