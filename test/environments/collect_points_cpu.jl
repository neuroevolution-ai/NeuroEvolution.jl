using Distributions

include("create_maze.jl")

struct CollectPointsCpuCfg
    maze_columns::Int
    maze_rows::Int
    maze_cell_size::Int
    agent_radius::Int
    point_radius::Int
    agent_movement_range::Float32
    number_of_sensors::Int
    reward_per_collected_positive_point::Float32
    reward_per_collected_negative_point::Float32
    number_time_steps::Int

    function CollectPointsCpuCfg(config::OrderedDict)

        new(
            config["maze_columns"],
            config["maze_rows"],
            config["maze_cell_size"],
            config["agent_radius"],
            config["point_radius"],
            config["agent_movement_range"],
            4,
            config["reward_per_collected_positive_point"],
            config["reward_per_collected_negative_point"],
            config["number_time_steps"],
        )

    end
end

mutable struct CollectPointsCpu
    config::CollectPointsCpuCfg
    screen_width::Int
    screen_height::Int
    agent_position_x::Int
    agent_position_y::Int
    point_x::Int
    point_y::Int
    negative_point_x::Int
    negative_point_y::Int
    maze::Any
    sensor_top::Float32
    sensor_bottom::Float32
    sensor_left::Float32
    sensor_right::Float32
    t::Float32

    function CollectPointsCpu(config::CollectPointsCpuCfg)

        screen_width = config.maze_cell_size * config.maze_columns
        screen_height = config.maze_cell_size * config.maze_rows

        # Agent coordinates
        agent_position_x, agent_position_y = place_randomly_in_maze(config, config.agent_radius)

        # Point coordinates for positive point
        point_x, point_y = place_randomly_in_maze(config, config.point_radius)

        # Point coordinates for negative point
        negative_point_x, negative_point_y = place_randomly_in_maze(config, config.point_radius)

        # Create Maze
        maze = make_maze(config.maze_columns, config.maze_rows)

        sensor_top = 0.0
        sensor_bottom = 0.0
        sensor_left = 0.0
        sensor_right = 0.0

        t = 0

        new(
            configuration,
            screen_width,
            screen_height,
            agent_position_x,
            agent_position_y,
            point_x,
            point_y,
            negative_point_x,
            negative_point_y,
            maze,
            sensor_top,
            sensor_bottom,
            sensor_left,
            sensor_right,
            t,
        )

    end
end

function step(env::CollectPointsCpu)

    action = rand(Uniform(-1.0,1.0), 2)

    # Movement range for agent
    mr = env.config.agent_movement_range

    # Move agent
    env.agent_position_x += convert(Int, round(clamp(action[1] * mr, -mr, mr)))
    env.agent_position_y += convert(Int, round(clamp(action[2] * mr, -mr, mr)))

    # Check agent collisions with outer walls
    env.agent_position_y = max(env.agent_position_y, env.config.agent_radius)  # Upper border
    env.agent_position_y = min(env.agent_position_y, env.screen_height - env.config.agent_radius)  # Lower bord.
    env.agent_position_x = min(env.agent_position_x, env.screen_width - env.config.agent_radius)  # Right border
    env.agent_position_x = max(env.agent_position_x, env.config.agent_radius)  # Left border

    # Get cell indizes of agents current position
    cell_x = convert(Int, ceil(env.agent_position_x / env.config.maze_cell_size))
    cell_y = convert(Int, ceil(env.agent_position_y / env.config.maze_cell_size))

    # Get current cell
    cell = env.maze[cell_x, cell_y]

    # Get coordinates of current cell
    x_left, x_right, y_top, y_bottom = get_coordinates_maze_cell(cell_x, cell_y, env.config.maze_cell_size)

    # Check agent collisions with maze walls
    if cell.walls[North] == true
        env.agent_position_y = max(env.agent_position_y, y_top + env.config.agent_radius)
    end
    if cell.walls[South] == true
        env.agent_position_y = min(env.agent_position_y, y_bottom - env.config.agent_radius)
    end
    if cell.walls[East] == true
        env.agent_position_x = min(env.agent_position_x, x_right - env.config.agent_radius)
    end
    if cell.walls[West] == true
        env.agent_position_x = max(env.agent_position_x, x_left + env.config.agent_radius)
    end

    reward = 0.0

    # Collect positive point in reach
    distance = sqrt((env.point_x - env.agent_position_x)^2 + (env.point_y - env.agent_position_y)^2)
    if distance <= env.config.point_radius + env.config.agent_radius
        env.point_x, env.point_y = place_randomly_in_maze(env.config, env.config.point_radius)
        reward += env.config.reward_per_collected_positive_point
    end

    # Collect negative point in reach
    distance = sqrt((env.negative_point_x - env.agent_position_x)^2 + (env.negative_point_y - env.agent_position_y)^2)
    if distance <= env.config.point_radius + env.config.agent_radius
        env.negative_point_x, env.negative_point_y = place_randomly_in_maze(env.config, env.config.point_radius)
        reward += env.config.reward_per_collected_negative_point
    end

    println(reward)

    return

end

function place_randomly_in_maze(config::CollectPointsCpuCfg, radius::Int)

    x = rand(radius:(config.maze_cell_size - radius)) + rand(0:config.maze_columns-1) * config.maze_cell_size
    y = rand(radius:(config.maze_cell_size - radius)) + rand(0:config.maze_rows-1) * config.maze_cell_size

    return x, y
end
