using CUDA
using Adapt
using DataStructures
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
    maze_walls_north::A
    maze_walls_east::A
    maze_walls_south::A
    maze_walls_west::A
    maze_randoms::B
    agents_positions::B
    positive_points_positions::B
    negative_points_positions::B

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
        CUDA.fill(true, (configuration["maze_columns"], configuration["maze_rows"], number_individuals)),
        CUDA.fill(true, (configuration["maze_columns"], configuration["maze_rows"], number_individuals)),
        CUDA.fill(true, (configuration["maze_columns"], configuration["maze_rows"], number_individuals)),
        CUDA.fill(true, (configuration["maze_columns"], configuration["maze_rows"], number_individuals)),
        CUDA.fill(0, (configuration["maze_columns"] * configuration["maze_rows"], number_individuals)),
        CUDA.fill(0, (2, number_individuals)),
        CUDA.fill(0, (2, number_individuals)),
        CUDA.fill(0, (2, number_individuals)),
    )
end


Adapt.@adapt_structure CollectPoints


function initialize(threadID, blockID, input, environments::CollectPoints, offset, env_seed)

    if threadID == 1
        Random.seed!(Random.default_rng(), env_seed)
        create_maze(threadID, blockID, environments, offset)
    end

    sync_threads()

    # Place agent randomly in maze
    if threadID == 1
        place_randomly_in_maze(blockID, environments.agents_positions, environments.agent_radius, environments)
    end

    # Place positive point randomly in maze
    if threadID == 2
        place_randomly_in_maze(blockID, environments.positive_points_positions, environments.point_radius, environments)
    end

    # Place negative point randomly in maze
    if threadID == 3
        place_randomly_in_maze(blockID, environments.negative_points_positions, environments.point_radius, environments)
    end
    return

end


function place_randomly_in_maze(blockID, Position, radius, environments::CollectPoints)

    Position[1, blockID] = convert(Int32, (abs(rand(Int32)) % (environments.maze_cell_size - (2 * radius))) +
        environments.agent_radius +
        ((abs(rand(Int32)) % environments.maze_columns)) * environments.maze_cell_size)

    Position[2, blockID] = convert(Int32, (abs(rand(Int32)) % (environments.maze_cell_size - (2 * radius))) +
        environments.agent_radius +
        ((abs(rand(Int32)) % environments.maze_rows)) * environments.maze_cell_size)

    return
end

function get_memory_requirements(environments::CollectPoints)
    total_amount_of_cells = environments.maze_columns * environments.maze_rows

    return sizeof(Int32) * (2 * total_amount_of_cells) + sizeof(Int32) * 8
end

function get_number_inputs(environments::CollectPoints)
    return environments.number_inputs
end

function get_number_outputs(environments::CollectPoints)
    return environments.number_outputs
end

function get_observation(threadID, blockID, input, environments::CollectPoints)

    cell_x = convert(Int32, ceil(@inbounds environments.agents_positions[1, blockID] / environments.maze_cell_size))
    cell_y = convert(Int32, ceil(@inbounds environments.agents_positions[2, blockID] / environments.maze_cell_size))

    screen_width = environments.maze_cell_size * environments.maze_columns
    screen_height = environments.maze_cell_size * environments.maze_rows

    x_left = environments.maze_cell_size * (cell_x - 1)
    x_right = environments.maze_cell_size * cell_x
    y_bottom = environments.maze_cell_size * (cell_y - 1)
    y_top = environments.maze_cell_size * cell_y

    sensor_north = begin
        sensor_distance = y_top - @inbounds environments.agents_positions[2, blockID] - environments.agent_radius
        direction = 1
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y + 1) > environments.maze_rows
                break
            end
            current_cell_y += 1
            if environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    sensor_east = begin
        sensor_distance = x_right - @inbounds environments.agents_positions[1, blockID] - environments.agent_radius
        direction = 2
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x - 1) < 1
                break
            end
            current_cell_x -= 1
            if environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    sensor_south = begin
        sensor_distance = @inbounds environments.agents_positions[2, blockID] - y_bottom - environments.agent_radius
        direction = 3
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y - 1) < 1
                break
            end
            current_cell_y -= 1
            if environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    sensor_west = begin
        sensor_distance = @inbounds environments.agents_positions[1, blockID] - x_left - environments.agent_radius
        direction = 4
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x + 1) > environments.maze_columns
                break
            end
            current_cell_x += 1
            if environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    @inbounds input[1] = convert(Float32, @inbounds environments.agents_positions[1, blockID] / screen_width)
    @inbounds input[2] = convert(Float32, @inbounds environments.agents_positions[2, blockID] / screen_height)
    @inbounds input[3] = convert(Float32, @inbounds environments.positive_points_positions[1, blockID] / screen_width)
    @inbounds input[4] = convert(Float32, @inbounds environments.positive_points_positions[2, blockID] / screen_height)
    @inbounds input[5] = convert(Float32, @inbounds environments.negative_points_positions[1, blockID] / screen_width)
    @inbounds input[6] = convert(Float32, @inbounds environments.negative_points_positions[2, blockID] / screen_height)
    @inbounds input[7] = convert(Float32, sensor_north / screen_height)
    @inbounds input[8] = convert(Float32, sensor_east / screen_width)
    @inbounds input[9] = convert(Float32, sensor_south / screen_height)
    @inbounds input[10] = convert(Float32, sensor_west / screen_height)
end

function env_step(threadID, blockID, input, action, environments::CollectPoints)

    screen_width = environments.maze_cell_size * environments.maze_columns
    screen_height = environments.maze_cell_size * environments.maze_rows

    @inbounds agent_x_coordinate = environments.environment_config_array[1, blockID] + 
        clamp(floor(action[1] * environments.agent_movement_range), -environments.agent_movement_range, environments.agent_movement_range)

    @inbounds agent_y_coordinate = environments.environment_config_array[2, blockID] + 
        clamp(floor(action[2] * environments.agent_movement_range), -environments.agent_movement_range, environments.agent_movement_range)

    # Check agent collisions with outer walls
    agent_y_coordinate = max(agent_y_coordinate, environments.agent_radius) # Upper border
    agent_y_coordinate = min(agent_y_coordinate, screen_height - environments.agent_radius) # Lower bord.
    agent_x_coordinate = min(agent_x_coordinate, screen_width - environments.agent_radius) # Right border
    agent_x_coordinate = max(agent_x_coordinate, environments.agent_radius) # Left border

    # Get cell indizes of agents current position
    cell_x = convert(Int32, ceil(agent_x_coordinate / environments.maze_cell_size))
    cell_y = convert(Int32, ceil(agent_y_coordinate / environments.maze_cell_size))

    # Get coordinates of current cell
    x_left = environments.maze_cell_size * (cell_x - 1)
    x_right = environments.maze_cell_size * cell_x
    y_bottom = environments.maze_cell_size * (cell_y - 1)
    y_top = environments.maze_cell_size * cell_y

    # Check agent collisions with maze walls
    if environments.mazes[cell_y, cell_x, 1, blockID] == 0 #check for Northern Wall
        agent_y_coordinate = min(agent_y_coordinate, y_top - environments.agent_radius)
    end

    if environments.mazes[cell_y, cell_x, 3, blockID] == 0 #check for Southern Wall
        agent_y_coordinate = max(agent_y_coordinate, y_bottom + environments.agent_radius)
    end

    if environments.mazes[cell_y, cell_x, 2, blockID] == 0 #check for Eastern Wall
        agent_x_coordinate = min(agent_x_coordinate, x_right - environments.agent_radius)
    end

    if environments.mazes[cell_y, cell_x, 4, blockID] == 0 #check for Western Wall
        agent_x_coordinate = max(agent_x_coordinate, x_left + environments.agent_radius)
    end

    if (agent_x_coordinate - x_left < environments.agent_radius) &&
       (y_top - agent_y_coordinate < environments.agent_radius)
        agent_x_coordinate = x_left + environments.agent_radius
        agent_y_coordinate = y_top - environments.agent_radius
    end

    # Check agent collision with top-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < environments.agent_radius) &&
       (y_top - agent_y_coordinate < environments.agent_radius)
        agent_x_coordinate = x_right - environments.agent_radius
        agent_y_coordinate = y_top - environments.agent_radius
    end

    # Check agent collision with bottom-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < environments.agent_radius) &&
       (agent_y_coordinate - y_bottom < environments.agent_radius)
        agent_x_coordinate = x_right - environments.agent_radius
        agent_y_coordinate = y_bottom + environments.agent_radius
    end

    # Check agent collision with bottom-left edge (prevents sneaking through the edge)
    if (agent_x_coordinate - x_left < environments.agent_radius) &&
       (agent_y_coordinate - y_bottom < environments.agent_radius)
        agent_x_coordinate = x_left + environments.agent_radius
        agent_y_coordinate = y_bottom + environments.agent_radius
    end

    @inbounds environments.environment_config_array[1, blockID] = agent_x_coordinate
    @inbounds environments.environment_config_array[2, blockID] = agent_y_coordinate

    sensor_north = begin
        sensor_distance = y_top - agent_y_coordinate - environments.agent_radius
        direction = 1
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y + 1) > environments.maze_rows
                break
            end
            current_cell_y += 1
            if @inbounds environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    sensor_east = begin
        sensor_distance = x_right - agent_x_coordinate - environments.agent_radius
        direction = 2
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x - 1) < 1
                break
            end
            current_cell_x -= 1
            if @inbounds environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    sensor_south = begin
        sensor_distance = agent_y_coordinate - y_bottom - environments.agent_radius
        direction = 3
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_y - 1) < 1
                break
            end
            current_cell_y -= 1
            if @inbounds environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    sensor_west = begin
        sensor_distance = agent_x_coordinate - x_left - environments.agent_radius
        direction = 4
        current_cell_x = cell_x
        current_cell_y = cell_y
        while true
            if (current_cell_x + 1) > environments.maze_columns
                break
            end
            current_cell_x += 1
            if @inbounds environments.mazes[current_cell_y, current_cell_x, direction, blockID] == 0
                break
            else
                sensor_distance += environments.maze_cell_size
            end
        end
        sensor_distance
    end

    rew = 0.0f0

    # Collect positive point in reach
    distance = sqrt((environments.environment_config_array[3, blockID] - agent_x_coordinate)^2 +
        (environments.environment_config_array[4, blockID] - agent_y_coordinate)^2)

    if distance <= environments.point_radius + environments.agent_radius

        #place new positive_point randomly in maze
        @inbounds environments.environment_config_array[3, blockID], environments.environment_config_array[4, blockID] = place_randomly_in_maze(environments)

        rew = environments.reward_per_collected_positive_point

    end

    # Collect negative point in reach
    distance = sqrt((environments.environment_config_array[5, blockID] - agent_x_coordinate)^2 + (environments.environment_config_array[6, blockID] - agent_y_coordinate)^2)

    if distance <= environments.point_radius + environments.agent_radius

        #place new negative_point randomly in maze
        @inbounds environments.environment_config_array[5, blockID], environments.environment_config_array[6, blockID] = place_randomly_in_maze(environments)

        rew = environments.reward_per_collected_negative_point
    end

    #get state of environment as Input for Brain
    @inbounds input[1] = convert(Float32, agent_x_coordinate / screen_width)
    @inbounds input[2] = convert(Float32, agent_y_coordinate / screen_height)
    @inbounds input[3] = convert(Float32, environments.environment_config_array[3, blockID] / screen_width)
    @inbounds input[4] = convert(Float32, environments.environment_config_array[4, blockID] / screen_height)
    @inbounds input[5] = convert(Float32, environments.environment_config_array[5, blockID] / screen_width)
    @inbounds input[6] = convert(Float32, environments.environment_config_array[6, blockID] / screen_height)
    @inbounds input[7] = convert(Float32, sensor_north / screen_height)
    @inbounds input[8] = convert(Float32, sensor_east / screen_width)
    @inbounds input[9] = convert(Float32, sensor_south / screen_height)
    @inbounds input[10] = convert(Float32, sensor_west / screen_width)

    return rew
end

function render(environments::CollectPoints)

    # White white background
    background("white")

    # Draw agent
    sethue("green")
    agent_position = Array(environments.agents_positions[:, 1])
    circle(agent_position[1], agent_position[2], environments.agent_radius, :fill)

    # Draw positive point
    sethue("blue")
    positive_points_position = Array(environments.positive_points_positions[:, 1])
    circle(positive_points_position[1], positive_points_position[2], environments.point_radius, :fill)

    # Draw negative point
    sethue("red")
    negative_points_position = Array(environments.negative_points_positions[:, 1])
    circle(negative_points_position[1], negative_points_position[2], environments.point_radius, :fill)

    # Draw maze
    sethue("black")
    setline(2)
    maze_walls_north = Array(environments.maze_walls_north[:, :, 1])
    maze_walls_east = Array(environments.maze_walls_east[:, :, 1])
    maze_walls_south = Array(environments.maze_walls_south[:, :, 1])
    maze_walls_west = Array(environments.maze_walls_west[:, :, 1])

    for cell_x = 1:environments.maze_columns
        for cell_y = 1:environments.maze_rows

            x_left, x_right, y_top, y_bottom = get_coordinates_maze_cell(cell_x, cell_y, environments.maze_cell_size)

            # Draw walls
            if maze_walls_north[cell_x, cell_y] == true
                line(Point(x_left, y_top), Point(x_right, y_top), :stroke)
            end

            if maze_walls_south[cell_x, cell_y] == true
                line(Point(x_left, y_bottom), Point(x_right, y_bottom), :stroke)
            end

            if maze_walls_east[cell_x, cell_y] == true
                line(Point(x_right, y_top), Point(x_right, y_bottom), :stroke)
            end

            if maze_walls_west[cell_x, cell_y] == true
                line(Point(x_left, y_top), Point(x_left, y_bottom), :stroke)
            end

        end
    end
end

function get_coordinates_maze_cell(cell_x::Int, cell_y::Int, maze_cell_size::Int32)

    x_left = maze_cell_size * (cell_x - 1)
    x_right = maze_cell_size * cell_x
    y_top = maze_cell_size * (cell_y - 1)
    y_bottom = maze_cell_size * cell_y

    return x_left, x_right, y_top, y_bottom

end


function create_maze(threadID, blockID, environments::CollectPoints, offset)

    # Total number of cells.
    total_amount_of_cells = environments.maze_columns * environments.maze_rows

    index_cell_stack = 0
    cell_stack = @cuDynamicSharedMem(Int32, (total_amount_of_cells, 2), offset)
    offset += sizeof(cell_stack)

    neighbours = @cuDynamicSharedMem(Int32, (4, 2), offset)

    # Total number of visited cells during maze construction.
    nv = 1

    current_cell_x = 1
    current_cell_y = 1

    while nv < total_amount_of_cells

        number_neighbours = find_valid_neighbors(blockID, current_cell_x, current_cell_y, neighbours, environments)

        if number_neighbours == 0
            # We've reached a dead end: backtrack, pop cell from cell stack
            current_cell_x = cell_stack[index_cell_stack, 1]
            current_cell_y = cell_stack[index_cell_stack, 2]
            index_cell_stack -= 1
            continue
        end

        # Choose a random neighbouring cell and move to it.
        k = rand(1:number_neighbours)
        next_cell_x = neighbours[k, 1]
        next_cell_y = neighbours[k, 2]

        # Store chosen random value to enable unit test for the create_maze function
        environments.maze_randoms[nv, blockID] = k

        # Knock down the wall between current_cell and next_cell
        knock_down_wall(blockID, current_cell_x, current_cell_y, next_cell_x, next_cell_y, environments)

        # Push current cell to cell stack
        index_cell_stack += 1
        cell_stack[index_cell_stack, 1] = current_cell_x
        cell_stack[index_cell_stack, 2] = current_cell_y

        current_cell_x = next_cell_x
        current_cell_y = next_cell_y

        nv += 1
    end

end

# Return a list of unvisited neighbours to cell
function find_valid_neighbors(blockID, cell_x, cell_y, neighbours, environments::CollectPoints)

    number_neighbours = 0

    neighbour_x = 0
    neighbour_y = 0

    for i = 1:4

        # Check West
        if i == 1
            neighbour_x = cell_x - 1
            neighbour_y = cell_y
        end

        # Check East
        if i == 2
            neighbour_x = cell_x + 1
            neighbour_y = cell_y
        end

        # Check South
        if i == 3
            neighbour_x = cell_x
            neighbour_y = cell_y + 1
        end

        # Check North
        if i == 4
            neighbour_x = cell_x
            neighbour_y = cell_y - 1
        end

        if is_valid_neighbor(blockID, neighbour_x, neighbour_y, environments)
            number_neighbours += 1
            neighbours[number_neighbours, 1] = neighbour_x
            neighbours[number_neighbours, 2] = neighbour_y
        end
    end

    return number_neighbours
end


function is_valid_neighbor(blockID, cell_x, cell_y, environments::CollectPoints)

    if (1 ≤ cell_x ≤ environments.maze_columns) && (1 ≤ cell_y ≤ environments.maze_rows)

        if has_all_walls(blockID, cell_x, cell_y, environments)
            return true
        end

    end

    return false
end


function has_all_walls(blockID, cell_x, cell_y, environments::CollectPoints)

    if environments.maze_walls_north[cell_x, cell_y, blockID] == false
        return false
    end

    if environments.maze_walls_south[cell_x, cell_y, blockID] == false
        return false
    end

    if environments.maze_walls_east[cell_x, cell_y, blockID] == false
        return false
    end

    if environments.maze_walls_west[cell_x, cell_y, blockID] == false
        return false
    end

    return true
end

# Knock down the wall between two cells
function knock_down_wall(blockID, cell1_x, cell1_y, cell2_x, cell2_y, environments::CollectPoints)

    # Both cells are in the same row, knock down north and south walls
    if cell1_x == cell2_x

        if cell1_y < cell2_y
            environments.maze_walls_south[cell1_x, cell1_y, blockID] = false
            environments.maze_walls_north[cell2_x, cell2_y, blockID] = false
        else
            environments.maze_walls_north[cell1_x, cell1_y, blockID] = false
            environments.maze_walls_south[cell2_x, cell2_y, blockID] = false
        end
    end

    # Both cells are in the same column, knock down east and west walls
    if cell1_y == cell2_y

        if cell1_x < cell2_x
            environments.maze_walls_east[cell1_x, cell1_y, blockID] = false
            environments.maze_walls_west[cell2_x, cell2_y, blockID] = false
        else
            environments.maze_walls_west[cell1_x, cell1_y, blockID] = false
            environments.maze_walls_east[cell2_x, cell2_y, blockID] = false
        end

    end
end