using CUDA
using Adapt
using DataStructures
using Random
using Distributions

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
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
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
        environments.agents_positions[1, blockID], environments.agents_positions[2, blockID] = place_randomly_in_maze(blockID, environments)
        environments.positive_points_positions[1, blockID], environments.positive_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments)
        environments.negative_points_positions[1, blockID], environments.negative_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments)
    end

    return

end

function step(threadID, blockID, action, offset, environments::CollectPoints)
    if threadID == 1
        move_range = environments.agent_movement_range
        maze_width = environments.maze_cell_size * environments.maze_rows
        maze_heigth = environments.maze_cell_size * environments.maze_columns

        
        # Move agent
        environments.agents_positions[1, blockID] += convert(Int, round(clamp(action[1, ] * move_range, -move_range, move_range)))
        environments.agents_positions[2, blockID] += convert(Int, round(clamp(action[2] * move_range, -move_range, move_range)))

        environments.agents_positions[2, blockID] = max(environments.agents_positions[2, blockID], environments.agent_radius)  # Upper border
        environments.agents_positions[2, blockID] = min(environments.agents_positions[2, blockID], maze_heigth - environments.agent_radius)  # Lower bord.
        environments.agents_positions[1, blockID] = min(environments.agents_positions[1, blockID], maze_width - environments.agent_radius)  # Right border
        environments.agents_positions[1, blockID] = max(environments.agents_positions[1, blockID], environments.agent_radius)  # Left border

        #Cell indices of current agent position
        cell_x = convert(Int, ceil(environments.agents_positions[1, blockID] / environments.maze_cell_size))
        cell_y = convert(Int, ceil(environments.agents_positions[2, blockID] / environments.maze_cell_size))

        #walls of current cell
        cell_left, cell_right, cell_top, cell_bottom = get_coordinates_maze_cell(cell_x, cell_y, environments.maze_cell_size)

        # Check maze wall collision
        if environments.maze_walls_north[cell_x, cell_y]
            environments.agents_positions[2, blockID] = max(environments.agents_positions[2, blockID], environments.agent_radius + cell_top)
        end
        if environments.maze_walls_south[cell_x, cell_y]
            environments.agents_positions[2, blockID] = min(environments.agents_positions[2, blockID], cell_bottom - environments.agent_radius)
        end
        if environments.maze_walls_east[cell_x, cell_y]
            environments.agents_positions[1, blockID] = min(environments.agents_positions[1, blockID], cell_right - environments.agent_radius)
        end
        if environments.maze_walls_west[cell_x, cell_y]
            environments.agents_positions[1, blockID] = max(environments.agents_positions[1, blockID], cell_left + environments.agent_radius)
        end

        reward = 0.0

        # Collect positive point in reach
        distance = sqrt((environments.positive_points_positions[1, blockID] - environments.agents_positions[1, blockID])^2 + (environments.positive_points_positions[2, blockID] - environments.agents_positions[2, blockID])^2)
        if distance <= environments.point_radius + environments.agent_radius
            environments.positive_points_positions[1, blockID], environments.positive_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments)
            reward += environments.reward_per_collected_positive_point
        end

        # Collect negative point in reach
        distance = sqrt((environments.negative_points_positions[1, blockID] - environments.agents_positions[1, blockID])^2 + (environments.negative_points_positions[2, blockID] - environments.agents_positions[2, blockID])^2)
        if distance <= environments.point_radius + environments.agent_radius
            environments.negative_points_positions[1, blockID], environments.negative_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments)
            reward += environments.reward_per_collected_negative_point
        end

        if blockID == 1    
            @cuprintln("reward: $reward")
        end    
    end    
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

function get_sensor_distance(environments::CollectPoints, blockID, direction:: String, cell_x:: Int, cell_y:: Int)
        agent_positon = Array(environments.agents_positions[:, blockID])

        # Get coordinates of current cell
        cell_left, cell_right, cell_top, cell_bottom = self.get_coordinates_maze_cell(cell_x, cell_y)

        sensor_distance = 0

        if direction == "top"
            sensor_distance = agent_positon[2] - cell_top - environments.agent_radius
            wall_to_check = 'N'
            i_step = 0
            j_step = -1
        elseif direction == "bottom"
            sensor_distance = y_bottom - self.agent_position_y - self.config.agent_radius
            wall_to_check = "S"
            i_step = 0
            j_step = 1
        elseif direction == "left"
            sensor_distance = self.agent_position_x - x_left - self.config.agent_radius
            wall_to_check = "W"
            i_step = -1
            j_step = 0
        elseif direction == "right"
            sensor_distance = x_right - self.agent_position_x - self.config.agent_radius
            wall_to_check = "E"
            i_step = 1
            j_step = 0
        end    
        i = 0
        j = 0

    end 


function render(environments::CollectPoints)

    # Draw white background
    background("white")

    # Draw agent
    sethue("green")
    agent_position = Array(environments.agents_positions[:, 1])
    circle(agent_position[1, 1], agent_position[2, 1], environments.agent_radius, :fill)

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
    sleep(0.05)
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

function place_randomly_in_maze(blockID, environments::CollectPoints)

    x_position = rand(0:environments.maze_rows) * environments.maze_cell_size
    y_position = rand(0:environments.maze_columns) * environments.maze_cell_size

    x_position += rand(environments.agent_radius:environments.maze_cell_size-environments.agent_radius)
    y_position += rand(environments.agent_radius:environments.maze_cell_size-environments.agent_radius)

    return x_position, y_position
end
