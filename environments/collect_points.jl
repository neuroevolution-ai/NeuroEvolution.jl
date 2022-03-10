using CUDA
using Adapt
using DataStructures
using Random
using Distributions
using LinearAlgebra

include("../tools/linear_congruential_generator.jl")
include("../brains/continuous_time_rnn.jl")

struct CollectPoints{A,B,C,D}

    maze_columns::Int64
    maze_rows::Int64
    maze_cell_size::Int32
    agent_radius::Int64
    point_radius::Int32
    agent_movement_range::Float32
    reward_per_collected_positive_point::Float32
    reward_per_collected_negative_point::Float32
    number_time_steps::Int32
    number_sensors::Int32
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
    sensor_distances::C
    ray_directions::D
    ray_cell_distances::D

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
        convert(Int32, configuration["number_sensors"]),
        2,
        convert(Int32, configuration["number_sensors"]) + 6,
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(true, (configuration["maze_rows"], configuration["maze_columns"], number_individuals)),
        CUDA.fill(0, (configuration["maze_columns"] * configuration["maze_rows"], number_individuals)),
        CUDA.fill(0, (2, number_individuals)),
        CUDA.fill(0, (2, number_individuals)),
        CUDA.fill(0, (2, number_individuals)),
        CUDA.fill(0.0f0, (configuration["number_sensors"], number_individuals)),
        CUDA.fill(0.0f0, (2, configuration["number_sensors"], number_individuals)),
        CUDA.fill(0.0f0, (2, configuration["number_sensors"], number_individuals)),
    )
end


Adapt.@adapt_structure CollectPoints

function kernel_eval_fitness(individuals, rewards, environment_seeds, number_rounds, brains, environments::CollectPoints)

    threadID = threadIdx().x
    blockID = blockIdx().x

    offset_shared_memory = 0

    rng_states = @cuDynamicSharedMem(Int64, 1, offset_shared_memory)
    rng_states[1] = environment_seeds[blockID]
    offset_shared_memory += sizeof(rng_states)

    sync_threads()

    observations = @cuDynamicSharedMem(Float32, environments.number_outputs, offset_shared_memory)
    offset_shared_memory += sizeof(observations)

    sync_threads()

    action = @cuDynamicSharedMem(Float32, environments.number_inputs, offset_shared_memory)
    offset_shared_memory += sizeof(action)

    sync_threads()

    # Initialize brains
    initialize(brains, individuals)

    sync_threads()


    #Initialize maze
    initialize(threadID, blockID, environments, offset_shared_memory, rng_states)

    sync_threads()

    offset_shared_memory += sizeof(Int32) * (environments.maze_columns * environments.maze_rows * 2) + sizeof(Int32) * 8

    sync_threads()
    
    reset(brains)

    sync_threads()

    # Get Observation
    if threadID <= environments.number_sensors
        get_observations(threadID, blockID, observations, environments)
    end

    # Iterate over given number of time steps
    for time_step = 1:environments.number_time_steps

        sync_threads()

        # Brain step
        step(brains, observations, action, offset_shared_memory)
        
        sync_threads()

        #Agent step
        step(threadID, blockID, action, observations, rewards, offset_shared_memory, environments, rng_states)

        sync_threads()
    end
end

function initialize(threadID, blockID, environments::CollectPoints, offset, rng_states)

    #Build maze 
    create_maze(threadID, blockID, environments, offset, rng_states)

    if threadID == 1 
        #Randomly place agent and points in maze
        environments.agents_positions[1, blockID], environments.agents_positions[2, blockID] = place_randomly_in_maze(blockID, environments, rng_states)
        environments.positive_points_positions[1, blockID], environments.positive_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments, rng_states)
        environments.negative_points_positions[1, blockID], environments.negative_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments, rng_states)
    end

    #Initialize rays
    angle_diff_per_ray = 360 / environments.number_sensors
    
    if threadID <= environments.number_sensors
        init_ray(threadID, blockID, 1, 0, angle_diff_per_ray, environments)
    end
    sync_threads()

    if threadID <= environments.number_sensors
        calculate_ray_distance(threadID, blockID, environments)
    end


end

function step(threadID, blockID, action, observations, rewards, offset, environments::CollectPoints, rng_states)

    if threadID == 1 
        move_range = environments.agent_movement_range
        maze_width = environments.maze_cell_size * environments.maze_rows
        maze_heigth = environments.maze_cell_size * environments.maze_columns

            
        # Move agent
        environments.agents_positions[1, blockID] += convert(Int, round(clamp(action[1] * move_range, -move_range, move_range)))
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

        # Check collision with edges
        if (environments.agents_positions[1, blockID] - cell_left < environments.agent_radius & environments.agents_positions[2, blockID] - cell_top < environments.agent_radius)
            environments.agents_positions[1, blockID] = cell_left + environments.agent_radius
            environments.agents_positions[2, blockID] = cell_top + environments.agent_radius
        end
        if (cell_right - environments.agents_positions[1, blockID] < environments.agent_radius & environments.agents_positions[2, blockID] - cell_top < environments.agent_radius)
            environments.agents_positions[1, blockID] = cell_right - environments.agent_radius
            environments.agents_positions[2, blockID] = cell_top + environments.agent_radius
        end
        if (cell_right - environments.agents_positions[1, blockID] < environments.agent_radius & cell_bottom - environments.agents_positions[2, blockID] < environments.agent_radius)
            environments.agents_positions[1, blockID] = cell_right - environments.agent_radius
            environments.agents_positions[2, blockID] = cell_bottom - environments.agent_radius
        end
        if (environments.agents_positions[1, blockID] - cell_left < environments.agent_radius & cell_bottom - environments.agents_positions[2, blockID] < environments.agent_radius)
            environments.agents_positions[1, blockID] = cell_left + environments.agent_radius
            environments.agents_positions[2, blockID] = cell_bottom - environments.agent_radius
        end
        
        reward = 0.0
        
        # Collect positive point in reach
        distance = sqrt((environments.positive_points_positions[1, blockID] - environments.agents_positions[1, blockID])^2 + (environments.positive_points_positions[2, blockID] - environments.agents_positions[2, blockID])^2)
        if distance <= environments.point_radius + environments.agent_radius
            environments.positive_points_positions[1, blockID], environments.positive_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments, rng_states)
            reward += environments.reward_per_collected_positive_point
        end

        # Collect negative point in reach
        distance = sqrt((environments.negative_points_positions[1, blockID] - environments.agents_positions[1, blockID])^2 + (environments.negative_points_positions[2, blockID] - environments.agents_positions[2, blockID])^2)
        if distance <= environments.point_radius + environments.agent_radius
            environments.negative_points_positions[1, blockID], environments.negative_points_positions[2, blockID] = place_randomly_in_maze(blockID, environments, rng_states)
            reward += environments.reward_per_collected_negative_point 
        end
        
        rewards[blockID] += reward
    end

    sync_threads()

    if threadID <= environments.number_sensors
        calculate_ray_distance(threadID, blockID, environments)
        get_observations(threadID, blockID, observations, environments)
    end
end

function get_observations(threadID, blockID, observations, environments::CollectPoints)

    maze_width = environments.maze_cell_size * environments.maze_rows
    maze_heigth = environments.maze_cell_size * environments.maze_columns
    normalization = max(maze_width, maze_heigth)

    #Information about Agent-position and Points-position
    if threadID == 1
        observations[1] = environments.agents_positions[1, blockID] / normalization
        observations[2] = environments.agents_positions[2, blockID] / normalization

        observations[3] = environments.positive_points_positions[1, blockID] / normalization
        observations[4] = environments.positive_points_positions[2, blockID] / normalization

        observations[5] = environments.negative_points_positions[1, blockID] / normalization
        observations[6] = environments.negative_points_positions[2, blockID] / normalization
    end    

    #Sensor distance Information
    observations[threadID + 6] = environments.sensor_distances[threadID, blockID] / normalization

end    

function is_valid_cell(x_index, y_index, environments::CollectPoints)

    if 0 <= x_index < environments.maze_rows & 0 <= y_index < environments.maze_columns
        return true
    else
        return false
    end        
end  

function render(environments::CollectPoints, individual)

    # Draw white background
    background("grey100")

    # Draw agent
    sethue("orange")
    agent_position = Array(environments.agents_positions[:, individual])
    circle(agent_position[1], agent_position[2], environments.agent_radius, :fill)

    # Draw positive point
    sethue("green")
    positive_points_position = Array(environments.positive_points_positions[:, individual])
    circle(positive_points_position[1], positive_points_position[2], environments.point_radius, :fill)

    # Draw negative point
    sethue("red4")
    negative_points_position = Array(environments.negative_points_positions[:, individual])
    circle(negative_points_position[1], negative_points_position[2], environments.point_radius, :fill)

    # Draw Sensor lines
    sethue("black")
    setline(1)
    setdash("dot")
    sensor_distances = Array(environments.sensor_distances[:, individual])

    angle_per_ray = 360 / environments.number_sensors
    
    for i = 1:environments.number_sensors
        angle_diff = deg2rad(angle_per_ray * i)
        agent_point = Point(agent_position[1] + environments.agent_radius * cos(angle_diff), agent_position[2] -  environments.agent_radius * sin(angle_diff))
        colission_point = Point(agent_position[1] + environments.agent_radius * cos(angle_diff) + sensor_distances[i] * cos(angle_diff), agent_position[2] -  environments.agent_radius * sin(angle_diff) - sensor_distances[i] * sin(angle_diff))
        if sensor_distances[i] > environments.agent_radius
            line(agent_point, colission_point, :stroke)
        end 
    end            


    # Draw maze
    sethue("black")
    setline(3)
    setdash("solid")
    maze_walls_north = Array(environments.maze_walls_north[:, :, individual])
    maze_walls_east = Array(environments.maze_walls_east[:, :, individual])
    maze_walls_south = Array(environments.maze_walls_south[:, :, individual])
    maze_walls_west = Array(environments.maze_walls_west[:, :, individual])

    for cell_x = 1:environments.maze_columns
        for cell_y = 1:environments.maze_rows

            cell_left, cell_right, cell_top, cell_bottom = get_coordinates_maze_cell(cell_x, cell_y, environments.maze_cell_size)

            # Draw walls
            if maze_walls_north[cell_x, cell_y] == true
                line(Point(cell_left, cell_top), Point(cell_right, cell_top), :stroke)
            end

            if maze_walls_south[cell_x, cell_y] == true
                line(Point(cell_left, cell_bottom), Point(cell_right, cell_bottom), :stroke)
            end

            if maze_walls_east[cell_x, cell_y] == true
                line(Point(cell_right, cell_top), Point(cell_right, cell_bottom), :stroke)
            end

            if maze_walls_west[cell_x, cell_y] == true
                line(Point(cell_left, cell_top), Point(cell_left, cell_bottom), :stroke)
            end

        end
    end

    sleep(0.05)
end

function get_coordinates_maze_cell(cell_x::Int, cell_y::Int, maze_cell_size::Int32)

    cell_left = maze_cell_size * (cell_x - 1)
    cell_right = maze_cell_size * cell_x
    cell_top = maze_cell_size * (cell_y - 1)
    cell_bottom = maze_cell_size * cell_y

    return cell_left, cell_right, cell_top, cell_bottom
end


function create_maze(threadID, blockID, environments::CollectPoints, offset, rng_states)

    # Reset all walls (set all entries to true)
    if threadID <= environments.maze_columns
        for y = 1:environments.maze_rows
            environments.maze_walls_south[threadID, y, blockID] = true
            environments.maze_walls_north[threadID, y, blockID] = true
            environments.maze_walls_east[threadID, y, blockID] = true
            environments.maze_walls_west[threadID, y, blockID] = true
        end
    end

    if threadID == 1
        # Total number of cells.
        total_amount_of_cells = environments.maze_columns * environments.maze_rows

        index_cell_stack = 0
        cell_stack = @cuDynamicSharedMem(Int32, (total_amount_of_cells, 2), offset)
        offset += sizeof(cell_stack)

        neighbours = @cuDynamicSharedMem(Int32, (4, 2), offset)
        offset += sizeof(neighbours)

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
            k = lgc_random(rng_states, 1, 1, number_neighbours)
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
end

# Calculates the distance, the ray has to travel to reach the next cell for each dimension seperatly
# Executed once at Maze initialization
function init_ray(sensor_number, individual, init_x, init_y, angle_diff_per_ray, environments::CollectPoints)

    #calulating the ray dircetion by rotating an initial ray direction in respect to the sensor number
    angle_diff_per_ray = deg2rad(sensor_number * angle_diff_per_ray)
    sine = sin(angle_diff_per_ray)
    cosine = cos(angle_diff_per_ray)

    environments.ray_directions[1, sensor_number, individual] = cosine * init_x + (-sine) * init_y
    environments.ray_directions[2, sensor_number, individual] = sine * init_x + cosine * init_y

    norm = sqrt(environments.ray_directions[1, sensor_number, individual]^2 + environments.ray_directions[2, sensor_number, individual]^2)

    #Normalization (if initial ray is always already normalized this is redundant)
    environments.ray_directions[1, sensor_number, individual] = environments.ray_directions[1, sensor_number, individual] / norm 
    environments.ray_directions[2, sensor_number, individual] = environments.ray_directions[2, sensor_number, individual] / norm  
    
    #If direction is zero for a dimension, the dimension is ignored by setting a high distance, else the delta_distance is assigned
    if -1e-10 < environments.ray_directions[1, sensor_number, individual] < 1e-10  
        environments.ray_directions[1, sensor_number, individual] = 0.0
        environments.ray_cell_distances[1, sensor_number, individual] = Inf32
    else
        environments.ray_cell_distances[1, sensor_number, individual] = sqrt(1 + (environments.ray_directions[2, sensor_number, individual] / environments.ray_directions[1, sensor_number, individual]) * (environments.ray_directions[2, sensor_number, individual] / environments.ray_directions[1, sensor_number, individual]))
    end

    if -1e-10 < environments.ray_directions[2, sensor_number, individual] < 1e-10 
        environments.ray_directions[2, sensor_number, individual] = 0.0
        environments.ray_cell_distances[2, sensor_number, individual] = Inf32
    else 
        environments.ray_cell_distances[2, sensor_number, individual] = sqrt(1 + (environments.ray_directions[1, sensor_number, individual] / environments.ray_directions[2, sensor_number, individual]) * (environments.ray_directions[1, sensor_number, individual] / environments.ray_directions[2, sensor_number, individual]))   
    end
    
    environments.ray_cell_distances[1, sensor_number, individual] *= environments.maze_cell_size 
    environments.ray_cell_distances[2, sensor_number, individual] *= environments.maze_cell_size

end

#Performs the DDA algorithm for a given ray
function calculate_ray_distance(sensor_number, individual, environments::CollectPoints)
    current_cell_x = convert(Int, ceil(environments.agents_positions[1, individual] / environments.maze_cell_size)) 
    current_cell_y = convert(Int, ceil(environments.agents_positions[2, individual] / environments.maze_cell_size))

    cell_left, cell_right, cell_top, cell_bottom = get_coordinates_maze_cell(current_cell_x, current_cell_y, environments.maze_cell_size)

    if environments.ray_directions[1, sensor_number, individual] < 0.0
        step_direction_x = -1
        ray_length_x = (environments.agents_positions[1, individual] - cell_left) / environments.maze_cell_size
        ray_length_x *= environments.ray_cell_distances[1, sensor_number, individual]
        maze_walls_x = environments.maze_walls_west
    else
        step_direction_x = 1
        ray_length_x = 1 - ((environments.agents_positions[1, individual] - cell_left) / environments.maze_cell_size)
        ray_length_x *= environments.ray_cell_distances[1, sensor_number, individual]
        maze_walls_x = environments.maze_walls_east
    end
    
    if environments.ray_directions[2, sensor_number, individual] < 0.0
        step_direction_y = 1
        ray_length_y = 1 - ((environments.agents_positions[2, individual] - cell_top) / environments.maze_cell_size)
        ray_length_y *= environments.ray_cell_distances[2, sensor_number, individual]
        maze_walls_y = environments.maze_walls_south
    else
        step_direction_y = -1
        ray_length_y = (environments.agents_positions[2, individual] - cell_top) / environments.maze_cell_size
        ray_length_y *= environments.ray_cell_distances[2, sensor_number, individual]
        maze_walls_y = environments.maze_walls_north
    end

    #DDA
    hit = false
    current_distance = 0.0
    side = 0

    while !hit
        if ray_length_x < ray_length_y
            current_distance = ray_length_x
            ray_length_x += environments.ray_cell_distances[1, sensor_number, individual]
            side = 1
            maze_walls = maze_walls_x   
        else
            current_distance = ray_length_y 
            ray_length_y += environments.ray_cell_distances[2, sensor_number, individual]
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
    environments.sensor_distances[sensor_number, individual] = current_distance
end

# Returns a list of unvisited neighbours to cell
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

    # Both cells are in the same column, knock down north and south walls
    if cell1_x == cell2_x

        if cell1_y < cell2_y
            environments.maze_walls_south[cell1_x, cell1_y, blockID] = false
            environments.maze_walls_north[cell2_x, cell2_y, blockID] = false
        else
            environments.maze_walls_north[cell1_x, cell1_y, blockID] = false
            environments.maze_walls_south[cell2_x, cell2_y, blockID] = false
        end
    end

    # Both cells are in the same row, knock down east and west walls
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

function place_randomly_in_maze(blockID, environments::CollectPoints, rng_states)

    x_position = lgc_random(rng_states, 1, environments.maze_rows - 1) * environments.maze_cell_size
    x_position += lgc_random(rng_states, 1, environments.agent_radius, (environments.maze_cell_size - environments.agent_radius))

    y_position = lgc_random(rng_states, 1, environments.maze_columns - 1) * environments.maze_cell_size
    y_position += lgc_random(rng_states, 1, environments.agent_radius, (environments.maze_cell_size - environments.agent_radius))

    return x_position, y_position
end

function get_memory_requirements(environments::CollectPoints)
    total_amount_of_cells = environments.maze_columns * environments.maze_rows

    return sizeof(Int32) * (2 * total_amount_of_cells) + sizeof(Int32) * 8 +    #Maze initialization
            sizeof(Float32) * environments.number_inputs +                      #Actions
            sizeof(Float32) * environments.number_outputs +                     #observations
            sizeof(Int64)                                                       #RNG states
end

function get_number_inputs(environments::CollectPoints)
    return environments.number_inputs
end

function get_number_outputs(environments::CollectPoints)
    return environments.number_outputs
end

function get_required_threads(environments::CollectPoints)
    return environments.number_sensors
end
