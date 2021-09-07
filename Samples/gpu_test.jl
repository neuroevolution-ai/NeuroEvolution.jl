using CUDA
using Adapt

mutable struct Test_Cfg
    A::CuArray
    B::CuArray
end
function Adapt.adapt_structure(to, test::Test_Cfg)
    A = Adapt.adapt_structure(to, test.A)
    B = Adapt.adapt_structure(to, test.B)
    Test_Cfg(A, B)
end
struct Kernel_test_Cfg
    A::Float32
    B::Int32
end
function Adapt.adapt_structure(to, test::Kernel_test_Cfg)
    A = Adapt.adapt_structure(to, test.A)
    B = Adapt.adapt_structure(to, test.B)
    Kernel_test_Cfg(A, B)
end

function kernel_test(test)

    @cuprintln(test.A)
    @cuprintln(test.B)
    return
end
#test = Kernel_test_Cfg(2.0f0,3)
# @cuda kernel_test(test)





function maze_kernel(maze,offset,maze_columns,maze_rows)
    total_amount_of_cells = maze_columns * maze_rows

    x_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,offset)
    y_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,offset+sizeof(x_coordinate_stack))
    neighbours = @cuDynamicSharedMem(Int32,4,offset+sizeof(x_coordinate_stack)+sizeof(y_coordinate_stack))
    @inbounds fill!(maze,1)
    cell_x_coordinate = 1
    cell_y_coordinate = 1
    amount_of_cells_visited = 1
    cell_stack_index = 1

    while amount_of_cells_visited < total_amount_of_cells

        @inbounds fill!(neighbours,0)

        if cell_y_coordinate-1 >= 1
            for i in 1:4
                if maze[cell_y_coordinate-1,cell_x_coordinate,i] == 0
                    break
                end
                if i == 4
                    neighbours[1] = 1
                end
            end
        end
        if cell_x_coordinate+1 <= maze_columns
            for i in 1:4
                if maze[cell_y_coordinate,cell_x_coordinate+1,i] == 0
                    break
                end
                if i == 4
                    neighbours[2] = 1
                end
            end
        end
        if cell_y_coordinate+1 <= maze_rows
            for i in 1:4
                if maze[cell_y_coordinate+1,cell_x_coordinate,i] == 0
                    break
                end
                if i == 4
                    neighbours[3] = 1
                end
            end
        end
        if cell_x_coordinate-1 >= 1
            for i in 1:4
                if maze[cell_y_coordinate,cell_x_coordinate-1,i] == 0
                    break
                end
                if i == 4
                    neighbours[4] = 1
                end
            end
        end
        if neighbours[1] == 0 && neighbours[2] == 0 && neighbours[3] == 0 && neighbours[4] == 0
            cell_stack_index = cell_stack_index - 1
            cell_x_coordinate = x_coordinate_stack[cell_stack_index]
            cell_y_coordinate = y_coordinate_stack[cell_stack_index]
            continue
        end

        move_x_coordinate = 0
        move_y_coordinate = 0
        rand_index = (abs(rand(Int32)) % 4)
        for i in 1:4
            index = ((rand_index+i) % 4) + 1
            if neighbours[index] == 1
                if index == 1
                    move_y_coordinate = -1
                    break
                end
                if index == 2
                    move_x_coordinate = 1
                    break
                end
                if index == 3
                    move_y_coordinate = 1
                    break
                end
                if index == 4
                    move_x_coordinate = -1
                    break
                end
            end
        end
        if move_y_coordinate == -1 
            maze[cell_y_coordinate,cell_x_coordinate,1] = 0
            maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,3] = 0
        end
        if move_x_coordinate == 1 
            maze[cell_y_coordinate,cell_x_coordinate,2] = 0
            maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,4] = 0
        end
        if move_y_coordinate == 1 
            maze[cell_y_coordinate,cell_x_coordinate,3] = 0
            maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,1] = 0
        end
        if move_x_coordinate == -1 
            maze[cell_y_coordinate,cell_x_coordinate,4] = 0
            maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,2] = 0
        end

        #step5: add origin cell to stack
        x_coordinate_stack[cell_stack_index] = cell_x_coordinate
        y_coordinate_stack[cell_stack_index] = cell_y_coordinate
        cell_stack_index = cell_stack_index +1
        #step6: set coordinates to new cell
        cell_x_coordinate = cell_x_coordinate + move_x_coordinate
        cell_y_coordinate = cell_y_coordinate + move_y_coordinate
        amount_of_cells_visited = amount_of_cells_visited +1
    end
    return
end

function kernel_env_step(maze,action,input,environment_config_array)
    maze_columns = 5
    maze_rows = 5
    maze_cell_size = 80
    agent_radius = 12
    agent_movement_range = 10.0f0
    point_radius = 8
    reward_per_collected_positive_point = 500.00f0
    reward_per_collected_negative_point = -700.00f0

    screen_height = maze_rows * maze_cell_size
    screen_width = maze_columns * maze_cell_size

    agent_x_coordinate = environment_config_array[1] + clamp(floor(action[1] * env_cfg.agent_movement_range),-env_cfg.agent_movement_range,env_cfg.agent_movement_range)
    agent_x_coordinate = min(agent_x_coordinate,screen_width - env_cfg.agent_radius) # Eastern border
    agent_x_coordinate = max(agent_x_coordinate,env_cfg.agent_radius) # Western Border

    agent_y_coordinate = environment_config_array[2] + clamp(floor(action[2] * env_cfg.agent_movement_range),-env_cfg.agent_movement_range,env_cfg.agent_movement_range)
    agent_y_coordinate = min(agent_y_coordinate,screen_height - env_cfg.agent_radius) # Eastern border
    agent_y_coordinate = max(agent_y_coordinate,env_cfg.agent_radius) # Western Border

    # Get cell indizes of agents current position
    cell_x = convert(Int32,ceil(agent_x_coordinate / env_cfg.maze_cell_size))
    overlay_y = convert(Int32,ceil(agent_y_coordinate / env_cfg.maze_cell_size))
    cell_y = maze_rows - (overlay_y-1)

    # Get edge coordinates of current cell
    x_left = env_cfg.maze_cell_size * (cell_x - 1)
    x_right = env_cfg.maze_cell_size * cell_x
    y_bottom = env_cfg.maze_cell_size * (overlay_y - 1)
    y_top = env_cfg.maze_cell_size * overlay_y

    if maze[cell_y,cell_x,1] == 1 #check for Northern Wall
        agent_y_coordinate = min(agent_y_coordinate,y_top - env_cfg.agent_radius)
    end
    if maze[cell_y,cell_x,3] == 1 #check for Southern Wall
        agent_y_coordinate = max(agent_y_coordinate,y_bottom + env_cfg.agent_radius)
    end
    if maze[cell_y,cell_x,2] == 1 #check for Eastern Wall
        agent_x_coordinate = min(agent_x_coordinate,x_right - env_cfg.agent_radius)
    end
    if maze[cell_y,cell_x,4] == 1 #check for Western Wall
        agent_x_coordinate = max(agent_x_coordinate,x_left + env_cfg.agent_radius)
    end
    # Check agent collision with top-left edge (prevents sneaking through the edge)
    if (agent_x_coordinate - x_left < env_cfg.agent_radius) && (y_top - agent_y_coordinate < env_cfg.agent_radius)
        agent_x_coordinate = x_left + env_cfg.agent_radius
        agent_y_coordinate = y_top - env_cfg.agent_radius
    end
    # Check agent collision with top-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < env_cfg.agent_radius) && (y_top - agent_y_coordinate < env_cfg.agent_radius)
        agent_x_coordinate = x_right - env_cfg.agent_radius
        agent_y_coordinate = y_top - env_cfg.agent_radius
    end
    # Check agent collision with bottom-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < env_cfg.agent_radius) && (agent_y_coordinate - y_bottom < env_cfg.agent_radius)
        agent_x_coordinate = x_right - env_cfg.agent_radius
        agent_y_coordinate = y_bottom + env_cfg.agent_radius
    end
    # Check agent collision with bottom-left edge (prevents sneaking through the edge)
    if (agent_x_coordinate - x_left < env_cfg.agent_radius) && (agent_y_coordinate - y_bottom < env_cfg.agent_radius)
        agent_x_coordinate = x_left + env_cfg.agent_radius
        agent_y_coordinate = y_bottom + env_cfg.agent_radius
    end
end

maze_columns = 5
maze_rows = 5
maze = fill(0,(maze_rows,maze_columns,4))

maze_gpu = CuArray(maze)
@cuda shmem=2*maze_columns*maze_rows+4 maze_kernel(maze_gpu,0,maze_columns,maze_rows)
CUDA.synchronize()
Array(maze_gpu)