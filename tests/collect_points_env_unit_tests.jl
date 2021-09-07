using Test
using Random
include("D:/NeuroEvolution.jl/environments/collect_points_env.jl")


function make_maze_kernel(maze,env_cfg,env_seed)
    if threadIdx().x == 1
        Random.seed!(Random.default_rng(),env_seed)
    end
    create_maze(maze,env_cfg,0)
    return
end

function kernel_env_step_test(maze,agent_position,env_cfg)


    return
end
function env_step_cpu(maze,agent_x_coordinate,agent_y_coordinate,input,action,env_cfg::Collect_Points_Env_Cfg)
    screen_width = env_cfg.maze_cell_size * env_cfg.maze_columns
    screen_height = env_cfg.maze_cell_size * env_cfg.maze_rows
    agent_x_coordinate = agent_x_coordinate + clamp(floor(action[1] * env_cfg.agent_movement_range),-env_cfg.agent_movement_range,env_cfg.agent_movement_range)
    agent_y_coordinate = agent_y_coordinate + clamp(floor(action[2] * env_cfg.agent_movement_range),-env_cfg.agent_movement_range,env_cfg.agent_movement_range)
            
    # Check agent collisions with outer walls
    agent_y_coordinate = max(agent_y_coordinate,env_cfg.agent_radius) # Upper border
    agent_y_coordinate = min(agent_y_coordinate,screen_height - env_cfg.agent_radius) # Lower bord.
    agent_x_coordinate = min(agent_x_coordinate,screen_width - env_cfg.agent_radius) # Right border
    agent_x_coordinate = max(agent_x_coordinate,env_cfg.agent_radius) # Left border

    # Get cell indizes of agents current position
    cell_x = convert(Int32,ceil(agent_x_coordinate / env_cfg.maze_cell_size))
    cell_y = convert(Int32,ceil(agent_y_coordinate / env_cfg.maze_cell_size))

            
    # Get coordinates of current cell
    x_left = env_cfg.maze_cell_size * (cell_x - 1)
    x_right = env_cfg.maze_cell_size * cell_x
    y_bottom = env_cfg.maze_cell_size * (cell_y - 1)
    y_top = env_cfg.maze_cell_size * cell_y
    # Check agent collisions with maze walls

    if maze[cell_y,cell_x,1] == 0 #check for Northern Wall
        agent_y_coordinate = min(agent_y_coordinate,y_top - env_cfg.agent_radius)
    end
    if maze[cell_y,cell_x,3] == 0 #check for Southern Wall
        agent_y_coordinate = max(agent_y_coordinate,y_bottom + env_cfg.agent_radius)
    end
    if maze[cell_y,cell_x,2] == 0 #check for Eastern Wall
        agent_x_coordinate = max(agent_x_coordinate,x_left + env_cfg.agent_radius)
    end
    if maze[cell_y,cell_x,4] == 0 #check for Western Wall
        agent_x_coordinate = min(agent_x_coordinate,x_right - env_cfg.agent_radius)
    end
    # Check agent collision with top-left edge (prevents sneaking through the edge)
    if (agent_x_coordinate - x_left < env_cfg.agent_radius) && ( agent_y_coordinate - y_top < env_cfg.agent_radius)
        agent_x_coordinate = x_left + env_cfg.agent_radius
        agent_y_coordinate = y_top + env_cfg.agent_radius
    end

    # Check agent collision with top-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < env_cfg.agent_radius) && (agent_y_coordinate - y_top < env_cfg.agent_radius)
        agent_x_coordinate = x_right - env_cfg.agent_radius
        agent_y_coordinate = y_top + env_cfg.agent_radius
    end
    # Check agent collision with bottom-right edge (prevents sneaking through the edge)
    if (x_right - agent_x_coordinate < env_cfg.agent_radius) && (y_bottom - agent_y_coordinate < env_cfg.agent_radius)
        agent_x_coordinate = x_right - env_cfg.agent_radius
        agent_y_coordinate = y_bottom - env_cfg.agent_radius
    end
    # Check agent collision with bottom-left edge (prevents sneaking through the edge)
    if (agent_x_coordinate - x_left < env_cfg.agent_radius) && (y_bottom - agent_y_coordinate < env_cfg.agent_radius)
        agent_x_coordinate = x_left + env_cfg.agent_radius
        agent_y_coordinate = y_bottom + env_cfg.agent_radius
    end

end
@testset "Maze" begin
env_seed = 100
maze_columns = 5
maze_rows = 5
maze = CUDA.fill(0,(maze_rows,maze_columns,4))
env_cfg = Collect_Points_Env_Cfg(maze_columns,maze_rows,80,12,8,10.0f0,500.00f0,-700.00f0,1000,10,2)
@cuda shmem=sizeof(Int32) * (maze_columns * maze_rows * 6 + 10) make_maze_kernel(maze,env_cfg,env_seed)
CUDA.synchronize()
maze_cpu = Array(maze)
#test if legitimate maze
display(maze_cpu)
input = [1.0f0,1.0f0]
x_coordinate = 380
y_coordinate = 380
#env_step_cpu(maze_cpu,x_coordinate,y_coordinate,action,env_cfg)

end