function create_maze()
end

function env_step(maze)
end


function get_sensor_distance(direction::Int,cell_x::Int,cell_y::Int,cell_size::Int, agent_position_x::Int, agent_position_y::Int, agent_radius::Int)
sensor_distance = 0
direction = 1
current_cell_x = cell_x
current_cell_y = cell_y
x_left = maze_cell_size * (cell_x - 1)
x_right = maze_cell_size * cell_x
y_bottom = maze_cell_size * (cell_y - 1)
y_top = maze_cell_size * cell_y
i_step = 0
j_step = 0
if direction == 1
    sensor_distance = agent_position_y - y_top - agent_radius
    j_step += 1
    elseif direction == 2
        sensor_distance = y_bottom - agent_position_y - agent_radius
while true  
    if (current_cell_y + 1) > maze_rows
        break
    end
    current_cell_y += 1
    if maze[current_cell_y,current_cell_x,1] == 0
        break
    else 
        sensor_distance += maze_cell_size
    end
end
return sensor_distance
end