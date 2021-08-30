function get_sensor_distance(agent_x_coordinate,agent_y_coordinate,cell_x,cell_y,x_left,x_right,y_bottom,y_top,agent)
    sensor_north =  begin
                            sensor_distance = y_top - agent_y_coordinate - agent_radius
                            direction = 1
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_y + 1) > maze_rows
                                    break
                                end
                                current_cell_y += 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end
    sensor_east = begin
                            sensor_distance = x_right - agent_x_coordinate - agent_radius
                            direction = 2
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_x - 1) < 1
                                    break
                                end
                                current_cell_x -= 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end
            sensor_south = begin
                            sensor_distance = agent_y_coordinate - y_bottom - agent_radius
                            direction = 3
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_y - 1) < 1
                                    break
                                end
                                current_cell_y -= 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end
            sensor_west = begin
                            sensor_distance = agent_x_coordinate - x_left - agent_radius
                            direction = 4
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_x + 1) > maze_columns
                                    break
                                end
                                current_cell_x += 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end

    return sensor_north,sensor_east,sensor_south,sensor_west
end


function env_step(maze,action,input,environment_config_array,agent_movement_radius,maze_cell_size,agent_radius,point_radius,screen_width,screen_height,reward_per_collected_positive_point,reward_per_collected_negative_point,maze_columns,maze_rows)



            agent_x_coordinate = environment_config_array[1] + clamp(floor(action[1] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
         
            agent_y_coordinate = environment_config_array[2] + clamp(floor(action[2] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
            

            # Check agent collisions with outer walls
            agent_y_coordinate = max(agent_y_coordinate,agent_radius) # Upper border
            agent_y_coordinate = min(agent_y_coordinate,screen_height - agent_radius) # Lower bord.
            agent_x_coordinate = min(agent_x_coordinate,screen_width - agent_radius) # Right border
            agent_x_coordinate = max(agent_x_coordinate,agent_radius) # Left border

            # Get cell indizes of agents current position
            cell_x = convert(Int32,ceil(agent_x_coordinate / maze_cell_size))
            cell_y = convert(Int32,ceil(agent_y_coordinate / maze_cell_size))

            
            # Get coordinates of current cell
            x_left = maze_cell_size * (cell_x - 1)
            x_right = maze_cell_size * cell_x
            y_bottom = maze_cell_size * (cell_y - 1)
            y_top = maze_cell_size * cell_y
            # Check agent collisions with maze walls

            if maze[cell_y,cell_x,1] == 0 #check for Northern Wall
                agent_y_coordinate = min(agent_y_coordinate,y_top - agent_radius)
            end
            if maze[cell_y,cell_x,3] == 0 #check for Southern Wall
                agent_y_coordinate = max(agent_y_coordinate,y_bottom + agent_radius)
            end
            if maze[cell_y,cell_x,2] == 0 #check for Eastern Wall
                agent_x_coordinate = max(agent_x_coordinate,x_left + agent_radius)
            end

            if maze[cell_y,cell_x,4] == 0 #check for Western Wall
                agent_x_coordinate = min(agent_x_coordinate,x_right - agent_radius)
            end
            # Check agent collision with top-left edge (prevents sneaking through the edge)
            if (agent_x_coordinate - x_left < agent_radius) && ( agent_y_coordinate - y_top < agent_radius)
                agent_x_coordinate = x_left + agent_radius
                agent_y_coordinate = y_top + agent_radius
            end

            # Check agent collision with top-right edge (prevents sneaking through the edge)
            if (x_right - agent_x_coordinate < agent_radius) && (agent_y_coordinate - y_top < agent_radius)
                agent_x_coordinate = x_right - agent_radius
                agent_y_coordinate = y_top + agent_radius
            end

            # Check agent collision with bottom-right edge (prevents sneaking through the edge)
            if (x_right - agent_x_coordinate < agent_radius) && (y_bottom - agent_y_coordinate < agent_radius)
                agent_x_coordinate = x_right - agent_radius
                agent_y_coordinate = y_bottom - agent_radius
            end

            # Check agent collision with bottom-left edge (prevents sneaking through the edge)
            if (agent_x_coordinate - x_left < agent_radius) && (y_bottom - agent_y_coordinate < agent_radius)
                agent_x_coordinate = x_left + agent_radius
                agent_y_coordinate = y_bottom + agent_radius
            end
            
            environment_config_array[1] = agent_x_coordinate
            environment_config_array[2] = agent_y_coordinate

            sensor_north =  begin
                            sensor_distance = y_top - agent_y_coordinate - agent_radius
                            direction = 1
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_y + 1) > maze_rows
                                    break
                                end
                                current_cell_y += 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end
            sensor_east = begin
                            sensor_distance = x_right - agent_x_coordinate - agent_radius
                            direction = 2
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_x - 1) < 1
                                    break
                                end
                                current_cell_x -= 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end
            sensor_south = begin
                            sensor_distance = agent_y_coordinate - y_bottom - agent_radius
                            direction = 3
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_y - 1) < 1
                                    break
                                end
                                current_cell_y -= 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end
            sensor_west = begin
                            sensor_distance = agent_x_coordinate - x_left - agent_radius
                            direction = 4
                            current_cell_x = cell_x
                            current_cell_y = cell_y
                            while true  
                                if (current_cell_x + 1) > maze_columns
                                    break
                                end
                                current_cell_x += 1
                                if maze[current_cell_y,current_cell_x,direction] == 0
                                    break
                                else 
                                    sensor_distance += maze_cell_size
                                end
                            end
                            sensor_distance
                            end



            rew = 0.0f0
            # Collect positive point in reach
            distance = sqrt((environment_config_array[3] - agent_x_coordinate) ^ 2 + (environment_config_array[4] - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new positive_point randomly in maze
                environment_config_array[3] = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
                environment_config_array[4] = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
                rew = reward_per_collected_positive_point
            end
            # Collect negative point in reach
            distance = sqrt((environment_config_array[5] - agent_x_coordinate) ^ 2 + (environment_config_array[6] - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new negative_point randomly in maze
                environment_config_array[5] =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
                environment_config_array[6] =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
                rew = reward_per_collected_negative_point
            end



            #get state of environment as Input for Brain
            #############################################
            
            input[1] = convert(Float32,agent_x_coordinate / screen_width)

            input[2] = convert(Float32,agent_y_coordinate / screen_height)

            input[3] = convert(Float32, sensor_north / screen_height)
            input[4] = convert(Float32, sensor_east / screen_width)
            input[5] = convert(Float32, sensor_south / screen_height)
            input[6] = convert(Float32, sensor_west / screen_width)

            input[7] = convert(Float32,environment_config_array[3] / screen_width)

            input[8] = convert(Float32,environment_config_array[4] / screen_height)

            input[9] = convert(Float32,environment_config_array[5] / screen_width)

            input[10] = convert(Float32,environment_config_array[6] / screen_height)



            return rew
end

function create_maze(maze,neighbours,x_coordinate_stack,y_coordinate_stack)
    maze_columns = 5
    maze_rows = 5
    total_amount_of_cells = maze_columns * maze_rows
        cell_x_coordinate = 1
        cell_y_coordinate = 1
        amount_of_cells_visited = 1
        cell_stack_index = 1
            for j in 1:4
                for k in 1:5
                    for l in 1:5
                        @inbounds maze[l,k,j] = convert(Int32,0)
                    end
                end
            end
    
        while amount_of_cells_visited < total_amount_of_cells
            for i in 1:4
                @inbounds neighbours[i] = 0
            end
            #step1: find all neighboring cells which have not been visited yet
                if  (cell_x_coordinate + 1) <= maze_columns
                    if maze[cell_y_coordinate,cell_x_coordinate+1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,4] == 0
                        @inbounds neighbours[1] = 1
                        else 
                        @inbounds neighbours[1] = 0
                    end
                end
                if  (cell_x_coordinate - 1) >= 1
                    if maze[cell_y_coordinate,cell_x_coordinate-1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,4] == 0
                        @inbounds neighbours[2] = 1
                        else 
                        @inbounds neighbours[2] = 0
                    end
                end
                if  (cell_y_coordinate + 1) <= maze_rows
                    if maze[cell_y_coordinate+1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,4] == 0
                        @inbounds neighbours[3] = 1
                        else 
                        @inbounds neighbours[3] = 0
                    end
                end
                if  (cell_y_coordinate - 1) >= 1
                    if maze[cell_y_coordinate-1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,4] == 0
                        @inbounds neighbours[4] = 1
                        else 
                        @inbounds neighbours[4] = 0
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
            #step3: choose random neighbor through Random state
            rand_index = (abs(rand(Int32)) % 4) 
            for i in 1:4
                index = ((rand_index+i) % 4) + 1
                if neighbours[index] == 1
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
                maze[cell_y_coordinate,cell_x_coordinate,2] = 1
                maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,4] = 1
            end
            if move_x_coordinate == -1 
                maze[cell_y_coordinate,cell_x_coordinate,4] = 1
                maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,2] = 1
            end
            if move_y_coordinate == 1 
                maze[cell_y_coordinate,cell_x_coordinate,1] = 1
                maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,3] = 1
            end
            if move_y_coordinate == -1 
                maze[cell_y_coordinate,cell_x_coordinate,3] = 1
                maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,1] = 1
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
    
end

