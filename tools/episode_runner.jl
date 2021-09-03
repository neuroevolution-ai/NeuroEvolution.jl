####Imports
using CUDA
#include("brains/continuous_time_rnn.jl")


function kernel_eval_fitness(individuals,results, env_seed,number_rounds,brain_cfg::CTRNN_Cfg,environment_cfg::Collect_Points_Env_Cfg)
    #Dynamic Memory necessary to be allotted: sizeof(Float32) * (number_neurons * input_size + number_neurons * number_neurons + number_neurons * output_size + input_size + number_neurons + number_neurons + output_size) + sizeof(Int32) * (maze_columns * maze_rows * 4 + 12 + maze_columns * maze_rows + 4)
    #Init Variables
    #####################################################
    tx = threadIdx().x
    input_size = 10
    output_size = 2
    number_timesteps = 1000
    fitness_total = 0
    if threadIdx().x == 1
    Random.seed!(Random.default_rng(),env_seed[1])
    end
    #####################################################
    #offset = 0

    V = @cuDynamicSharedMem(Float32,(brain_cfg.number_neurons,input_size))
    W = @cuDynamicSharedMem(Float32,(brain_cfg.number_neurons,brain_cfg.number_neurons),sizeof(V))
    T = @cuDynamicSharedMem(Float32,(output_size,brain_cfg.number_neurons),sizeof(V)+sizeof(W))

    input = @cuDynamicSharedMem(Float32,input_size,sizeof(V)+sizeof(W)+sizeof(T))

    sync_threads()
  
    brain_initialize(tx,blockIdx().x, V,W,T,individuals)

    sync_threads()

    x = @cuDynamicSharedMem(Float32,brain_cfg.number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input))


    temp_V = @cuDynamicSharedMem(Float32,brain_cfg.number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(x))
    action = @cuDynamicSharedMem(Float32,output_size,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x))
    #Need to be reset at every timestep

    #environment variables
    #####################################################
    
    maze_columns = environment_cfg.maze_columns
    maze_rows = environment_cfg.maze_rows
    total_amount_of_cells = maze_columns * maze_rows
    maze_cell_size = 80#environment_cfg.maze_cell_size
    screen_width = maze_cell_size * maze_columns
    screen_height = maze_cell_size * maze_rows
    point_radius = 8
    agent_radius = 12
    agent_movement_radius = 10.0f0
    reward_per_collected_positive_point = 500.0f0 # 500.0f0
    reward_per_collected_negative_point = -700.0f0 # -700.0f0
    maze = @cuDynamicSharedMem(Int32,(maze_columns,maze_rows,4),sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)) 
    environment_config_array = @cuDynamicSharedMem(Int32,6,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze))  # Format [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate]
    x_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(environment_config_array))
    y_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(environment_config_array)+sizeof(x_coordinate_stack))
    neighbours = @cuDynamicSharedMem(Int32,4,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(environment_config_array)+sizeof(x_coordinate_stack)+sizeof(y_coordinate_stack))
    ####################################################

    #fitness_total = 0
    sync_threads()
    #Loop through Rounds
    #####################################################
    for j in 1:number_rounds
        #reset brain
        @inbounds x[tx] = 0.0f0
        #
        fitness_current = 0
        
        agent_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)

        agent_y_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)

        #setup Environment
        #################################################
        if tx == 1
            create_maze(maze,neighbours,x_coordinate_stack,y_coordinate_stack)
        end

        #####################################################
        
        #Maze created

        #Setup Rest
        ############
        #Place agent, positive_point, negative_point randomly in maze

        if tx == 1 || tx == 3 || tx == 5 
            @inbounds environment_config_array[tx] = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns) + 1) * maze_cell_size)
        end
        if tx == 2 || tx == 4 || tx == 6 
            @inbounds environment_config_array[tx] = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows) + 1) * maze_cell_size)#
        end
        sync_threads()
        
        ############
            cell_x = convert(Int32,ceil(agent_x_coordinate / maze_cell_size))
            cell_y = convert(Int32,ceil(agent_y_coordinate / maze_cell_size))

            
            # Get coordinates of current cell
            x_left = maze_cell_size * (cell_x - 1)
            x_right = maze_cell_size * cell_x
            y_bottom = maze_cell_size * (cell_y - 1)
            y_top = maze_cell_size * cell_y
        #get first sensor_data:
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

        #################################################
        #environment finished
            if tx == 1
                @inbounds input[tx] = convert(Float32,environment_config_array[tx] / screen_width)
            end
            if tx == 2
                @inbounds input[tx] = convert(Float32,environment_config_array[tx] / screen_height)
            end
            if tx == 3
                @inbounds input[tx] = convert(Float32,environment_config_array[tx] / screen_width)
            end
            if tx == 4
                @inbounds input[tx] = convert(Float32,environment_config_array[tx] / screen_height)
            end
            if tx == 5
                @inbounds input[tx] = convert(Float32,environment_config_array[tx] / screen_width)
            end
            if tx == 6
                @inbounds input[tx] = convert(Float32,environment_config_array[tx] / screen_height)
            end
            if tx == 7
                @inbounds input[tx] = convert(Float32,sensor_north / screen_height)
            end
            if tx == 8
                @inbounds input[tx] = convert(Float32,sensor_east / screen_width)
            end
            if tx == 9
                @inbounds input[tx] = convert(Float32,sensor_south / screen_height)
            end
            if tx == 10
                @inbounds input[tx] = convert(Float32,sensor_west / screen_height)
            end

            sync_threads()


        #Loop through Timesteps
        #################################################
        for index in 1:number_timesteps


            brain_step(tx,temp_V, V, W, T, x, input, action,brain_cfg)

            sync_threads()
            #environment array [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate,reward_per_collected_negative_point,reward_per_collected_positive_point,agent_movement_radius,agent_radius,point_radius,maze_cell_size,maze_columns,maze_rows,screen_height,screen_width]
            if tx == 1
                rew = env_step(maze,action,input,environment_config_array,agent_movement_radius,maze_cell_size,agent_radius,point_radius,screen_width,screen_height,reward_per_collected_positive_point,reward_per_collected_negative_point,maze_columns,maze_rows)
                fitness_current += rew
            
            end

            sync_threads()
        end

        ####################################################
        #end of Timestep
        if tx == 1
        fitness_total += fitness_current
        end
        sync_threads()

    end
    ######################################################
    #end of Round
    if tx == 1
        @inbounds results[blockIdx().x] = fitness_total / number_rounds
    end
    sync_threads()
    return
end


function kernel_eval_fitness_collect_points()


    #init Brain



    fitness_total = 0

    for round in 1:number_of_rounds

        #init_env
        #get first observation
        #reset Brain

        fitness_current = 0

        for step in 1:number_timesteps
            #action = brain_step(observation)
            #observation,rew = env_step()
            #fitness_current += rew
        end
        #fitness_total += fitness_current
    end
    return
end

