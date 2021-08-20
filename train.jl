using JSON
using Random
using Statistics
using CUDA
using BenchmarkTools
using StaticArrays
using Adapt

include("brains/brain.jl")
include("environments/environment.jl")
include("optimizers/optimizer.jl")
#include("tools/episode_runner.jl")


#Environment function
function place_randomly_in_maze
end

function kernel_eval_fitness(individuals)# input)#,individuals,env_seed,number_rounds)
    #Dynamic Memory necessary to be allotted: sizeof(Float32) * (number_neurons * input_size + number_neurons * number_neurons + number_neurons * output_size + input_size + number_neurons + number_neurons + output_size) + sizeof(Int32) * (maze_columns * maze_rows * 4 + 12 + maze_columns * maze_rows + 4)
    #Init Variables
    #####################################################
    delta_t = 0.05f0
    tx = threadIdx().x
    bx = blockIdx().x
    v_size = 6*50 # input_size * number_neurons
    w_size = 50*50 # number_neurons * number_neurons
    number_rounds = 20
    number_neurons = 50
    input_size = 6
    output_size = 2
    number_timesteps = 1000
    clipping_range = 1.0f0
    alpha = 0.0f0

    #####################################################
    #=
    V = @cuStaticSharedMem(Float32,(number_neurons,input_size))
    W = @cuStaticSharedMem(Float32,(number_neurons,number_neurons),sizeof(V))
    T = @cuStaticSharedMem(Float32,(output_size,number_neurons),sizeof(V)+sizeof(W))
    =#
    #offset = 0
    V = @cuDynamicSharedMem(Float32,(number_neurons,input_size))
    W = @cuDynamicSharedMem(Float32,(number_neurons,number_neurons),sizeof(V))
    T = @cuDynamicSharedMem(Float32,(output_size,number_neurons),sizeof(V)+sizeof(W))
    input = @cuDynamicSharedMem(Float32,input_size,sizeof(V)+sizeof(W)+sizeof(T))
    if tx <= 6
    @inbounds input[tx] = 1.0f0
    end
    #@cuprintln(T[1])
    
    
    #Fill V,W,T with genome data
    #####################################################
    for i in 1:input_size #range(1:input_size)
        @inbounds V[tx,i] = individuals[blockIdx().x,i+((tx-1)*input_size)]  #tx * input_size
        #@cuprintln("KoordinateV:",tx,";",i,";",blockIdx().x,":",V[tx,i,blockIdx().x]," Genom:",individuals[blockIdx().x,i+((tx-1)*input_size)])
    end
    for i in 1:number_neurons #range(1:number_neurons)
        @inbounds W[tx,i] = individuals[blockIdx().x,v_size+(i+((tx-1)*number_neurons))] #tx * number_neurons
        #@cuprintln("KoordinateW:",tx,";",i,":",W[tx,i])
    end
    for i in 1:output_size #range(1:output_size)
        @inbounds T[i,tx] = individuals[blockIdx().x,v_size+w_size+(tx+((i-1)*number_neurons))] #tx * output_size
        #@cuprintln("KoordinateT:",tx,";",i,":",T[tx,i])
    end
    #####################################################
    sync_threads()

    #x = @cuStaticSharedMem(Float32,number_neurons)
    x = @cuDynamicSharedMem(Float32,number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input))
    x[tx]= 0.0f0
    #temp_V = @cuStaticSharedMem(Float32,number_neurons)
    #temp_W = @cuStaticSharedMem(Float32,number_neurons)
    #action = @cuStaticSharedMem(Float32,output_size)

    temp_V = @cuDynamicSharedMem(Float32,number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(x))
    action = @cuDynamicSharedMem(Float32,output_size,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x))
    #Need to be reset at every timestep

    #environment variables
    #####################################################
    
    maze_columns = 5
    maze_rows = 5
    total_amount_of_cells = maze_columns * maze_rows
    maze_cell_size = 80
    screen_width = maze_cell_size * maze_columns
    screen_height = maze_cell_size * maze_rows
    point_radius = 8
    agent_radius = 12
    agent_movement_radius = 10.0f0
    reward_per_collected_positive_point = 500.0f0
    reward_per_collected_negative_point = -700.0f0
    maze = @cuDynamicSharedMem(Int32,(5,5,4),sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)) 
    maze_objects_array = @cuDynamicSharedMem(Int32,12,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze))  # Format [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate,cell_x,cell_y,x_left,x_right,y_top,y_bottom]
    x_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(maze_objects_array))
    y_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(maze_objects_array)+sizeof(x_coordinate_stack))
    neighbours = @cuDynamicSharedMem(Int32,4,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(maze_objects_array)+sizeof(x_coordinate_stack)+sizeof(y_coordinate_stack))
    #agent_x_coordinate = 200
    #agent_y_coordinate = 200
    #positive_point_x_coordinate = 200
    #positive_point_y_coordinate = 200
    #negative_point_x_coordinate = 200
    #negative_point_y_coordinate = 200
    ####################################################

    #fitness_total = 0
    sync_threads()
    #Loop through Rounds
    #####################################################
    for j in 1:number_rounds
        #fitness_current = 0
        

        #setup Environment
        #################################################
        cell_x_coordinate = 1
        cell_y_coordinate = 1
        amount_of_cells_visited = 1
        cell_stack_index = 1
                if tx == 1
            for j in 1:4
            for k in 1:5
            for l in 1:5
            maze[l,k,j] = convert(Int32,0)
            end
            end
            end
    
    while amount_of_cells_visited < total_amount_of_cells
            for i in 1:4
                neighbours[i] = 0
            end
            #@cuprintln("Iteration:",amount_of_cells_visited," cell_x_coordinate:", cell_x_coordinate," cell_y_coordinate:", cell_y_coordinate )
            #step1: find all neighboring cells which have not been visited yet
                if  (cell_x_coordinate + 1) <= maze_columns
                    if maze[cell_y_coordinate,cell_x_coordinate+1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,4] == 0
                        neighbours[1] = 1
                        else 
                        neighbours[1] = 0
                    end
                end
                #@cuprintln("Field:",1," Value:",neighbours[1])
                if  (cell_x_coordinate - 1) >= 1
                    if maze[cell_y_coordinate,cell_x_coordinate-1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,4] == 0
                        neighbours[2] = 1
                        else 
                        neighbours[2] = 0
                    end
                end
                #@cuprintln("Field:",2," Value:",neighbours[2])
                if  (cell_y_coordinate + 1) <= maze_rows
                    if maze[cell_y_coordinate+1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,4] == 0
                        neighbours[3] = 1
                        else 
                        neighbours[3] = 0
                    end
                end
                #@cuprintln("Field:",3," Value:",neighbours[3])
                if  (cell_y_coordinate - 1) >= 1
                    if maze[cell_y_coordinate-1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,4] == 0
                        neighbours[4] = 1
                        else 
                        neighbours[4] = 0
                    end
                end
                #@cuprintln("Field:",4," Value:",neighbours[4])
                    #@cuprintln("N:",neighbours[3]," E:",neighbours[2]," S:",neighbours[4]," W:",neighbours[1])
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
            #@cuprintln("Iteration:",amount_of_cells_visited," move_x_coordinate:", move_x_coordinate)
            #@cuprintln("Iteration:",amount_of_cells_visited," move_y_coordinate:", move_y_coordinate)
            #step4: knock down the wall between the cells for both cells
            
            if move_x_coordinate == 1 
                maze[cell_y_coordinate,cell_x_coordinate,2] = 1
                maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,4] = 1
                #@cuprintln("W")
            end
            if move_x_coordinate == -1 
                maze[cell_y_coordinate,cell_x_coordinate,4] = 1
                maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,2] = 1
                #@cuprintln("E")
            end
            if move_y_coordinate == 1 
                maze[cell_y_coordinate,cell_x_coordinate,1] = 1
                maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,3] = 1
                #@cuprintln("N")
            end
            if move_y_coordinate == -1 
                maze[cell_y_coordinate,cell_x_coordinate,3] = 1
                maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,1] = 1
                #@cuprintln("S")
            end
            #@cuprintln("changed maze!")
            #step5: add origin cell to stack
            x_coordinate_stack[cell_stack_index] = cell_x_coordinate
            y_coordinate_stack[cell_stack_index] = cell_y_coordinate
            #@cuprintln("Top_of_stack_index:",cell_stack_index," x coordinate:",x_coordinate_stack[cell_stack_index]," y coordinate:",y_coordinate_stack[cell_stack_index])
            cell_stack_index = cell_stack_index +1
            #step6: set coordinates to new cell
            cell_x_coordinate = cell_x_coordinate + move_x_coordinate
            cell_y_coordinate = cell_y_coordinate + move_y_coordinate
            #@cuprintln("Iteration done!")
    amount_of_cells_visited = amount_of_cells_visited +1

    end
    
    end


        #####################################################
#=
        #Maze created

        #Setup Rest
        ############
        #Place agent randomly in maze

    # Format [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate,cell_x,cell_y,x_left,x_right,y_top,y_bottom]

        if tx == 1 || tx == 3 || tx == 5 
            @inbounds maze_objects_array[tx] = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns) + 1) * maze_cell_size)
        end
        if tx == 2 || tx == 4 || tx == 6 
            @inbounds maze_objects_array[tx] = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows) + 1) * maze_cell_size)#
        end
        sync_threads()
        #positive_point_coordinates = 
        ############
        #################################################
        #environment finished
            if tx == 1
            input[tx] = convert(Float32,maze_objects_array[tx] / screen_width)
            #@cuprintln(input[tx])
            end
            if tx == 2
            input[tx] = convert(Float32,maze_objects_array[tx] / screen_height)
            #@cuprintln(input[tx])
            end
            #sensor data
            if tx == 3
            input[tx] = convert(Float32,maze_objects_array[tx] / screen_width)
            #@cuprintln(input[tx])
            end
            if tx == 4
            input[tx] = convert(Float32,maze_objects_array[tx] / screen_height)
            #@cuprintln(input[tx])
            end
            if tx == 5
            input[tx] = convert(Float32,maze_objects_array[tx] / screen_width)
            #@cuprintln(input[tx])
            end
            if tx == 6
            input[tx] = convert(Float32,maze_objects_array[tx] / screen_height)
            #@cuprintln(input[tx])
            end
        =#
        #if tx <= 6
        #@inbounds @cuprintln(input[tx])
        #end
        #Loop through Timesteps
        #################################################
        for index in 1:number_timesteps

            #Brain step()
            #############################################
            #V*input matmul:
            V_value = 0.0f0
            if tx == 1 && blockIdx().x == 1
            end
            for i = 1:input_size 
                @inbounds V_value += V[tx, i] * input[i] #+ V_value
                if tx == 1 && blockIdx().x == 1
                #@cuprintln("Thread:",tx," V_Element:",V[tx, i]," * input:",input[i]," + V_Value = ",V_value)
                end
            end

            @inbounds temp_V[tx] = tanh(x[tx] + V_value) #find faster option for this step
            sync_threads()
            #W*temp_V matmul:
            W_value = 0.0f0
            for i = 1:number_neurons 
                @inbounds W_value = W[tx, i] * temp_V[i] + W_value
            end
            @inbounds x[tx] += (delta_t * ((-alpha * x[tx]) + W_value))
            @inbounds x[tx] = clamp(x[tx],-clipping_range,clipping_range)
            sync_threads()


            #T*temp_W matmul:
            #=
                T_value = 0.0f0
                for i in 1:output_size
                   @inbounds @atomic action[i] +=  T[i,tx] * x[i]

                end
                if tx <= 2
                @inbounds action[tx] = tanh(action[tx])
                end
                =#
            
            if tx <= output_size
                T_value = 0.0f0
                for i in 1:number_neurons
                   @inbounds T_value = T_value + T[tx,i] * x[i]
                end
                @inbounds action[tx] = tanh(T_value)
            end
            


            #############################################
            #end of Brain step()
            sync_threads()
            #env step()
            #############################################
            #=

            #Move agent
        
            #@cuprintln("Block: ",blockIdx().x," Davor:",agent_x_coordinate)
        
            agent_x_coordinate = convert(Int32,agent_x_coordinate + clamp(floor(action[1] * agent_movement_radius),-agent_movement_radius,agent_movement_radius))
         
            agent_y_coordinate = agent_y_coordinate + clamp(floor(action[2] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
           
            #@cuprintln("Block: ",blockIdx().x," Danach:",agent_x_coordinate)
            #@cuprintln(floor(action[1] * agent_movement_radius))
            sync_threads()
            # Check agent collisions with outer walls
            agent_y_coordinate = max(agent_y_coordinate,agent_radius) # Upper border
            agent_y_coordinate = min(agent_y_coordinate,screen_height - agent_radius) # Lower bord.
            agent_x_coordinate = min(agent_x_coordinate,screen_width - agent_radius) # Right border
            agent_x_coordinate = max(agent_x_coordinate,agent_radius) # Left border
            #@cuprintln(agent_x_coordinate)
            #@cuprintln(agent_y_coordinate)
            # Get cell indizes of agents current position
            cell_x = convert(Int32,ceil(agent_x_coordinate / maze_cell_size))
            #@cuprintln(cell_x)
            cell_y = convert(Int32,ceil(agent_y_coordinate / maze_cell_size))
            #@cuprintln(cell_y)

            # Get coordinates of current cell
            x_left = maze_cell_size * cell_x
            x_right = maze_cell_size * (cell_x + 1)
            y_top = maze_cell_size * cell_y
            y_bottom = maze_cell_size * (cell_y + 1)
            #@cuprintln(agent_y_coordinate)
            # Check agent collisions with maze walls

            if maze[cell_x,cell_y,1] == 1 #check for Northern Wall
                agent_y_coordinate = max(agent_y_coordinate,y_top + agent_radius)
            end
            #@cuprintln(agent_y_coordinate)
            if maze[cell_x,cell_y,3] == 1 #check for Southern Wall
                agent_y_coordinate = min(agent_y_coordinate,y_bottom - agent_radius)
            end
            if maze[cell_x,cell_y,2] == 1 #check for Eastern Wall
                agent_x_coordinate = min(agent_x_coordinate,x_right - agent_radius)
            end

            if maze[cell_x,cell_y,4] == 1 #check for Western Wall
                agent_x_coordinate = max(agent_x_coordinate,x_left + agent_radius)
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

            #get sensor signals
            #
            #
            #
            #

            rew = 0.0f0

            # Collect positive point in reach
            distance = sqrt((positive_point_x_coordinate - agent_x_coordinate) ^ 2 + (positive_point_y_coordinate - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new positive_point randomly in maze
                #
                #
                rew = reward_per_collected_positive_point
            end
            # Collect negative point in reach
            distance = sqrt((negative_point_x_coordinate - agent_x_coordinate) ^ 2 + (negative_point_y_coordinate - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new negative_point randomly in maze
                #
                #
                rew = reward_per_collected_negative_point
            end


            #end
            sync_threads()
            #get state of environment as Input for Brain
            #############################################
            
            if tx == 1
            input[tx] = convert(Float32,agent_x_coordinate / screen_width)
            end
            if tx == 2
            input[tx] = convert(Float32,agent_y_coordinate / screen_height)
            end
            #sensor data
            if tx == 3
            input[tx] = convert(Float32,positive_point_x_coordinate / screen_width)
            end
            if tx == 4
            input[tx] = convert(Float32,positive_point_y_coordinate / screen_height)
            end
            if tx == 5
            input[tx] = convert(Float32,negative_point_x_coordinate / screen_width)
            end
            if tx == 6
            input[tx] = convert(Float32,negative_point_y_coordinate / screen_height)
            end
            
            #############################################
            sync_threads()

            #Accumulate fitness
            #############################################
            #fitness_current += reward 
            #############################################
            sync_threads()
            #if tx <= output_size
            #@inbounds action[tx] = 0.0f0
            #end


            =#
        end

        ####################################################
        #end of Timestep
        sync_threads()
    end
    ######################################################
    #end of Round

    return
end

function kernel_env_step(action,input)
            tx = threadIdx().x
            #env step()
            #############################################
            if tx == 1
            maze_columns = 5
            maze_rows = 5
            agent_radius = 12
            maze_cell_size = 80
            agent_movement_radius = 10.0f0
            screen_height = maze_cell_size * maze_rows
            screen_width = maze_cell_size * maze_columns
            agent_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
            agent_y_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
            positive_point_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
            positive_point_y_coordinate =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
            negative_point_x_coordinate =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
            negative_point_y_coordinate =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
            reward_per_collected_positive_point = 500.00f0
            reward_per_collected_negative_point = -700.00f0
            @cuprintln("agent_x_coordinate:",agent_x_coordinate)
            @cuprintln("agent_y_coordinate:",agent_y_coordinate)
            #Move agent
        
            #@cuprintln("Block: ",blockIdx().x," Davor:",agent_x_coordinate)
            
            agent_x_coordinate += clamp(floor(action[1] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
         
            agent_y_coordinate +=  clamp(floor(action[2] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
            @cuprintln("agent_x_coordinate:",agent_x_coordinate)
            @cuprintln("agent_y_coordinate:",agent_y_coordinate)
            
            #@cuprintln("Block: ",blockIdx().x," Danach:",agent_x_coordinate)
            #@cuprintln(floor(action[1] * agent_movement_radius))
            #sync_threads()
            # Check agent collisions with outer walls
            agent_y_coordinate = max(agent_y_coordinate,agent_radius) # Upper border
            agent_y_coordinate = min(agent_y_coordinate,screen_height - agent_radius) # Lower bord.
            agent_x_coordinate = min(agent_x_coordinate,screen_width - agent_radius) # Right border
            agent_x_coordinate = max(agent_x_coordinate,agent_radius) # Left border
            @cuprintln("agent_x_coordinate:",agent_x_coordinate)
            @cuprintln("agent_y_coordinate:",agent_y_coordinate)
            # Get cell indizes of agents current position
            #=
            cell_x = convert(Int32,ceil(agent_x_coordinate / maze_cell_size))
            #@cuprintln(cell_x)
            cell_y = convert(Int32,ceil(agent_y_coordinate / maze_cell_size))
            #@cuprintln(cell_y)

            # Get coordinates of current cell
            x_left = maze_cell_size * cell_x
            x_right = maze_cell_size * (cell_x + 1)
            y_top = maze_cell_size * cell_y
            y_bottom = maze_cell_size * (cell_y + 1)
            #@cuprintln(agent_y_coordinate)
            # Check agent collisions with maze walls

            if maze[cell_y,cell_x,1] == 1 #check for Northern Wall
                agent_y_coordinate = max(agent_y_coordinate,y_top + agent_radius)
            end
            #@cuprintln(agent_y_coordinate)
            if maze[cell_y,cell_x,3] == 1 #check for Southern Wall
                agent_y_coordinate = min(agent_y_coordinate,y_bottom - agent_radius)
            end
            if maze[cell_y,cell_x,2] == 1 #check for Eastern Wall
                agent_x_coordinate = min(agent_x_coordinate,x_right - agent_radius)
            end

            if maze[cell_y,cell_x,4] == 1 #check for Western Wall
                agent_x_coordinate = max(agent_x_coordinate,x_left + agent_radius)
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

            #get sensor signals
            #
            #
            #
            #

            rew = 0.0f0

            # Collect positive point in reach
            distance = sqrt((positive_point_x_coordinate - agent_x_coordinate) ^ 2 + (positive_point_y_coordinate - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new positive_point randomly in maze
                #
                #
                rew = reward_per_collected_positive_point
            end
            # Collect negative point in reach
            distance = sqrt((negative_point_x_coordinate - agent_x_coordinate) ^ 2 + (negative_point_y_coordinate - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new negative_point randomly in maze
                #
                #
                rew = reward_per_collected_negative_point
            end


            #end
            #sync_threads()
            #get state of environment as Input for Brain
            #############################################
            
            if tx == 1
            input[tx] = convert(Float32,agent_x_coordinate / screen_width)
            end
            if tx == 2
            input[tx] = convert(Float32,agent_y_coordinate / screen_height)
            end
            #sensor data
            if tx == 3
            input[tx] = convert(Float32,positive_point_x_coordinate / screen_width)
            end
            if tx == 4
            input[tx] = convert(Float32,positive_point_y_coordinate / screen_height)
            end
            if tx == 5
            input[tx] = convert(Float32,negative_point_x_coordinate / screen_width)
            end
            if tx == 6
            input[tx] = convert(Float32,negative_point_y_coordinate / screen_height)
            end
            
            #############################################
            =#
    end
    return
end

function has_all_walls(maze1,x,y)
    for i in 1:4
        if maze1[y,x,i] == 1
            return false
        end
    end
    return true
end
function has_no_neighbours(neighbours)
    for i in 1:4
        if neighbours[i] == 0
            return false
        end
        return true
    end
 end
function kernel_create_maze(maze)
    tx = threadIdx().x
    maze_columns = 5
    maze_rows = 5
    total_amount_of_cells = maze_columns * maze_rows
    maze_cell_size = 80
    screen_width = maze_cell_size * maze_columns
    screen_height = maze_cell_size * maze_rows
    point_radius = 8
    agent_radius = 12
    agent_movement_radius = 10.0f0
    reward_per_collected_positive_point = 500.0f0
    reward_per_collected_negative_point = -700.0f0
    #maze = @cuDynamicSharedMem(Int32,(5,5,4)) 
    maze_objects_array = @cuDynamicSharedMem(Int32,12,sizeof(maze))  # Format [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate,cell_x,cell_y,x_left,x_right,y_top,y_bottom]
    x_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(maze_objects_array))
    y_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(maze_objects_array)+sizeof(x_coordinate_stack))
    neighbours = @cuDynamicSharedMem(Int32,4,sizeof(maze_objects_array)+sizeof(x_coordinate_stack)+sizeof(y_coordinate_stack))
    for i in 1:20
    cell_x_coordinate = 1
    cell_y_coordinate = 1
    amount_of_cells_visited = 1
    cell_stack_index = 1
        if tx == 1
            for j in 1:4
            for k in 1:5
            for l in 1:5
            maze[l,k,j] = convert(Int32,0)
            end
            end
            end
    
    while amount_of_cells_visited < total_amount_of_cells
            for i in 1:4
                neighbours[i] = 0
            end
            #@cuprintln("Iteration:",amount_of_cells_visited," cell_x_coordinate:", cell_x_coordinate," cell_y_coordinate:", cell_y_coordinate )
            #step1: find all neighboring cells which have not been visited yet
                if  (cell_x_coordinate + 1) <= maze_columns
                    if maze[cell_y_coordinate,cell_x_coordinate+1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,4] == 0
                        neighbours[1] = 1
                        else 
                        neighbours[1] = 0
                    end
                end
                #@cuprintln("Field:",1," Value:",neighbours[1])
                if  (cell_x_coordinate - 1) >= 1
                    if maze[cell_y_coordinate,cell_x_coordinate-1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,4] == 0
                        neighbours[2] = 1
                        else 
                        neighbours[2] = 0
                    end
                end
                #@cuprintln("Field:",2," Value:",neighbours[2])
                if  (cell_y_coordinate + 1) <= maze_rows
                    if maze[cell_y_coordinate+1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,4] == 0
                        neighbours[3] = 1
                        else 
                        neighbours[3] = 0
                    end
                end
                #@cuprintln("Field:",3," Value:",neighbours[3])
                if  (cell_y_coordinate - 1) >= 1
                    if maze[cell_y_coordinate-1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,4] == 0
                        neighbours[4] = 1
                        else 
                        neighbours[4] = 0
                    end
                end
                #@cuprintln("Field:",4," Value:",neighbours[4])
                    #@cuprintln("N:",neighbours[3]," E:",neighbours[2]," S:",neighbours[4]," W:",neighbours[1])
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
            #@cuprintln("Iteration:",amount_of_cells_visited," move_x_coordinate:", move_x_coordinate)
            #@cuprintln("Iteration:",amount_of_cells_visited," move_y_coordinate:", move_y_coordinate)
            #step4: knock down the wall between the cells for both cells
            
            if move_x_coordinate == 1 
                maze[cell_y_coordinate,cell_x_coordinate,2] = 1
                maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,4] = 1
                #@cuprintln("W")
            end
            if move_x_coordinate == -1 
                maze[cell_y_coordinate,cell_x_coordinate,4] = 1
                maze[cell_y_coordinate,cell_x_coordinate+move_x_coordinate,2] = 1
                #@cuprintln("E")
            end
            if move_y_coordinate == 1 
                maze[cell_y_coordinate,cell_x_coordinate,1] = 1
                maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,3] = 1
                #@cuprintln("N")
            end
            if move_y_coordinate == -1 
                maze[cell_y_coordinate,cell_x_coordinate,3] = 1
                maze[cell_y_coordinate+move_y_coordinate,cell_x_coordinate,1] = 1
                #@cuprintln("S")
            end
            #@cuprintln("changed maze!")
            #step5: add origin cell to stack
            x_coordinate_stack[cell_stack_index] = cell_x_coordinate
            y_coordinate_stack[cell_stack_index] = cell_y_coordinate
            #@cuprintln("Top_of_stack_index:",cell_stack_index," x coordinate:",x_coordinate_stack[cell_stack_index]," y coordinate:",y_coordinate_stack[cell_stack_index])
            cell_stack_index = cell_stack_index +1
            #step6: set coordinates to new cell
            cell_x_coordinate = cell_x_coordinate + move_x_coordinate
            cell_y_coordinate = cell_y_coordinate + move_y_coordinate
            #@cuprintln("Iteration done!")
    amount_of_cells_visited = amount_of_cells_visited +1

    end
    
    end
    end
    return
end

function kernel_random(env_seed)
    r_state = Random.seed!(env_seed)
    test = CUDA.rand(r_state,Int32)
    @cuprintln(test)
    return
end

function main()
    configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json")

    number_generations = configuration["number_generations"]
    number_validation_runs = configuration["number_validation_runs"]
    number_rounds = configuration["number_rounds"]
    maximum_env_seed = configuration["maximum_env_seed"]
    environment = configuration["environment"]
    brain = configuration["brain"]
    optimizer = configuration["optimizer"]

    maze_columns = 5
    maze_rows = 5
    number_inputs = 6
    number_outputs = 2
    number_neurons = brain["number_neurons"]
    number_individuals = optimizer["population_size"]
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(number_inputs,number_outputs,brain,brain_state)

    optimizer = inititalize_optimizer(free_parameters,optimizer)
    #display(optimizer)

    #@device_code_warntype @cuda threads=10 kernel_random(10)
    #return
    #get start time of training and Date

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    for generation in 1:number_generations
        env_seed = Random.rand((number_validation_runs:maximum_env_seed), 1)

        individuals = fill(0.0f0,number_individuals,free_parameters) # number_individuals, free_parameters
        genomes = convert(Array{Array{Float32}},ask(optimizer))
        for i in 1:number_individuals
            for j in 1:free_parameters
                individuals[i,j] = (genomes[i])[j]
            end
        end

        input = CUDA.fill(1.0f0,6)
        rewards_training = CUDA.fill(0,number_individuals)
        individuals_gpu = CuArray(individuals) 
        action = CUDA.fill(1.0f0,2)
        maze_cpu = fill(convert(Int32,0),(5,5,4))
        #display(maze_cpu)
        maze = CuArray(maze_cpu)

        @cuda threads=1 blocks=1 kernel_env_step(input,action)
        #@cuda threads=number_neurons blocks=number_individuals shmem=sizeof(Float32)*(number_neurons*(number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 16) kernel_eval_fitness(individuals_gpu)#,input)#,input)#,individuals,1,5)
        #@cuda threads=number_neurons blocks=1 shmem=sizeof(Int32) * (maze_columns * maze_rows * 2 + 16) kernel_create_maze(maze)
        CUDA.synchronize()
        #display(maze)
        #opt.tell(rewards_training)

        #best_genome_current_generation = genomes[findmax(rewards_training)]

        #rewards_validation = 
        for i in 1:number_validation_runs
            #env_seed = i
            number_validation_rounds = 1
            #@cuda threads=50 blocks=number_validation_runs kernel_eval_fitness(best_genome_current_generation,V,W,T,input)
        end
        CUDA.synchronize()

        #=
        best_reward_current_generation = mean(rewards_validation)
        if best_reward_current_generation > best_reward_overall
            best_genome_overall = best_genome_current_generation
            best_reward_overall = best_reward_current_generation
        end
        =#
    end

end


#display(individual)
#@cuda threads=50 blocks=1 kernel_eval_fitness(individual,V,W,T,input)#,individuals,1,5)
#CUDA.synchronize()
main()
