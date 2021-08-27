using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics


include("brains/brain.jl")
include("brains/continuous_time_rnn.jl")
include("environments/environment.jl")
include("optimizers/optimizer.jl")
#include("tools/episode_runner.jl")


#Environment function

function kernel_eval_fitness(individuals,results, env_seed,number_rounds_given)# input)#,individuals,env_seed,number_rounds)
    #Dynamic Memory necessary to be allotted: sizeof(Float32) * (number_neurons * input_size + number_neurons * number_neurons + number_neurons * output_size + input_size + number_neurons + number_neurons + output_size) + sizeof(Int32) * (maze_columns * maze_rows * 4 + 12 + maze_columns * maze_rows + 4)
    #Init Variables
    #####################################################
    delta_t = 0.05f0
    tx = threadIdx().x
    bx = blockIdx().x
    number_neurons = 50
    input_size = 10
    output_size = 2
    v_size = input_size * number_neurons
    w_size = number_neurons * number_neurons
    number_rounds = number_rounds_given[1]
    number_timesteps = 1000
    clipping_range = 1.0f0
    alpha = 0.0f0
    fitness_total = 0
    if threadIdx().x == 1
    Random.seed!(Random.default_rng(),env_seed[1])
    end
    #####################################################
    #offset = 0

    V = @cuDynamicSharedMem(Float32,(number_neurons,input_size))
    W = @cuDynamicSharedMem(Float32,(number_neurons,number_neurons),sizeof(V))
    T = @cuDynamicSharedMem(Float32,(output_size,number_neurons),sizeof(V)+sizeof(W))
    input = @cuDynamicSharedMem(Float32,input_size,sizeof(V)+sizeof(W)+sizeof(T))

  
    brain_initialize(tx,blockIdx().x,V,W,T,individuals)

    sync_threads()

    #x = @cuStaticSharedMem(Float32,number_neurons)
    x = @cuDynamicSharedMem(Float32,number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input))


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
    reward_per_collected_positive_point = 500.0f0 # 500.0f0
    reward_per_collected_negative_point = -700.0f0 # -700.0f0
    maze = @cuDynamicSharedMem(Int32,(5,5,4),sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)) 
    maze_objects_array = @cuDynamicSharedMem(Int32,12,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze))  # Format [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate,cell_x,cell_y,x_left,x_right,y_top,y_bottom]
    x_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(maze_objects_array))
    y_coordinate_stack = @cuDynamicSharedMem(Int32,total_amount_of_cells,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(maze_objects_array)+sizeof(x_coordinate_stack))
    neighbours = @cuDynamicSharedMem(Int32,4,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(maze_objects_array)+sizeof(x_coordinate_stack)+sizeof(y_coordinate_stack))
    ####################################################

    #fitness_total = 0
    sync_threads()
    #Loop through Rounds
    #####################################################
    for j in 1:number_rounds
        fitness_current = 0

        agent_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)

        agent_y_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)

        positive_point_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)

        positive_point_y_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)

        negative_point_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)

        negative_point_y_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
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
            #step1: find all neighboring cells which have not been visited yet
                if  (cell_x_coordinate + 1) <= maze_columns
                    if maze[cell_y_coordinate,cell_x_coordinate+1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate+1,4] == 0
                        neighbours[1] = 1
                        else 
                        neighbours[1] = 0
                    end
                end
                if  (cell_x_coordinate - 1) >= 1
                    if maze[cell_y_coordinate,cell_x_coordinate-1,1] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,2] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,3] == 0 && maze[cell_y_coordinate,cell_x_coordinate-1,4] == 0
                        neighbours[2] = 1
                        else 
                        neighbours[2] = 0
                    end
                end
                if  (cell_y_coordinate + 1) <= maze_rows
                    if maze[cell_y_coordinate+1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate+1,cell_x_coordinate,4] == 0
                        neighbours[3] = 1
                        else 
                        neighbours[3] = 0
                    end
                end
                if  (cell_y_coordinate - 1) >= 1
                    if maze[cell_y_coordinate-1,cell_x_coordinate,1] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,2] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,3] == 0 && maze[cell_y_coordinate-1,cell_x_coordinate,4] == 0
                        neighbours[4] = 1
                        else 
                        neighbours[4] = 0
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
        =#
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
                input[tx] = convert(Float32,agent_x_coordinate / screen_width)
            end
            if tx == 2
                input[tx] = convert(Float32,agent_y_coordinate / screen_height)
            end
            if tx == 3
                input[tx] = convert(Float32,sensor_north / screen_height)
            end
            if tx == 4
                input[tx] = convert(Float32,sensor_east / screen_width)
            end
            if tx == 5
                input[tx] = convert(Float32,sensor_south / screen_height)
            end
            if tx == 6
                input[tx] = convert(Float32,sensor_west / screen_width)
            end
            if tx == 7
                input[tx] = convert(Float32,positive_point_x_coordinate / screen_width)
            end
            if tx == 8
                input[tx] = convert(Float32,positive_point_y_coordinate / screen_height)

            end
            if tx == 9
                input[tx] = convert(Float32,negative_point_x_coordinate / screen_width)

            end
            if tx == 10
                input[tx] = convert(Float32,negative_point_y_coordinate / screen_height)
            end

            sync_threads()


        #Loop through Timesteps
        #################################################
        for index in 1:number_timesteps


            brain_step(tx,temp_V, V, W, T, x, input, action,alpha,delta_t,clipping_range)

            sync_threads()
            #env step()
            #############################################
            if tx == 1
            agent_x_coordinate += clamp(floor(action[1] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
         
            agent_y_coordinate +=  clamp(floor(action[2] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
            

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
            distance = sqrt((positive_point_x_coordinate - agent_x_coordinate) ^ 2 + (positive_point_y_coordinate - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new positive_point randomly in maze
                positive_point_x_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
                positive_point_y_coordinate = convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
                rew = reward_per_collected_positive_point
            end
            # Collect negative point in reach
            distance = sqrt((negative_point_x_coordinate - agent_x_coordinate) ^ 2 + (negative_point_y_coordinate - agent_y_coordinate) ^ 2)
            if distance <= point_radius + agent_radius
                #place new negative_point randomly in maze
                negative_point_x_coordinate =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_columns)) * maze_cell_size)
                negative_point_y_coordinate =  convert(Int32,(abs(rand(Int32)) % (maze_cell_size - (2*agent_radius))) + agent_radius +((abs(rand(Int32)) % maze_rows)) * maze_cell_size)
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

            input[7] = convert(Float32,positive_point_x_coordinate / screen_width)

            input[8] = convert(Float32,positive_point_y_coordinate / screen_height)

            input[9] = convert(Float32,negative_point_x_coordinate / screen_width)

            input[10] = convert(Float32,negative_point_y_coordinate / screen_height)



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
    results[blockIdx().x] = fitness_total / number_rounds
    end
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
    number_inputs = 10
    number_outputs = 2
    number_neurons = brain["number_neurons"]
    number_individuals = optimizer["population_size"]
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(number_inputs,number_outputs,brain,brain_state)

    optimizer = inititalize_optimizer(free_parameters,optimizer)

    #get start time of training and Date

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    for generation in 1:number_generations
        env_seed = Random.rand((number_validation_runs:maximum_env_seed), 1)

        individuals = fill(0.0f0,number_individuals,free_parameters)
        genomes = convert(Array{Array{Float32}},ask(optimizer))
        for i in 1:number_individuals
            for j in 1:free_parameters
                individuals[i,j] = (genomes[i])[j]
            end
        end

        input = CUDA.fill(1.0f0,10)
        rewards_training = CUDA.fill(0f0,number_individuals)
        individuals_gpu = CuArray(individuals) 
        action = CUDA.fill(1.0f0,2)
        fitness_results = CUDA.fill(0f0,112)
        rounds = CUDA.fill(number_rounds,1)
        println("start Generation:",generation)
        @cuda threads=number_neurons blocks=number_individuals shmem=sizeof(Float32)*(number_neurons*(number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 16) kernel_eval_fitness(individuals_gpu,fitness_results,CuArray(env_seed),rounds)
        CUDA.synchronize()
        println("finished Generation:",generation)
        rewards_training = Array(fitness_results)

        tell(optimizer,rewards_training)

        best_genome_current_generation = genomes[(findmax(rewards_training))[2]]


        env_seeds = Array(1:number_validation_runs)
        number_validation_rounds = 1
        rewards_validation = CUDA.fill(0f0,number_validation_runs)
        validation_individuals = fill(0.0f0,number_validation_runs,free_parameters)
        for i in 1:number_validation_runs
            for j in 1:free_parameters
                validation_individuals[i,j] = best_genome_current_generation[j]
            end
        end
        rounds = CUDA.fill(1,1)
        println("started Validation Generation:",generation)
        @cuda threads=number_neurons blocks=number_validation_runs shmem=sizeof(Float32)*(number_neurons*(number_neurons+number_inputs+number_outputs+2) + number_inputs + number_outputs) + sizeof(Int32) * (maze_columns * maze_rows * 6 + 16) kernel_eval_fitness(CuArray(validation_individuals),rewards_validation, CuArray(env_seeds), rounds)
        CUDA.synchronize()
        println("finished Validation Generation:",generation)
        rewards_validation_cpu = Array(rewards_validation)
        display(mean(rewards_validation_cpu))

        #=
        best_reward_current_generation = mean(rewards_validation)
        if best_reward_current_generation > best_reward_overall
            best_genome_overall = best_genome_current_generation
            best_reward_overall = best_reward_current_generation
        end
        =#
    end

end

main()
