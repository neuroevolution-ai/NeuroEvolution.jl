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
    V = @cuDynamicSharedMem(Float32,(number_neurons,input_size))
    W = @cuDynamicSharedMem(Float32,(number_neurons,number_neurons),sizeof(V))
    T = @cuDynamicSharedMem(Float32,(output_size,number_neurons),sizeof(V)+sizeof(W))
    input = @cuStaticSharedMem(Float32,input_size)
    
    
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
    x = @cuDynamicSharedMem(Float32,number_neurons,sizeof(V)+sizeof(W)+sizeof(T))
    #x[threadIdx().x]= 0.0f0
    #temp_V = @cuStaticSharedMem(Float32,number_neurons)
    #temp_W = @cuStaticSharedMem(Float32,number_neurons)
    #temp_T = @cuStaticSharedMem(Float32,output_size)

    temp_V = @cuDynamicSharedMem(Float32,number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(x))
    temp_T = @cuDynamicSharedMem(Float32,output_size,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(temp_V)+sizeof(x))
    #Need to be reset at every timestep

    #environment variables
    #####################################################
    maze = @cuStaticSharedMem(Int32,(5,5,4))
    #maze = @cuDynamicSharedMem(Int32,(5,5,4))
    maze_columns = 5
    maze_rows = 5
    maze_cell_size = 80
    screen_width = maze_cell_size * maze_columns
    screen_height = maze_cell_size * maze_rows
    point_radius = 8
    agent_radius = 12
    agent_movement_radius = 10.0f0
    reward_per_collected_positive_point = 500.0f0
    reward_per_collected_negative_point = -700.0f0
    ####################################################

    #fitness_total = 0
    sync_threads()
    #Loop through Rounds
    #####################################################
    for j in 1:number_rounds
        #fitness_current = 0


        #setup Environment
        #################################################
        x_coordinate = 1
        y_coordinate = 1
        total_amount_of_cells = maze_columns*maze_rows
        amount_of_cells_visited = 1
        cell_stack = @cuStaticSharedMem(Int32,total_amount_of_cells)
        neighbours = @cuStaticSharedMem(Int32,4)
        cell_stack_index = 1
        #last_visited_cell_x = 1
        #last_visited_cell_y = 1
        random_state = Random.seed!(1234)
        
        #a123sd = CUDA.rand(random_state,1:5,1)
        #@cuprintln(a123sd[1])
                function has_all_walls(x,y)
                    for i in 1:4
                        if maze[x,y,i] == 0
                            return false
                        end
                    end
                    return true
                end

        while amount_of_cells_visited < total_amount_of_cells
            #step1: find all neighboring cells which have not been visited yet

            if  x_coordinate + 1 <= maze_columns
                if has_all_walls(x_coordinate + 1,y_coordinate)
                    neighbours[1] = 1
                    else 
                    neighbours[1] = 0
                end
            end
            if  x_coordinate - 1 >= 1
                if has_all_walls(x_coordinate - 1,y_coordinate)
                    neighbours[2] = 1
                    else 
                    neighbours[2] = 0
                end
            end
            if  y_coordinate + 1 <= maze_rows
                if has_all_walls(x_coordinate,y_coordinate + 1)
                    neighbours[3] = 1
                    else 
                    neighbours[3] = 0
                end
            end
            if  y_coordinate - 1 >= 1
                if has_all_walls(x_coordinate,y_coordinate - 1)
                    neighbours[4] = 1
                    else 
                    neighbours[4] = 0
                end
            end

            #step1.5: check if a cell has all walls
                
            #########################################

            #step2: if there are no neighbors then backtrack to last visited cell
            #=
            function no_neighbours()
            for i in 1:4
                if neighbours[i] == 1
                    return false
                end
            return true
            end
            end
            =#
            #=
            if tx == 1
                if no_neighbours()
                    cell_stack_index = cell_stack_index -1
                    #x_coordinate = 
                    #y_coordinate = 
                end
            end
            =#
            #step3: choose random neighbor through Random state
                
            #step4: knock down the wall between the cells for both cells

            #step5: add origin cell to stack

            #step6: set coordinates to new cell

           amount_of_cells_visited = amount_of_cells_visited+1
        end
        #####################################################
        
        #Maze created

        #Setup Rest
        ############
        #Place agent randomly in maze
        
        agent_x_coordinate = (abs(rand(Int32)) % maze_columns) + 1
        agent_y_coordinate = (abs(rand(Int32)) % maze_rows) + 1
        positive_point_x_coordinate = (abs(rand(Int32)) % maze_columns) + 1
        positive_point_y_coordinate = (abs(rand(Int32)) % maze_rows) + 1
        negative_point_x_coordinate = (abs(rand(Int32)) % maze_columns) + 1
        negative_point_y_coordinate = (abs(rand(Int32)) % maze_rows) + 1

        #@cuprintln(agent_coordinates[1])

        #positive_point_coordinates = 
        ############
        #################################################
        #environment finished
        
            if tx == 1
            @inbounds input[tx] = convert(Float32,agent_x_coordinate / screen_width)
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

        #Loop through Timesteps
        #################################################
        for index in 1:number_timesteps

            #Brain step()
            #############################################
            #V*input matmul:
            V_value = 0.0f0
            for i = 1:input_size 
                #@inbounds V_value = V[tx, i] * input[i] + V_value
                #@cuprintln(V_value)
            end
            @inbounds temp_V[tx] = tanh(x[tx] + V_value) #find faster option for this step
            sync_threads()
            # @cuprintln(temp_V[tx])
            #W*temp_V matmul:
            W_value = 0.0f0
            for i = 1:number_neurons 
                @inbounds W_value = W[tx, i] * temp_V[i] + W_value
                #@cuprintln(W_value)
            end

            @inbounds x[tx] += (delta_t * ((-alpha * x[tx]) + W_value))
            #@cuprintln(x[tx,bx])
            @inbounds x[tx] = clamp(x[tx],-clipping_range,clipping_range)
            #@cuprintln(x[tx,bx])
            sync_threads()
            #@cuprintln(x[tx])

            #T*temp_W matmul:
            #=
                T_value = 0.0f0
                for i in 1:output_size
                   @inbounds @atomic temp_T[i] = temp_T[i] + T[i,tx] * x[i]

                end
                if tx <= 2
                @inbounds temp_T[tx] = tanh(temp_T[tx])
                end
            =#
            if tx <= output_size
                T_value = 0.0f0
                for i in 1:number_neurons
                   @inbounds T_value = T_value + T[tx,i] * x[i]

                end
                @inbounds temp_T[tx] = tanh(T_value)
            end
            


            #############################################
            #end of Brain step()
            sync_threads()
            #env step()
            #############################################
            #############################################
            #end of env step()
            
            #Move agent
            agent_x_coordinate = agent_x_coordinate + clamp(floor(temp_T[1] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)
            agent_y_coordinate = agent_y_coordinate + clamp(floor(temp_T[2] * agent_movement_radius),-agent_movement_radius,agent_movement_radius)

            # Check agent collisions with outer walls
            agent_y_coordinate = max(agent_y_coordinate,agent_radius) # Upper border
            agent_y_coordinate = max(agent_y_coordinate,screen_height - agent_radius) # Lower bord.
            agent_x_coordinate = max(agent_x_coordinate,screen_width - agent_radius) # Right border
            agent_x_coordinate = max(agent_x_coordinate,agent_radius) # Left border

            # Get cell indizes of agents current position
            cell_x = floor(agent_x_coordinate / maze_cell_size)
            cell_y = floor(agent_y_coordinate / maze_cell_size)

            # Get coordinates of current cell
            x_left = maze_cell_size * cell_x
            x_right = maze_cell_size * (cell_x + 1)
            y_top = maze_cell_size * cell_y
            y_bottom = maze_cell_size * (cell_y + 1)
            #=
            # Check agent collisions with maze walls
            if maze[cell_x,cell_y,1] == 1 #check for Northern Wall
                agent_y_coordinate = max(agent_y_coordinate,y_top + agent_radius)
            end
            if maze[cell_x,cell_y,3] == 1 #check for Southern Wall
                agent_y_coordinate = min(agent_y_coordinate,y_bottom - agent_radius)
            end
            if maze[cell_x,cell_y,2] == 1 #check for Eastern Wall
                agent_x_coordinate = min(agent_x_coordinate,x_right - agent_radius)
            end
            if maze[cell_x,cell_y,4] == 1 #check for Western Wall
                agent_x_coordinate = max(agent_x_coordinate,x_left + agent_radius)
            end
            =#
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
            #@inbounds temp_T[tx] = 0.0f0
            #end
        end

        ####################################################
        #end of Timestep
        sync_threads()
    end
    ######################################################
    #end of Round

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


    number_inputs = 6
    number_outputs = 2
    number_neurons = brain["number_neurons"]
    number_individuals = optimizer["population_size"]
    brain_state = generate_brain_state(number_inputs,number_outputs,brain)
    free_parameters = get_individual_size(number_inputs,number_outputs,brain,brain_state)

    optimizer = inititalize_optimizer(free_parameters,optimizer)
    #display(optimizer)


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
        #evaluations = [value = [genome, env_seed, config.number_rounds] for genome in genomes]

        #V = CUDA.fill(0.0f0,number_neurons,number_inputs,number_individuals)
        #W = CUDA.fill(0.0f0,number_neurons,number_neurons,number_individuals)
        #T = CUDA.fill(0.0f0,number_outputs,number_neurons,number_individuals)
        #x = CUDA.fill(0.0f0,number_neurons,number_individuals)
        input = CUDA.fill(1.0f0,6)
        rewards_training = CUDA.fill(0,number_individuals)
        individuals_gpu = CuArray(individuals)
        @cuda threads=number_neurons blocks=number_individuals shmem=sizeof(Float32)*((number_inputs*(number_neurons+3))+(number_neurons*number_neurons)+((number_outputs+1)*number_neurons)) kernel_eval_fitness(individuals_gpu)#,input)#,input)#,individuals,1,5)
        #@cuda threads=number_neurons blocks=number_individuals kernel_eval_fitness(individuals_gpu,input)
        CUDA.synchronize()
        #display(x)
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
