using JSON
using Random
using Statistics
using CUDA
using BenchmarkTools
using StaticArrays
using Adapt
using Random

include("brains/brain.jl")
include("environments/environment.jl")
include("optimizers/optimizer.jl")
#include("tools/episode_runner.jl")

struct TrainingCfg
    number_generations ::Int
    number_validation_runs ::Int
    number_rounds::Int
    maximum_env_seed::Int
    environment ::Dict
    brain::Dict
    optimizer::Dict
    experiment_id::Int
end

# number_blocks = 112; number_threads = number_neurons
function kernel_eval_fitness(individuals,V,W,T,input)#,individuals,env_seed,number_rounds)

    #Init Variables
    #####################################################
    delta_t = 0.05f0
    tx= threadIdx().x
    v_size = 6*50 # input_size * number_neurons
    w_size = 50*50 # number_neurons * number_neurons
    number_rounds = 20
    number_neurons = 50
    input_size = 6
    output_size = 2
    number_timesteps = 1000
    clipping_range = 1.0f0

    #####################################################

    #Fill V,W,T with genome data
    #####################################################
    for i in 1:input_size #range(1:input_size)
        @inbounds V[tx,i,blockIdx().x] = individuals[blockIdx().x,i+((tx-1)*input_size)]  #tx * input_size
        #@cuprintln("KoordinateV:",tx,";",i,";",blockIdx().x,":",V[tx,i,blockIdx().x]," Genom:",individuals[blockIdx().x,i+((tx-1)*input_size)])
    end
    for i in 1:number_neurons #range(1:number_neurons)
        @inbounds W[tx,i,blockIdx().x] = individuals[blockIdx().x,v_size+(i+((tx-1)*number_neurons))] #tx * number_neurons
        #@cuprintln("KoordinateW:",tx,";",i,":",W[tx,i])
    end
    for i in 1:output_size #range(1:output_size)
        @inbounds T[i,tx,blockIdx().x] = individuals[blockIdx().x,v_size+w_size+(tx+((i-1)*number_neurons))] #tx * output_size
        #@cuprintln("KoordinateT:",tx,";",i,":",T[tx,i])
    end
    #####################################################


    x = @cuStaticSharedMem(Float32,number_neurons)
    x[threadIdx().x]= 0.0f0
    temp_V = @cuStaticSharedMem(Float32,number_neurons)
    temp_W = @cuStaticSharedMem(Float32,number_neurons)
    temp_T = @cuStaticSharedMem(Float32,output_size)

    #environment variables
    #####################################################
    maze = @cuStaticSharedMem(Int32,(5,5,4))
    maze_columns = 5
    maze_rows = 5
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
        #observation = @cuStaticSharedMem(Float32,6)
        #fitness_current = 0

        #=
        #setup Environment
        #################################################
        x= 1
        y= 1
        total_amount_of_cells = maze_columns*maze_rows
        amount_of_cells_visited = 1
        cell_stack = @cuStaticSharedMem(Int32,total_amount_of_cells)
        cell_stack_index = 1
        #last_visited_cell_x = 1
        #last_visited_cell_y = 1
        random_state = Random.seed!(1234)
        
        while nv < n
            #step1: find all neighboring cells which have not been visited yet
                function find_neighbours(x,y)

                end
            #step1.5: check if a cell has all walls
                function has_all_walls(x,y)
                    for i in 1:4
                        if maze[x,y,i] == 0
                            return false
                        end
                    end
                    return true
                end
            #########################################

            #step2: if there are no neighbors then backtrack to last visited cell

            #step3: choose random neighbor through Random state

            #step4: knock down the wall between the cells for both cells

            #step5: add origin cell to stack

            #step6: set coordinates to new cell

            nv = nv+1
        end

        #################################################
        #environment finished
        =#

        #Loop through Timesteps
        #################################################
        for index in 1:number_timesteps

            #Brain step()
            #############################################
            #V*input matmul:
            V_value = 0.0f0
            for i = 1:input_size 
                @inbounds V_value = V[tx, i,blockIdx().x] * input[i] + V_value
                #@cuprintln(V_value)
            end
            @inbounds temp_V[tx] = tanh(V_value) #find faster option for this step

            #sync_threads()

            #W*temp_V matmul:
            W_value = 0.0f0
            for i = 1:number_neurons 
                @inbounds W_value = W[tx, i,blockIdx().x] * temp_V[i] + W_value
            end

            @inbounds x[tx] = x[tx] + (delta_t * W_value)
            #@cuprintln(x[tx])
            x[tx] = clamp(x[tx],-clipping_range,clipping_range)

            sync_threads()
            #T*temp_W matmul:
            if tx <= output_size
                T_value = 0.0f0
                for i in 1:output_size
                    @inbounds T_value = T_value + T[i,tx,blockIdx().x] * x[i]
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
            sync_threads()

            #Accumulate fitness
            #############################################
            #fitness_current += reward 
            #############################################
            sync_threads()
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

    config = TrainingCfg(number_generations, number_validation_runs, number_rounds,maximum_env_seed,environment,brain,optimizer, -1)
    b = generate_brain_state(6,2,brain)
    a = get_individual_size(6,2,brain,generate_brain_state(6,2,brain))

    optimizer = inititalize_optimizer(a)

    #get start time of training and Date

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    for generation in 1:config.number_generations
        env_seed = Random.rand((config.number_validation_runs:config.maximum_env_seed), 1)

        individuals = fill(0.0f0,112,2900) # number_individuals, free_parameters
        genomes = convert(Array{Array{Float32}},ask(optimizer))
        for i in 1:(size(genomes,1))
            for j in 1:(size(genomes[1],1))
                individuals[i,j] = (genomes[i])[j]
            end
        end
        #evaluations = [value = [genome, env_seed, config.number_rounds] for genome in genomes]

        V = CUDA.fill(0.0f0,50,6,112)
        W = CUDA.fill(0.0f0,50,50,112)
        T = CUDA.fill(0.0f0,2,50,112)
        x = CUDA.fill(0.0f0,50,112)
        input = CUDA.fill(1.0f0,6)
        rewards_training = CUDA.fill(0,112)
        individuals_gpu = CuArray(individuals)
        @cuda threads=50 blocks=112 kernel_eval_fitness(individuals_gpu,V,W,T,input)#,input)#,individuals,1,5)
        CUDA.synchronize()

        #opt.tell(rewards_training)

        #best_genome_current_generation = genomes[findmax(rewards_training)]

        #rewards_validation = 
        for i in 1:number_validation_runs
            #env_seed = i
            #number_rounds = 1
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
#display(T)
#print("Finished")


#A = CuArray(1:2900)
#A = CUDA.fill(1,50,6)
    

