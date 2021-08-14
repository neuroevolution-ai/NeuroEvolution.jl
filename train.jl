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
    number_rounds = 5
    number_neurons = 50
    input_size = 6
    output_size = 2
    number_timesteps = 1000
    clipping_range = 1.0f0

    #####################################################

    #Fill V,W,T with genome data
    #####################################################
    for i in 1:6 #range(1:input_size)
        @inbounds V[tx,i] = individuals[blockIdx().x,i+((tx-1)*input_size)]  #tx * input_size
        #@cuprintln("KoordinateV:",tx,";",i,":",V[tx,i])
    end
    for i in 1:50 #range(1:number_neurons)
        @inbounds W[tx,i] = individuals[blockIdx().x,v_size+(i+((tx-1)*number_neurons))] #tx * number_neurons
        #@cuprintln("KoordinateW:",tx,";",i,":",W[tx,i])
    end
    for i in 1:2 #range(1:output_size)
        @inbounds T[i,tx] = individuals[blockIdx().x,v_size+w_size+(tx+((i-1)*number_neurons))] #tx * output_size
        #@cuprintln("KoordinateT:",tx,";",i,":",T[tx,i])
    end
    #####################################################
    sync_threads()


    x = @cuStaticSharedMem(Float32,number_neurons)
    x[threadIdx().x]= 0.0f0
    sync_threads()
    temp_V = @cuStaticSharedMem(Float32,number_neurons)
    temp_W = @cuStaticSharedMem(Float32,number_neurons)
    temp_T = @cuStaticSharedMem(Float32,output_size)

    #fitness_total = 0
    #Loop through Rounds
    #####################################################
    for j in 1:number_rounds
        #fitness_current = 0
        #Loop through Timesteps
        #################################################
        for index in 1:number_timesteps

            #Brain step()
            #############################################
            #V*input matmul:
            V_value = 0.0f0
            for i = 1:input_size 
                    @inbounds V_value = V[tx, i] * input[i] + V_value
            end
            #@inbounds temp_V[tx] = V_value
            @inbounds temp_V[tx] = tanh(V_value) #find faster option for this step
            #@cuprintln(temp_V[tx])
            sync_threads()

            #W*temp_V matmul:
            W_value = 0.0f0
            for i = 1:number_neurons 
                @inbounds W_value = W[tx, i] * temp_V[i] + W_value
            end
            #@inbounds x[tx] = W_value
            @inbounds x[tx] = x[tx] + (delta_t * W_value)
            #@cuprintln(temp_W[tx])
            x[tx] = clamp(x[tx],-clipping_range,clipping_range)
            sync_threads()
            #T*temp_W matmul:
            if tx <= output_size
                T_value = 0.0f0
                for i in 1:number_neurons
                    @inbounds T_value =T[i,tx] * temp_W[i]
                end
                @inbounds temp_T[tx] = T_value
                #@inbounds temp_T[tx] = tanh(T_value)
                #@cuprintln(temp_T[tx])
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
    #display(a)

    #environment_class = get_environment_class(config.environment["type"])

    #brain_class = get_brain_class(config.brain["type"])

    #optimizer_class = get_optimizer_class(config.optimizer["type"])

    #display(environment)
    #episode_runner = EpisodeRunner()
    optimizer = inititalize_optimizer(a)

    #get start time of training and Date

    best_genome_overall = nothing
    best_reward_overall = typemin(Int32)

    for generation in 1:config.number_generations

        individuals = fill(0.0f0,112,2900)
        genomes = convert(Array{Array{Float32}},ask(optimizer))
        for i in 1:(size(genomes,1))
            for j in 1:(size(genomes[1],1))
                individuals[i,j] = (genomes[i])[j]
            end
        end
        #env_seed = Random.rand((config.number_validation_runs:config.maximum_env_seed), 1)
        #evaluations = [value = [genome, env_seed, config.number_rounds] for genome in genomes]
        V = CUDA.fill(0.0f0,50,6)
        W = CUDA.fill(0.0f0,50,50)
        T = CUDA.fill(0.0f0,2,50)
        input = CUDA.fill(1.0f0,6)
        individuals_gpu = CuArray(individuals)
        @cuda threads=50 blocks=112 kernel_eval_fitness(individuals_gpu,V,W,T,input)#,input)#,individuals,1,5)
        CUDA.synchronize()
        #display(V)


    end

end


#display(individual)
#@cuda threads=50 blocks=1 kernel_eval_fitness(individual,V,W,T,input)#,individuals,1,5)
#CUDA.synchronize()
main()
#display(T)
#print("Finished")
#main()



#A = CuArray(1:2900)
#A = CUDA.fill(1,50,6)
    

