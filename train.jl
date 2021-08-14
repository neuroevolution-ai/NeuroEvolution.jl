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
    #index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # V,W,T global intialisieren
    delta_t = 0.05f0
    tx= threadIdx().x
    v_size = 6*50 # input_size * number_neurons
    w_size = 50*50 # number_neurons * number_neurons
    #t_size = 50*2 # number_neurons * output_size
    #V = @cuStaticSharedMem(Float32,(50,6))
    for i in 1:6 #range(1:input_size)
        @inbounds V[tx,i] = individuals[i+((tx-1)*6)]  #tx * input_size
        #@cuprintln("KoordinateV:",tx,";",i,":",V[tx,i])
    end
    sync_threads()

    #W = @cuStaticSharedMem(Float32,(50,50))
    for i in 1:50 #range(1:number_neurons)
        @inbounds W[tx,i] = individuals[v_size+(i+((tx-1)*50))] #tx * number_neurons
        #@cuprintln("KoordinateW:",tx,";",i,":",W[tx,i])
    end
    sync_threads()
    #

    #T = @cuStaticSharedMem(Float32,(50,2))

    for i in 1:2 #range(1:output_size)
        @inbounds T[i,tx] = individuals[v_size+w_size+(tx+((i-1)*50))] #tx * output_size
        #@cuprintln("KoordinateT:",tx,";",i,":",T[tx,i])

    end


    x = @cuStaticSharedMem(Float32,50)
    x[threadIdx().x]= 0.0f0
    sync_threads()
    temp_V = @cuStaticSharedMem(Float32,50)
    temp_W = @cuStaticSharedMem(Float32,50)
    temp_T = @cuStaticSharedMem(Float32,2)
    for index in 1:1000
        #V*input matmul:
        V_value = 0.0f0
            for i = 1:6 
                @inbounds V_value = V[tx, i] * input[i] + V_value
            end
            #@inbounds temp_V[tx] = V_value
            @inbounds temp_V[tx] = tanh(V_value) #find faster option for this step
            #@cuprintln(temp_V[tx])
        sync_threads()

        #W*temp_V matmul:
        W_value = 0.0f0
            for i = 1:50 
                @inbounds W_value = W[tx, i] * temp_V[i] + W_value
            end
            #@inbounds x[tx] = W_value
            @inbounds x[tx] = x[tx] + (delta_t * W_value)
        #@cuprintln(temp_W[tx])
        x[tx] = clamp(x[tx],-1.0f0,1.0f0)
        sync_threads()
        #T*temp_W matmul:
        if tx <= 2
        T_value = 0.0f0
            for i in 1:50
                @inbounds T_value =T[i,tx] * temp_W[i]
            end
            @inbounds temp_T[tx] = T_value
            #@inbounds temp_T[tx] = tanh(T_value)
            #@cuprintln(temp_T[tx])
        end
        sync_threads()
    end
    sync_threads()
    @cuprintln("Index:",tx," Value:",temp_V[tx])
    if tx <= 2
    #@cuprintln("Index:",tx," Value:",temp_T[tx])
    end



    #sync_threads()
    #@cuprintln("Index:",tx," Value:",temp_V[tx])


    #MatMul, needs adaption for each specific multiplication
    #=
    tx = threadIdx().x

        for index in 1:1000

                Cvalue = 0.0f0

                if tx <= m
                    for i = 1:p 
                        Cvalue += A[tx, i] * B[i]
                        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
                    end
                C[tx] = Cvalue
                #@cuprintln(C[tx])
                end

            #step()
            #env()
            #fitness_current += reward
        end

        =#

#=
  tx = threadIdx().x

  for j in 1:1000
    Cvalue = 0.0f0

    if tx <= m
      for i = 1:p 
        Cvalue += A[tx, i] * B[i]
        #@cuprintln("tx $tx, i $i, res: $(A[tx, i] * B[i])")
      end
    C[tx] = Cvalue
      #@cuprintln(C[tx])
    end
  end
  =#

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
    input = CUDA.fill(1.0,6)
    output = CUDA.fill(undef,2)


    for generation in 1:config.number_generations
        #println("Generation:", generation)

        genomes = convert(Array{Array{Float32}},ask(optimizer))
        env_seed = Random.rand((config.number_validation_runs:config.maximum_env_seed), 1)
        evaluations = [value = [genome, env_seed, config.number_rounds] for genome in genomes]
        display(evaluations)


    end

end

#get elapsed time total
#write Results to Simulation_results


#A = CUDA.rand(Float32,100)
#B = CUDA.rand(Float32,50)
#C = similar(B)

#m = 50
#p = 50
#individual = CUDA.rand(Float32,2900,112)
individual = CUDA.rand(Float32,2900)
#display(individual)
V = CUDA.fill(0.0f0,50,6)
W = CUDA.fill(0.0f0,50,50)
T = CUDA.fill(0.0f0,2,50)
input = CUDA.fill(1.0f0,6)
#display(individual)
@cuda threads=50 blocks=1 kernel_eval_fitness(individual,V,W,T,input)#,individuals,1,5)
#CUDA.synchronize()
#main()
#display(T)
#print("Finished")
#main()



#A = CuArray(1:2900)
#A = CUDA.fill(1,50,6)
    



#get start time of generation

    #genomes = ask(optimizer)
 #evaluations = [value = [genome, env_seed, config.number_rounds] for genome in genomes]
#display(evaluations)


    #Array for eval_fitness: [env_class, env_configuration, brain_class, brain_configuration]

    #rewards_training = ep_runner.eval_fitness(evaluations)

    #opt.tell(rewards_training)  --> tell optimizer new rewards

    #best_genome_current_generation = genomes[argmax(rewards_training)]

#Valdiation runs for best genome
#validation_evaluations = [value = [best_genome_current_generation, index, 1] for index in 1:config.number_validation_runs]

    #rewards_validation = ep_runner.eval_fitness(evaluations)

    #best_reward_current_generation = mean(rewards_validation)
    #        if best_reward_current_generation > best_reward_overall
    #        best_genome_overall = best_genome_current_generation
    #        best_reward_overall = best_reward_current_generation 
    #end

    #get  elapsed time of current generation


    #write Log