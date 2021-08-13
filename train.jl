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
function all_eval_fitness_kernel(individual)#,individuals,env_seed,number_rounds)
    #index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # V,W,T global intialisieren
    V = @cuStaticSharedMem(Float32,(6,50))
    tx= threadIdx().x
    for i in 1:6
        @inbounds V[i,tx] = individual[i*tx]
    @cuprintln(V[i,tx])
    end
    #@cuprintln(V[1][1])
    #@cuprintln(V[1][1][1])
    #assign genome values to V Matrix
    #for i in 1:6 #range(1:input_size)
    #V[threadIdx().x] = individuals[threadIdx().x]
    #V[threadIdx().x][i][blockIdx().x] = individuals[blockIdx][i*threadIdx]

    #end
    #

    W = @cuStaticSharedMem(Float32,(50,50,112))
    #assign genome values to W Matrix
    #for i in 1:50 #range(1:number_neurons)
    #W[threadIdx().x] = individuals[threadIdx().x]
    #W[threadIdx().x][i][blockIdx().x] = individuals[blockIdx][i*threadIdx]

    #end

    #

    T = @cuStaticSharedMem(Float32,(50,2,112))
    #assign genome values to T Matrix

    #for i in 1:2 #range(1:output_size)
    #T[threadIdx().x] = individuals[threadIdx().x]
    #T[threadIdx().x][i][blockIdx().x] = individuals[blockIdx][i*threadIdx]

    #end
    #

    x = @cuStaticSharedMem(Float32,(50,1,112))
    #=

    x[threadIdx().x][blockIdx().x] = 0.0f0
    sync_threads()
    =#

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
        #evalFitness(genomes[1])
    
        #start eval_fitness
            for evaluation in evaluations

                individual = evaluation[1]
                number_of_rounds = evaluation[3]
                
                brain_struct = inititalize(6,2,individual,brain)
                x = CUDA.fill(0, 50)
            #fitness_total = 0
    
            #start core routine
                for i in 1:number_of_rounds
                    #build environment
                    #get first output from environment
                    ob = CUDA.rand(Float32,6)#giveu
                    x = CUDA.fill(0, 50) #brain reset()
            
                    for index in 1:2
                        #result,x = step(brain_struct,x,ob)
                        ob = CUDA.rand(Float32,6)#env_step() here
                #done = true
                        if ((generation == 1 && i == 1) || (generation == number_generations && i == number_rounds))
                        println("Generation:",generation," Eval_Index:",index," Round:", i)
                        end            
                        #fitness_current += rew 
                    end
            
                end
                                                                                                            #end core routine
                                                                                                                        #total_result = fitness_total / number_rounds
                                                                                                                        #end eval_fitness
            CUDA.unsafe_free!(brain_struct.V)
            CUDA.unsafe_free!(brain_struct.W)
            CUDA.unsafe_free!(brain_struct.T)
            CUDA.unsafe_free!(brain_struct.x)
        #synchronize()
        #println("Generation:",generation," Genome:")
        #display(individual)

    end
    
end
end

#get elapsed time total
#write Results to Simulation_results


A = CUDA.rand(Float32,100)
#B = CUDA.rand(Float32,50)
#C = similar(B)

#m = 50
#p = 50
individual = CUDA.rand(Float32,300)
display(individual)
@cuda threads=50 blocks=1 all_eval_fitness_kernel(individual)#,individuals,1,5)
CUDA.synchronize()
#print("Finished")
#main()





    



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