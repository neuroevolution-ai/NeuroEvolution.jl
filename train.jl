using JSON
using Random
using Statistics
using CUDA

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
function all_eval_fitness_kernel(A,B,C,individuals,env_seed,number_rounds)
    #index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # V,W,T global intialisieren


    #for i in 1:6
    #@cuprintln("Block:",blockIdx().x, " Thread:", threadIdx().x," Genomepart:",individuals[blockIdx().x][threadIdx().x])
    #V[threadIdx().x] = individuals[threadIdx().x]
    #V[threadIdx().x][i][blockIdx().x] = individuals[blockIdx][i*threadIdx]

    #end
    #=
    for i in 1:50
    W[threadIdx().x][i][blockIdx().x] = individuals[blockIdx][i*threadIdx]

    end
    for i in 1:2
    T[threadIdx().x][i][blockIdx().x] = individuals[blockIdx][i*threadIdx]
    end
    x[threadIdx().x][blockIdx().x] = 0.0f0
    sync_threads()
    =#

    tx = threadIdx().x
    m = convert(Int32,50)
    p = convert(Int32,50)

        for index in 1:1000

            for j in 1:2000
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

            #step()
            #env()
            #fitness_current += reward
        end

    #out[blockIdx] = fitness_total / 5#number_rounds
    return
end

function one_eval_fitness_kernel(individual,env_seed,number_rounds)

    return
end


function alloc_mem(size)
    mem = CUDA.@cuStaticSharedMem(Float32,size)
    return mem
end
function get_brain(individual,input_size::Int32,output_size::Int32,number_neurons::Int32)#,v_size,w_size,t_size)

        return ptr_V
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


#get elapsed time total
#write Results to Simulation_results
end


individuals = CUDA.fill(1.0f0,10,10)
display(individuals)
A = CUDA.rand(Int32,50,50)
B = CUDA.rand(Int32,50)
C = similar(B)
#display(individuals)
@device_code_warntype @cuda threads=50 blocks=1 all_eval_fitness_kernel(A,B,C,individuals,1,5)
synchronize()
display(C)
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