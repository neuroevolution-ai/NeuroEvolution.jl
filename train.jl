using JSON
using Random
using Statistics

#include("brains/brain.jl")
include("environments/environment.jl")
include("optimizers/optimizer.jl")



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

configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json")

number_generations = configuration["number_generations"]
number_validation_runs = configuration["number_validation_runs"]
number_rounds = configuration["number_rounds"]
maximum_env_seed = configuration["maximum_env_seed"]
environment = configuration["environment"]
brain = configuration["brain"]
optimizer = configuration["optimizer"]

config = TrainingCfg(number_generations, number_validation_runs, number_rounds,maximum_env_seed,environment,brain,optimizer, -1)


environment_class = get_environment_class(config.environment["type"])

#brain_class = get_brain_class(config.brain["type"])

#optimizer_class = get_optimizer_class(config.optimizer["type"])



#instantiate episode runner
#instantiate opimizer

#get start time of training and Date

best_genome_overall = nothing
best_reward_overall = typemin(Int32)


for generation in 1:config.number_generations

    #get start time of generation
env_seed = Random.rand((config.number_validation_runs:config.maximum_env_seed), 1)

    #get genomes from optimizer -> genomes opt.ask()
genomes = [1,2,3,4,5] #temp genomes till optimizer functions
evaluations = [value = [genome, env_seed, config.number_rounds] for genome in genomes]
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
end

#get elapsed time total
#write Results to Simulation_results
