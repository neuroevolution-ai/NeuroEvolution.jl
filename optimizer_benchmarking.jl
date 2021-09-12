using JSON
using Random
using CUDA
using BenchmarkTools
using Statistics
using Logging
using Dates
using DataStructures
include("optimizers/optimizer.jl")
include("environments/collect_points_env.jl")
include("brains/continuous_time_rnn.jl")

configuration = JSON.parsefile("configurations/CMA_ES_Deap_CTRNN_Dense.json",dicttype=DataStructures.OrderedDict)




number_generations = configuration["number_generations"]
number_validation_runs = configuration["number_validation_runs"]
number_rounds = configuration["number_rounds"]
maximum_env_seed = configuration["maximum_env_seed"]
environment = configuration["environment"]
brain = configuration["brain"]
optimizer = configuration["optimizer"]
brain_cfg = CTRNN_Cfg(brain["delta_t"],brain["number_neurons"],separated,brain["clipping_range_min"],brain["clipping_range_max"],brain["alpha"])
number_inputs = get_number_inputs()
number_outputs = get_number_outputs()
environment_cfg = Collect_Points_Env_Cfg(environment["maze_columns"],environment["maze_rows"],environment["maze_cell_size"],environment["agent_radius"],environment["point_radius"],environment["agent_movement_range"],environment["reward_per_collected_positive_point"],environment["reward_per_collected_negative_point"],environment["number_time_steps"],number_inputs,number_outputs)
number_individuals = optimizer["population_size"]
brain_state = generate_brain_state(number_inputs,number_outputs,brain)
free_parameters = get_individual_size(brain_state)

optimizer = inititalize_optimizer(free_parameters,optimizer)

for generation in 1:number_generations
    individuals = fill(0.0f0,number_individuals,free_parameters)
    genomes = ask(optimizer)
    for i in 1:number_individuals
        for j in 1:free_parameters
            individuals[i,j] = (genomes[i])[j]
        end
    end
    rewards_training = 100 .* rand(Float32,number_individuals)
    tell(optimizer,rewards_training)
end