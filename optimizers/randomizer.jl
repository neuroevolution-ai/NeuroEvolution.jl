using LinearAlgebra
using CUDA
using Distributions
using Random

struct OptimizerRandomCfg
    population_size::Int

    function OptimizerRandomCfg(configuration::OrderedDict)
        new(configuration["population_size"])
    end
end

mutable struct OptimizerRandom
    population_size::Int
    individual_size::Int

    function OptimizerRandom(individual_size::Int, optimizer_configuration::OrderedDict)

        config = OptimizerRandomCfg(optimizer_configuration)
        optimizer = new(config.population_size, individual_size)

        return optimizer

    end

end

function ask(optimizer::OptimizerRandom)

    return rand(optimizer.population_size, optimizer.individual_size)

end

function tell(optimizer::OptimizerRandom, rewards_training)

end