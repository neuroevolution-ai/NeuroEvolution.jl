

###Imports
using PyCall
using Conda
using JSON

function inititalize_optimizer(individual_size,configuration)#arguments: indiviual size // optimizer.config
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    optimizer = pyimport("cma_es_deap")
    opt = optimizer.OptimizerCmaEsDeap(individual_size,configuration)
    return opt
end

function ask(optimizer)
    return optimizer.ask()
end

function tell(optimizer, rewards)
    return optimizer.tell(rewards)
end
