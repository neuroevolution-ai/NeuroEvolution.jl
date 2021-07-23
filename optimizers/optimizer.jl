module Optimizer


###Imports
using PyCall
using Conda
#creator = pyimport("deap.creator")
#base = pyimport("deap.base")
#cma = pyimport("deap.cma")
#array = pyimport("array")
#np = pyimport("numpy")
###


#cma_es_deap.py implementieren
function get_optimizer_class(input:: AbstractString)
    return input
end






struct OptimizerCmaEsDeapCfg
    type :: AbstractString
    population_size :: Int
    sigma :: Float32
end

struct OptimizerCmaEsDeap2
    #individual_size ::
    #toolbox ::
    #population ::
end


function __init__()#individual_size ::Int, config::Dict)

    py"""
    from deap import base
    from deap import creator
    from deap import cma


    individual_size = 100 # vorher bestimmen
    #config = OptimizerCmaEsDeapCfg("CMA-ES-Deap",112,1.0)
    #sigma
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
    strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=1.0, lambda_=112)# config.sigma, config.population_size)

    toolbox = base.Toolbox()
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    population = None
    def _ask(self):
        pass

    def _tell(self,rewards):
        pass
    #opt = OptimizerCmaEsDeap()
    """
end
#config = OptimizerCmaEsDeapCfg("CMA-ES-Deap",112,1.0)

@pydef struct OptimizerCmaEsDeap
    function __init__(self, individual_size)
        self.individual_size = individual_size
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
        strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=1.0, lambda_=112)# config.sigma, config.population_size)

        self.toolbox = base.Toolbox()
        self.toolbox.register("generate", strategy.generate, creator.Individual)
        self.toolbox.register("update", strategy.update)
        self.population = None
    end
end




function ask(toolbox2)
    py"_ask()"

    population = toolbox2.generate()
    genomes = []
    for individual in population
        array = [individual]
        #append!(genomes, np.array(individual))
    end
    return genomes
end


end
