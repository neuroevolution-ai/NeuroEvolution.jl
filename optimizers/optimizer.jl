

###Imports
using PyCall
using Conda
using JSON

function inititalize_optimizer(individual_size)#arguments: indiviual size // optimizer.config
    scriptdir = @__DIR__
    pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
    optimizer = pyimport("cma_es_deap")
    opt = optimizer.OptimizerCmaEsDeap(individual_size,Dict("A"=>1, "B"=>2))
    return opt
end

function ask(optimizer)
    return optimizer.ask()
end

function tell(optimizer, rewards)
    return optimizer.tell(rewards)
end
#optimizer = inititalize_optimizer()
#print(optimizer.ask())

#opt = optimizer.OptimizerCmaEsDeap(100,config)
#print(opt.ask())
#opt = optimizer.OptimizerCma(100)
#print(opt.ask2())
#creator = pyimport("deap.creator")
#base = pyimport("deap.base")
#cma = pyimport("deap.cma")
#array = pyimport("array")
#np = pyimport("numpy")
###


#cma_es_deap.py implementieren
#function get_optimizer_class(input:: AbstractString)
#    return input
#end






#struct OptimizerCmaEsDeapCfg
#    type :: AbstractString
#    population_size :: Int
#    sigma :: Float32
#end

#struct OptimizerCmaEsDeap2
    #individual_size ::
    #toolbox ::
    #population ::
#end

#=
function __init__()#individual_size ::Int, config::Dict)
    #println(1)
    py"""
    from deap import base
    from deap import creator
    from deap import cma
    import numpy as np

    class OptimizerCma:
        def __init__(self, individual_size):
            self.individual_size = individual_size
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
            self.strategy = cma.Strategy(centroid=[0.0] * self.individual_size, sigma=1.0, lambda_=112) # get lambda and sigma from config
            self.toolbox = base.Toolbox()
            self.toolbox.register("generate", self.strategy.generate, creator.Individual)
            self.toolbox.register("update", self.strategy.update)

            self.population = None


        def ask2(self):
            self.population = self.toolbox.generate()
            genomes = []
            for individual in self.population:
                genomes.append(np.array(individual))  #genomes = [individual for individual in self.population]
            return genomes
        def tell2(self,rewards):
            for ind, fit in zip(self.population, rewards):
                ind.fitness.values = (fit,)
            self.toolbox.update(self.population)
            pass

    def ask3():
        opt = OptimizerCma(100)
        genomes = opt.ask2()
        return genomes
        pass

    def _tell(rewards):
        pass
    #print(ask3())
    #print(2)
    """
end
#config = OptimizerCmaEsDeapCfg("CMA-ES-Deap",112,1.0)

function ask()
    #a = py"_ask"()
    #print(a)
end

=#
