from deap import base
from deap import creator
from deap import cma
import numpy as np


class OptimizerCmaEsDeap():

    def __init__(self, individual_size: int, configuration: dict):

        self.individual_size = individual_size
        self.population_size = configuration["population_size"]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b',
                       fitness=creator.FitnessMax)

        self.strategy = cma.Strategy(centroid=[
                                     0.0] * individual_size, sigma=configuration["sigma"], lambda_=configuration["population_size"])

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "generate", self.strategy.generate, creator.Individual)
        self.toolbox.register("update", self.strategy.update)

        self.population = None

    def ask(self):
        # Generate a new population
        self.population = self.toolbox.generate()
        genomes = []
        for individual in self.population:
            genomes.append(np.array(individual))

        return genomes, self.population_size, self.individual_size, self.strategy

    def tell(self, rewards):
        for ind, fit in zip(self.population, rewards):
            ind.fitness.values = (fit,)

        # Update the strategy with the evaluated individuals
        self.toolbox.update(self.population)

        return self.strategy
