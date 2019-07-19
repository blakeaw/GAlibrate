import numpy as np
import pandas as pd
import pyximport; pyximport.install()
from . import run_gao
#from . import run_gao_py as run_gao
#from . import run_gao_numba as run_gao
class GAO(object):
    """continuous Genetic Algorithm Optimizer
    """

    def __init__(self, sampled_parameters, fitness_function, population_size,
                 generations=100, mutation_rate=0.1):
        self.sampled_parameters = sampled_parameters
        self.fitness_function = fitness_function
        self.population_size = int(population_size/4.) * 4
        self.generations = generations
        self.mutation_rate = mutation_rate
        #self.survival_rate = survival_rate
        #self._n_survive = int(survival_rate*population_size)
        self._sp_locs = np.array([sampled_parameter.loc for sampled_parameter in sampled_parameters])
        self._sp_widths = np.array([sampled_parameter.width for sampled_parameter in sampled_parameters])
        self._n_sp = len(sampled_parameters)
        self._pop_idxs = np.array(list(range(self.population_size)))
        return

    def run(self, verbose=False):

        #chromosomes = self._initialization()
        #print(chromosomes)
        #self._chromosomes = chromosomes
        #fitnesses = np.array([self.fitness_function(chromosome) for chromosome in chromosomes])
        #print(fitnesses)
        #self._fitnesses = fitnesses
        #for i in range(self.generations):
        chromosomes = run_gao.run_gao(self.population_size, self._n_sp, self._sp_locs,
                                      self._sp_widths, self.generations, self.mutation_rate,
                                      self.fitness_function)
        fitnesses = np.array([self.fitness_function(chromosome) for chromosome in chromosomes])
        print(chromosomes)
        print(fitnesses)
        print(fitnesses.max())

    #def _initialization(self):
    #    return gao_funcs.random_population(self.population_size, self._n_sp,
    #                                       self._sp_locs, self._sp_widths)

    #def _selection(self):
        #weights = self._fitnesses - self._fitnesses.min()
        #prob = weights/ weights.sum()
        #mating_pairs = np.random.choice(self.population_size, size=[self.population_size/2, 2], p=prob)
        #return mating_pairs
    #    return gao_funcs.choose_mating_pairs(self._fitnesses, self.population_size)

    #def _crossover(self):
    #    pass

    #def _mutation(self):
    #    pass
