import warnings
import numpy as np
#import pandas as pd

_run_gao_import = False
# Try the numba version of run_gao
try:
    from . import run_gao_numba as run_gao
    _run_gao_import = True
    warnings.warn("------Running GAO with numba optimization.------", RuntimeWarning)
except ImportError:
    _run_gao_import = False
# Numba didn't work, so try the Cython version
if not _run_gao_import:
    try:
        import pyximport; pyximport.install(language_level=3)
        from . import run_gao_cython as run_gao
        _run_gao_import = True
        warnings.warn("------Running GAO with Cython optimization.------", RuntimeWarning)
    except ImportError:
        _run_gao_import = False
#Numba nor Cython worked, so fallback to the pure Python version
if not _run_gao_import:
    from . import run_gao_py as run_gao
    _run_gao_import = True

class GAO(object):
    """A continuous Genetic Algorithm-based Optimizer.
    This class an optimizer that uses a continuous genetic algorithm.

    Attributes:
        sampled_parameters (list of :obj:galibrate.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Genetic Algorithm
            optimization run.
        fitness_function (function): The fitness function to use for
            assigning fitness to chromosomes (i.e., parameter vectors) during
            the sampling.
        population_size (int): The size of the Genetic Algorithm active population.
        generations (int, optional): Sets the total number of generations to
            iterate through. Default: 100
        mutation_rate (float, optional): Sets the probability (i.e., on [0:1])
            that genes within new chromosomes (i.e., parameters within new
            candidate paramter vectors) undergo mutation. Default: 0.1)

    References:
        1. Carr, Jenna. "An Introduction to Genetic Algorithms." (2014).
            https://www.whitman.edu/Documents/Academics/Mathematics/2014/carrjk.pdf
    """

    def __init__(self, sampled_parameters, fitness_function, population_size,
                 generations=100, mutation_rate=0.1):
        """Initialize the GAO object.
        """
        self.sampled_parameters = sampled_parameters
        self.fitness_function = fitness_function
        if population_size < 4:
            population_size = 4
        self.population_size = int(population_size/4.) * 4
        self.generations = generations
        self.mutation_rate = mutation_rate
        #self.survival_rate = survival_rate
        #self._n_survive = int(survival_rate*population_size)
        self._n_sp = len(sampled_parameters)
        self._pop_idxs = np.array(list(range(self.population_size)))
        self._last_generation = None
        self._fittest_chromosome = None
        self._fittest_fitness = None

        return

    def run(self, verbose=False):
        """Run the GAO.
        Returns:
            tuple of (numpy.ndarray, float): Tuple containing the
            vector of the parameter values with the highest fitness found
            during the search and the corresponding fitness value:
            (theta, fitness)
        """
        sp_locs = np.array([sampled_parameter.loc for sampled_parameter in self.sampled_parameters])
        sp_widths = np.array([sampled_parameter.width for sampled_parameter in self.sampled_parameters])
        last_gen_chromosomes = run_gao.run_gao(self.population_size, self._n_sp,
                                               sp_locs, sp_widths,
                                               self.generations, self.mutation_rate,
                                               self.fitness_function)
        fitnesses = np.array([self.fitness_function(chromosome) for chromosome in last_gen_chromosomes])
        fittest_idx = np.argmax(fitnesses)
        fittest_chromosome = last_gen_chromosomes[fittest_idx]
        fittest_fitness = fitnesses[fittest_idx]
        self._fittest_chromosome = fittest_chromosome
        self._fittest_fitness = fittest_fitness
        return fittest_chromosome, fittest_fitness
