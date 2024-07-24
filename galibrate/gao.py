import warnings
import numpy as np

# import pandas as pd
from .par_fitness_eval import par_fitness_eval

_run_gao_import = False
run_gao = None


def _set_run_gao_numba():
    try:
        from . import run_gao_numba
        warnings.warn("------Running GAO with numba optimization.------", RuntimeWarning)
        return run_gao_numba
    except:
        return None

def _set_run_gao_cython():
    try:
        import pyximport
        # Added the setup_args with include_dirs for numpy so it pyximport can build
        # the code on Windows machine.
        pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
        from . import run_gao_cython
        warnings.warn(
            "------Running GAO with Cython optimization.------", RuntimeWarning
        )
        return run_gao_cython
    except:
        return None


def _set_run_gao_julia():
    try:
        from . import run_gao_julia
        warnings.warn(
            "------Running GAO with Julia optimization.------", RuntimeWarning
        )
        return run_gao_julia
    except:
        return None


def _set_run_gao_py():
    try:
        from . import run_gao_py
        return run_gao_py
    except:
        return None

print(run_gao, _run_gao_import)
# Try the numba version of run_gao
run_gao = _set_run_gao_numba()

if run_gao is None:
# Numba didn't work, so try the Cython version
    run_gao = _set_run_gao_cython()
if run_gao is None:
# Neither Numba nor Cython worked, so try the Julia version
    run_gao = _set_run_gao_julia()
if run_gao is None:
# None of Numba, Cython, or Julia worked, so fallback to the pure Python version
    run_gao = _set_run_gao_py()

print(run_gao, _run_gao_import)

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

    def __init__(
        self,
        sampled_parameters,
        fitness_function,
        population_size,
        generations=100,
        mutation_rate=0.1,
    ):
        """Initialize the GAO object."""
        self.sampled_parameters = sampled_parameters
        self.fitness_function = fitness_function
        if population_size < 4:
            population_size = 4
        self.population_size = int(population_size / 4.0) * 4
        if self.population_size != population_size:
            warn_string = "--Population size was adjusted from {} to {} to make it a multiple of four--".format(
                population_size, self.population_size
            )
            warnings.warn(warn_string)
        self.generations = generations
        self.mutation_rate = mutation_rate
        # self.survival_rate = survival_rate
        # self._n_survive = int(survival_rate*population_size)
        self._n_sp = len(sampled_parameters)
        self._pop_idxs = np.array(list(range(self.population_size)))
        self._last_generation = None
        self._last_generation_fitnesses = None
        self._fittest_chromosome = None
        self._fittest_fitness = None
        self._best_fitness_per_generation = None
        self._total_generations = 0

        return

    def run(self, verbose=False, nprocs=1):
        """Run the GAO.
        Returns:
            tuple of (numpy.ndarray, float): Tuple containing the
            vector of the parameter values with the highest fitness found
            during the search and the corresponding fitness value:
            (theta, fitness)
        """
        sp_locs = np.array(
            [sampled_parameter.loc for sampled_parameter in self.sampled_parameters]
        )
        sp_widths = np.array(
            [sampled_parameter.width for sampled_parameter in self.sampled_parameters]
        )
        last_gen_chromosomes, best_fitness_per_generation = run_gao.run_gao(
            self.population_size,
            self._n_sp,
            sp_locs,
            sp_widths,
            self.generations,
            self.mutation_rate,
            self.fitness_function,
            nprocs,
        )

        if nprocs > 1:
            fitnesses = par_fitness_eval(
                self.fitness_function, last_gen_chromosomes, 0, nprocs
            )
        else:
            fitnesses = np.array(
                [
                    self.fitness_function(chromosome)
                    for chromosome in last_gen_chromosomes
                ]
            )
        self._last_generation = last_gen_chromosomes
        self._last_generation_fitnesses = fitnesses
        fittest_idx = np.argmax(fitnesses)
        fittest_chromosome = last_gen_chromosomes[fittest_idx]
        fittest_fitness = fitnesses[fittest_idx]
        self._fittest_chromosome = fittest_chromosome
        self._fittest_fitness = fittest_fitness
        best_fitness_per_generation[-1] = fittest_fitness
        self._best_fitness_per_generation = best_fitness_per_generation
        self._total_generations = self.generations
        return fittest_chromosome, fittest_fitness

    def resume(self, generations=None, verbose=False, nprocs=1):
        """Continue the GAO for additional generations.
        Returns:
            tuple of (numpy.ndarray, float): Tuple containing the
            vector of the parameter values with the highest fitness found
            during the search and the corresponding fitness value:
            (theta, fitness)
        """
        if generations is None:
            generations = self.generations
        sp_locs = np.array(
            [sampled_parameter.loc for sampled_parameter in self.sampled_parameters]
        )
        sp_widths = np.array(
            [sampled_parameter.width for sampled_parameter in self.sampled_parameters]
        )
        last_gen_chromosomes, best_fitness_per_generation = run_gao.continue_gao(
            self.population_size,
            self._n_sp,
            self._last_generation,
            self._last_generation_fitnesses,
            sp_locs,
            sp_widths,
            generations,
            self.mutation_rate,
            self.fitness_function,
            nprocs,
        )

        if nprocs > 1:
            fitnesses = par_fitness_eval(
                self.fitness_function, last_gen_chromosomes, 0, nprocs
            )
        else:
            fitnesses = np.array(
                [
                    self.fitness_function(chromosome)
                    for chromosome in last_gen_chromosomes
                ]
            )
        self._last_generation = last_gen_chromosomes
        self._last_generation_fitnesses = fitnesses
        fittest_idx = np.argmax(fitnesses)
        fittest_chromosome = last_gen_chromosomes[fittest_idx]
        fittest_fitness = fitnesses[fittest_idx]
        self._fittest_chromosome = fittest_chromosome
        self._fittest_fitness = fittest_fitness
        best_fitness_per_generation[-1] = fittest_fitness
        self._best_fitness_per_generation = np.concatenate(
            (self._best_fitness_per_generation, best_fitness_per_generation)
        )
        self._total_generations += generations
        return fittest_chromosome, fittest_fitness

    @property
    def best(self):
        return self._fittest_chromosome, self._fittest_fitness

    @property
    def best_fitness_per_generation(self):
        return self._best_fitness_per_generation

    @property
    def total_generations(self):
        return self._total_generations

    @property
    def final_population(self):
        return self._last_generation

    @property
    def final_population_fitness(self):
        return self._last_generation_fitnesses
