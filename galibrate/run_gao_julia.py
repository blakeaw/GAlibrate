from __future__ import print_function
import os
import warnings
import numpy as np
# Import PyJulia and setup
# First, check for a custom Julia runtime 
# executable via the JULIA_RUNTIME environment variable.
JLR = os.environ.get('JULIA_RUNTIME')
if JLR is not None:
    from julia import Julia
    JLR = os.path.abspath(JLR)
    warn_message = "Setting a custom Julia runtime from environment variable JULIA_RUNTIME: Julia(runtime={})".format(JLR)
    warnings.warn(warn_message, RuntimeWarning)
    Julia(runtime=JLR)
# Now import the Julia Main namespace and 
# include the module with the GAO functions.    
from julia import Main
MODPATH = os.path.dirname(os.path.abspath(__file__))
JLPATH = os.path.join(MODPATH, 'gaofunc.jl')
Main.include(JLPATH)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator

from .par_fitness_eval import par_fitness_eval
#from par_fitness_eval import par_fitness_eval

#@numba.jit(nopython=False)
def run_gao(pop_size, n_sp, locs, widths, n_gen,
            mutation_rate, fitness_func, nprocs):

    #@numba.jit('float64(float64[:])', cache=True)
    #def wrap_fitness_func(theta):
    #    return fitness_func(theta)
    # Initialize
    chromosomes = Main.random_population(pop_size, n_sp, locs, widths)
    new_chromosome = np.zeros((pop_size, n_sp))
    best_fitness_per_generation = np.zeros(n_gen+1)
    if nprocs > 1:
        def evaluate_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses, nprocs):
            new_fitnesses = par_fitness_eval(fitness_func, chromosomes, i_n_new, nprocs)
            fitnesses[i_n_new:] = new_fitnesses[:]
            return fitnesses
    else:
        def evaluate_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses, nprocs):
            fitnesses = _compute_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses)
            return fitnesses
    # Begin generating new generations
    for i_gen in tqdm(range(n_gen), desc='Generations: '):
        i_n_new = int(pop_size/2)

        if i_gen == 0:
            fitnesses = np.zeros(pop_size)
            fitnesses = evaluate_fitnesses(fitness_func, chromosomes, pop_size, 0, fitnesses, nprocs)
        else:
            fitnesses = evaluate_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses, nprocs)


        fitnesses_idxs = np.zeros((pop_size, 2), dtype=np.double)
        fitnesses_idxs = Main._fill_fitness_idxs(pop_size, fitnesses, fitnesses_idxs)

        # Selection
        ind = np.argsort(fitnesses_idxs[:,0])
        fitnesses_idxs_sort = fitnesses_idxs[ind]
        best_fitness_per_generation[i_gen] = fitnesses_idxs_sort[-1,0]
        survivors = fitnesses_idxs_sort[int(pop_size/2):]
        # Move over the survivors
        new_chromosome = Main._move_over_survivors(pop_size, survivors, chromosomes, new_chromosome)
        # Choose the mating pairs
        mating_pairs = choose_mating_pairs(survivors, pop_size)
        # Generate children
        new_chromosome = Main._generate_children(pop_size, n_sp, i_n_new, mating_pairs, chromosomes, new_chromosome)
        # Replace the old population with the new one
        chromosomes = new_chromosome.copy()
        # Mutation
        if i_gen < (n_gen-1):
            chromosomes = Main.mutation(chromosomes, locs, widths, pop_size, n_sp, mutation_rate)
        fitnesses = Main._copy_survivor_fitnesses(pop_size, survivors, fitnesses)
    return chromosomes, best_fitness_per_generation

def continue_gao(pop_size, n_sp, chromosomes, fitnesses, locs, widths, n_gen,
            mutation_rate, fitness_func, nprocs):

    # Initialize

    new_chromosome = np.zeros((pop_size, n_sp))
    best_fitness_per_generation = np.zeros(n_gen+1)
    if nprocs > 1:
        def evaluate_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses, nprocs):
            new_fitnesses = par_fitness_eval(fitness_func, chromosomes, i_n_new, nprocs)
            fitnesses[i_n_new:] = new_fitnesses[:]
            return fitnesses
    else:
        def evaluate_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses, nprocs):
            fitnesses = _compute_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses)
            return fitnesses
    # Begin generating new generations
    for i_gen in tqdm(range(n_gen), desc='Generations: '):
        i_n_new = int(pop_size/2)

        if i_gen > 0:
            fitnesses = evaluate_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses, nprocs)

        fitnesses_idxs = np.zeros((pop_size, 2), dtype=np.double)
        fitnesses_idxs = Main._fill_fitness_idxs(pop_size, fitnesses, fitnesses_idxs)

        # Selection
        ind = np.argsort(fitnesses_idxs[:,0])
        fitnesses_idxs_sort = fitnesses_idxs[ind]
        best_fitness_per_generation[i_gen] = fitnesses_idxs_sort[-1,0]
        survivors = fitnesses_idxs_sort[int(pop_size/2):]
        # Move over the survivors
        new_chromosome = Main._move_over_survivors(pop_size, survivors, chromosomes, new_chromosome)
        # Choose the mating pairs
        mating_pairs = choose_mating_pairs(survivors, pop_size)
        # Generate children
        new_chromosome = Main._generate_children(pop_size, n_sp, i_n_new, mating_pairs, chromosomes, new_chromosome)
        # Replace the old population with the new one
        chromosomes = new_chromosome.copy()
        # Mutation
        if i_gen < (n_gen-1):
            chromosomes = Main.mutation(chromosomes, locs, widths, pop_size, n_sp, mutation_rate)
        fitnesses = Main._copy_survivor_fitnesses(pop_size, survivors, fitnesses)
    return chromosomes, best_fitness_per_generation


def _compute_fitnesses(fitness_func, chromosomes, pop_size, start, fitness_array):
    for i in range(start, pop_size):
        fitness_array[i] = fitness_func(chromosomes[i])
    return fitness_array


def choose_mating_pairs(survivors, pop_size):
    weights = survivors[:,0] - survivors[:,0].min() + 1.
    prob = weights/ weights.sum()
    pre_mating_pairs = np.random.choice(int(pop_size/2), size=(int(pop_size/4), 2), p=prob)
    mating_pairs = np.zeros((int(pop_size/4), 2), dtype=np.int64)

    return Main._set_mating_pairs(pop_size, mating_pairs, pre_mating_pairs, survivors)
