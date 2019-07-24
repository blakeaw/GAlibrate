from __future__ import print_function
import numpy as np
import numba

#@numba.jit(nopython=False)
def run_gao(pop_size, n_sp, locs, widths, n_gen,
            mutation_rate, fitness_func):

    #@numba.jit('float64(float64[:])', cache=True)
    #def wrap_fitness_func(theta):
    #    return fitness_func(theta)
    # Initialize
    chromosomes = random_population(pop_size, n_sp, locs, widths)
    new_chromosome = np.zeros((pop_size, n_sp))
    #fitnesses = np.zeros(pop_size)

    # Begin generating new generations
    for i_gen in range(n_gen):
        i_n_new = int(pop_size/2)
        #fitnesses = np.array([fitness_func(chromosome) for chromosome in chromosomes])
        if i_gen == 0:
            fitnesses = np.array([fitness_func(chromosome) for chromosome in chromosomes])
            #fitnesses = _compute_fitnesses(fitness_func, chromosomes, pop_size, 0, fitnesses)
        else:
            #fitnesses = np.array([fitness_func(chromosome) for i in range(pop)])
            fitnesses = _compute_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses)

        fitnesses_idxs = np.zeros((pop_size, 2), dtype=np.double)
        fitnesses_idxs = _fill_fitness_idxs(pop_size, fitnesses, fitnesses_idxs)

        # Selection
        fitnesses_idxs_sort = np.sort(fitnesses_idxs, axis=0)
        survivors = fitnesses_idxs_sort[int(pop_size/2):]
        # Move over the survivors
        new_chromosome = _move_over_survivors(pop_size, survivors, chromosomes, new_chromosome)
        #for i_mp in range(int(pop_size/2)):
        #    new_chromosome[i_mp] = chromosomes[int(survivors[i_mp][1])][:]
        mating_pairs = choose_mating_pairs(survivors, pop_size)
        # Generate children
        new_chromosome = _generate_children(pop_size, n_sp, i_n_new, mating_pairs, chromosomes, new_chromosome)
#        for i_mp in range(int(pop_size/4)):
#            i_mate1_idx = mating_pairs[i_mp][0]
#            i_mate2_idx = mating_pairs[i_mp][1]
#            chromosome1 = chromosomes[i_mate1_idx,:]
#            chromosome2 = chromosomes[i_mate2_idx,:]
#            # Crossover and update the chromosomes
#            children = crossover(chromosome1, chromosome2, n_sp)
#            child1 = children[0,:]
#            child2 = children[1, :]
#            new_chromosome[i_n_new] = child1
#            i_n_new = i_n_new + 1
#            new_chromosome[i_n_new] = child2
#            i_n_new = i_n_new + 1
        # Replace the old population with the new one
        chromosomes = new_chromosome.copy()
        # Mutation
        if i_gen < (n_gen-1):
            mutation(chromosomes, locs, widths, pop_size, n_sp, mutation_rate)
        fitnesses = _copy_survivor_fitnesses(pop_size, survivors, fitnesses)
    return chromosomes

@numba.njit(cache=True)
def _fill_fitness_idxs(pop_size, fitnesses, fitnesses_idxs):
    for i_mp in range(pop_size):
        fitnesses_idxs[i_mp][0] = fitnesses[i_mp]
        fitnesses_idxs[i_mp][1] = i_mp
    return fitnesses_idxs

@numba.njit(cache=True)
def _move_over_survivors(pop_size, survivors, chromosomes, new_chromosome):
    for i_mp in range(int(pop_size/2)):
        new_chromosome[i_mp] = chromosomes[int(survivors[i_mp][1])][:]
    return new_chromosome

#@numba.jit(forceobj=True)
def _compute_fitnesses(fitness_func, chromosomes, pop_size, start, fitness_array):
    for i in range(start, pop_size):
        fitness_array[i] = fitness_func(chromosomes[i])
    return fitness_array

@numba.njit(cache=True)
def _copy_survivor_fitnesses(pop_size, survivors, fitness_array):
    stop = int(pop_size/2)
    for i in range(0, stop):
        fitness_array[i] = survivors[i][0]
    return fitness_array

@numba.njit(cache=True)
def _generate_children(pop_size, n_sp, i_n_new, mating_pairs, chromosomes, new_chromosome):

    for i_mp in range(int(pop_size/4)):
        i_mate1_idx = mating_pairs[i_mp][0]
        i_mate2_idx = mating_pairs[i_mp][1]
        chromosome1 = chromosomes[i_mate1_idx,:]
        chromosome2 = chromosomes[i_mate2_idx,:]
        # Crossover and update the chromosomes
        children = crossover(chromosome1, chromosome2, n_sp)
        child1 = children[0,:]
        child2 = children[1, :]
        new_chromosome[i_n_new] = child1
        i_n_new = i_n_new + 1
        new_chromosome[i_n_new] = child2
        i_n_new = i_n_new + 1
    return new_chromosome

@numba.njit(cache=True)
def random_population(pop_size, n_sp,
                      locs, widths):
    chromosomes = np.zeros((pop_size, n_sp))
    u = np.random.random((pop_size, n_sp))

    for i in range(pop_size):
        for j in range(n_sp):
            chromosomes[i][j] = locs[j] + u[i][j]*widths[j]

    return chromosomes


def choose_mating_pairs(survivors, pop_size):
    weights = survivors[:,0] - survivors[:,0].min() + 1.
    prob = weights/ weights.sum()
    pre_mating_pairs = np.random.choice(int(pop_size/2), size=(int(pop_size/4), 2), p=prob)
    mating_pairs = np.zeros((int(pop_size/4), 2), dtype=np.int)

    return _set_mating_pairs(pop_size, mating_pairs, pre_mating_pairs, survivors)

@numba.njit(cache=True)
def _set_mating_pairs(pop_size, mating_pairs, pre_mating_pairs, survivors):
    e0 = 0
    e1 = 1
    for i_hps in range(int(pop_size/4)):
        mating_pairs[i_hps][e0] = int(survivors[pre_mating_pairs[i_hps][e0]][e1])
        mating_pairs[i_hps][e1] = int(survivors[pre_mating_pairs[i_hps][e1]][e1])
    return mating_pairs

@numba.njit(cache=True)
def crossover(c1, c2, n_sp):

    crossover_point = int(n_sp * np.random.random())
    crossover_beta = np.random.random()
    #cdef np.ndarray[np.double_t, ndim=2] children = np.zeros([2, n_sp], dtype=np.double)
    children = np.zeros((2, n_sp), dtype=np.double)
    x1 = c1[crossover_point]
    x2 = c2[crossover_point]
    x1_c = (1. - crossover_beta)*x1 + crossover_beta*x2
    x2_c = (1. - crossover_beta)*x2 + crossover_beta*x1
    children[0] = c1[:]
    children[1] = c2[:]
    children[0][crossover_point] = x1_c
    children[1][crossover_point] = x2_c

    return children

@numba.njit(cache=True)
def mutation(chromosomes,
             locs,
             widths,
             pop_size, n_sp, mutation_rate):


    half_pop_size = int(pop_size/2)

    for i in range(half_pop_size, pop_size):

        for j in range(n_sp):
            u = np.random.random()

            if u < mutation_rate:
                v = np.random.random()

                chromosomes[i][j] = locs[j] + widths[j]*v
    return
