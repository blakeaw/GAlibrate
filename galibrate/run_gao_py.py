from __future__ import print_function
import numpy as np



def run_gao(pop_size, n_sp, locs, widths, n_gen,
            mutation_rate, fitness_func):

    # Initialize
    chromosomes = random_population(pop_size, n_sp, locs, widths)
    new_chromosome = np.zeros([pop_size, n_sp], dtype=np.double)
    # Begin generating new generations
    for i_gen in range(n_gen):

        fitnesses = np.array([fitness_func(chromosome) for chromosome in chromosomes])
        i_n_new = int(pop_size/2)
        fitnesses_idxs = np.zeros([pop_size, 2], dtype=np.double)
        for i_mp in range(pop_size):
            fitnesses_idxs[i_mp][0] = fitnesses[i_mp]
            fitnesses_idxs[i_mp][1] = i_mp
        # Selection
        fitnesses_idxs_sort = np.sort(fitnesses_idxs, axis=0)
        survivors = fitnesses_idxs_sort[int(pop_size/2):]
        # Move over the survivors
        for i_mp in range(int(pop_size/2)):
            new_chromosome[i_mp] = chromosomes[int(survivors[i_mp][1])][:]
        mating_pairs = choose_mating_pairs(survivors, pop_size)
        # Generate children
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
        # Replace the old population with the new one
        chromosomes = new_chromosome.copy()
        # Mutation
        if i_gen < (n_gen-1):
            mutation(chromosomes, locs, widths, pop_size, n_sp, mutation_rate)

    return chromosomes

def random_population(pop_size, n_sp,
                      locs, widths):
    chromosomes = np.zeros([pop_size, n_sp], dtype=np.double)
    u = np.random.random([pop_size, n_sp])

    for i in range(pop_size):
        for j in range(n_sp):
            chromosomes[i][j] = locs[j] + u[i][j]*widths[j]

    return chromosomes


def choose_mating_pairs(survivors, pop_size):
    weights = survivors[:,0] - survivors[:,0].min() + 1.
    prob = weights/ weights.sum()
    pre_mating_pairs = np.random.choice(int(pop_size/2), size=[int(pop_size/4), 2], p=prob)
    mating_pairs = np.zeros([int(pop_size/4), 2], dtype=np.int)

    e0 = 0
    e1 = 1
    for i_hps in range(int(pop_size/4)):
        mating_pairs[i_hps][e0] = int(survivors[pre_mating_pairs[i_hps][e0]][e1])
        mating_pairs[i_hps][e1] = int(survivors[pre_mating_pairs[i_hps][e1]][e1])
    return mating_pairs

def crossover(c1, c2, n_sp):

    crossover_point = int(n_sp * np.random.random())
    crossover_beta = np.random.random()
    #cdef np.ndarray[np.double_t, ndim=2] children = np.zeros([2, n_sp], dtype=np.double)
    children = np.zeros([2, n_sp], dtype=np.double)
    x1 = c1[crossover_point]
    x2 = c2[crossover_point]
    x1_c = (1. - crossover_beta)*x1 + crossover_beta*x2
    x2_c = (1. - crossover_beta)*x2 + crossover_beta*x1
    children[0] = c1[:]
    children[1] = c2[:]
    children[0][crossover_point] = x1_c
    children[1][crossover_point] = x2_c

    return children

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
