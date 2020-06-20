from __future__ import print_function
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):
        return iterator
cimport numpy as np
import cython


@cython.cdivision(True)
def run_gao(int pop_size, int n_sp, np.ndarray[np.double_t, ndim=1] locs,
           np.ndarray[np.double_t, ndim=1] widths, int n_gen,
            double mutation_rate, object fitness_func):
    cdef int i_gen, i_mp
    cdef int i_mate1_idx, i_mate2_idx
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] fitnesses = np.zeros(pop_size, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] new_chromosome
    cdef int i_n_new
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] fitnesses_idxs
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] fitnesses_idxs_sort
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] survivors
    cdef np.ndarray[np.int_t, ndim=2, mode='c'] mating_pairs
    cdef np.ndarray chromosome1, chromosome2, children, child1, child2
    # Initialize
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] chromosomes = random_population(pop_size, n_sp, locs, widths)
    new_chromosome = np.zeros([pop_size, n_sp], dtype=np.double)
    # Begin generating new generations
    for i_gen in tqdm(range(n_gen), desc='Generations: '):
        i_n_new = pop_size/2

        if i_gen == 0:
            fitnesses = np.array([fitness_func(chromosome) for chromosome in chromosomes])
        else:
            fitnesses = _compute_fitnesses(fitness_func, chromosomes, pop_size, i_n_new, fitnesses)

        fitnesses_idxs = np.zeros([pop_size, 2], dtype=np.double)
        _fill_fitness_idxs(pop_size, fitnesses, fitnesses_idxs)
        #for i_mp in range(pop_size):
        #    fitnesses_idxs[i_mp][0] = fitnesses[i_mp]
        #    fitnesses_idxs[i_mp][1] = i_mp
        # Selection
        fitnesses_idxs_sort = np.sort(fitnesses_idxs, axis=0)
        survivors = fitnesses_idxs_sort[pop_size/2:]
        # Move over the survivors
        _move_over_survivors(pop_size, survivors, chromosomes, new_chromosome)
        #for i_mp in range(pop_size/2):
        #    new_chromosome[i_mp] = chromosomes[int(survivors[i_mp][1])][:]
        mating_pairs = choose_mating_pairs(survivors, pop_size)
        # Generate children
        _generate_children(pop_size, n_sp, i_n_new, mating_pairs, chromosomes, new_chromosome)
#        for i_mp in range(pop_size/4):
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
        #chromosomes = new_chromosome.copy()
        double_deepcopy_2d(chromosomes, new_chromosome, pop_size, n_sp)
        # Mutation
        if i_gen < (n_gen-1):
            mutation(chromosomes, locs, widths, pop_size, n_sp, mutation_rate)
        _copy_survivor_fitnesses(pop_size, survivors, fitnesses)
    return chromosomes

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_fitness_idxs(int pop_size, double[:] fitnesses,
                             double[:,:] fitnesses_idxs):
    cdef Py_ssize_t i_mp
    for i_mp in range(pop_size):
        fitnesses_idxs[i_mp][0] = fitnesses[i_mp]
        fitnesses_idxs[i_mp][1] = i_mp
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _move_over_survivors(int pop_size, double[:,:] survivors,
                               double[:,:] chromosomes,
                               double[:,:] new_chromosome):
    cdef Py_ssize_t i_mp
    for i_mp in range(pop_size/2):
        new_chromosome[i_mp] = chromosomes[<int>survivors[i_mp][1]][:]
    return

def _compute_fitnesses(fitness_func, chromosomes, pop_size, start, fitness_array):
    for i in range(start, pop_size):
        fitness_array[i] = fitness_func(chromosomes[i])
    return fitness_array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _copy_survivor_fitnesses(int pop_size, double[:,:] survivors,
                                   double[:] fitness_array):
    cdef int stop = pop_size/2
    cdef Py_ssize_t i
    for i in range(0, stop):
        fitness_array[i] = survivors[i][0]
    return

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _generate_children(int pop_size, int n_sp, int i_n_new,
                             long[:,:] mating_pairs, double[:,:] chromosomes,
                             double[:,:] new_chromosome):
    cdef Py_ssize_t i_mp, i_mate1_idx, i_mate2_idx
    cdef double[:] chromosome1
    cdef double[:] chromosome2
    cdef double[:] child1
    cdef double[:] child2
    cdef double[:,:] children

    for i_mp in range(pop_size/4):
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
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=2] random_population(int pop_size, int n_sp,
                      double[:] locs,
                      double[:] widths):
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=2] chromosomes = np.zeros([pop_size, n_sp], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] u = np.random.random([pop_size, n_sp])
    cdef double[:,:] chromosomes_v = chromosomes
    cdef double[:,:] u_v = u

    for i in range(pop_size):
        for j in range(n_sp):
            chromosomes_v[i][j] = locs[j] + u_v[i][j]*widths[j]

    return chromosomes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.int_t, ndim=2] choose_mating_pairs(np.ndarray[np.double_t, ndim=2] survivors,
                                                      int pop_size):
    cdef np.ndarray[np.double_t, ndim=1] weights = survivors[:,0] - survivors[:,0].min() + 1.
    cdef double[:,:] survivors_view = survivors
    cdef np.ndarray[np.double_t, ndim=1] prob = weights/ weights.sum()
    cdef np.ndarray[np.int_t, ndim=2] pre_mating_pairs = np.random.choice(pop_size/2, size=[pop_size/4, 2], p=prob)
    cdef long[:,:] pre_mating_pairs_view = pre_mating_pairs
    cdef Py_ssize_t i_hps, e0, e1
    cdef np.ndarray[np.int_t, ndim=2] mating_pairs = np.zeros([pop_size/4, 2], dtype=np.int)
    cdef long[:,:] mating_pairs_view = mating_pairs
    e0 = 0
    e1 = 1
    for i_hps in range(pop_size/4):
        mating_pairs_view[i_hps][e0] = <int>survivors_view[pre_mating_pairs_view[i_hps][e0]][e1]
        mating_pairs_view[i_hps][e1] = <int>survivors_view[pre_mating_pairs_view[i_hps][e1]][e1]
    return mating_pairs

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=2] crossover(double[:] c1,
                           double[:] c2,
                           int n_sp):

    cdef Py_ssize_t crossover_point = <int>(n_sp * np.random.random())
    cdef double crossover_beta = np.random.random()
    #cdef np.ndarray[np.double_t, ndim=2] children = np.zeros([2, n_sp], dtype=np.double)
    cdef children = np.zeros([2, n_sp], dtype=np.double)
    cdef double[:,:] children_v
    children_v = children
    cdef double x1, x2, x1_c, x2_c

    x1 = c1[crossover_point]
    x2 = c2[crossover_point]
    x1_c = (1. - crossover_beta)*x1 + crossover_beta*x2
    x2_c = (1. - crossover_beta)*x2 + crossover_beta*x1
    children_v[0] = c1[:]
    children_v[1] = c2[:]
    children_v[0][crossover_point] = x1_c
    children_v[1][crossover_point] = x2_c

    return children

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void mutation(double[:,:] chromosomes,
                   double[:] locs,
                   double[:] widths,
              int pop_size, int n_sp, double mutation_rate):

    cdef Py_ssize_t i, j
    #cdef int i, j
    # cdef double u, v
    cdef int half_pop_size = pop_size/2
    # cdef int half_pop_size_p2 = half_pop_size + 2
    cdef np.ndarray[np.double_t, ndim=2] u = np.random.random([half_pop_size, n_sp])
    cdef double[:,:] u_v = u
    cdef np.ndarray[np.double_t, ndim=2] v = np.random.random([half_pop_size, n_sp])
    cdef double[:,:] v_v = v
    cdef double uij, vij
    cdef Py_ssize_t iadjust

    for i in range(half_pop_size, pop_size):
        iadjust = i - half_pop_size
        for j in range(n_sp):
            # u = np.random.random()
            uij = u_v[iadjust][j]
            if uij < mutation_rate:
                # v = np.random.random()
                vij = v_v[iadjust][j]
                chromosomes[i][j] = locs[j] + widths[j]*vij
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void double_deepcopy_2d(double[:,:] copy_to, double[:,:] copy_from, int x, int y):
    cdef Py_ssize_t i, j
    for i in range(x):
        for j in range(y):
            copy_to[i][j] = copy_from[i][j]
    return
