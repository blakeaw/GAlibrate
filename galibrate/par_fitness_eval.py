import multiprocessing as mp
import numpy as np


def par_fitness_eval(fitness_func, chromosomes, first_chromosome, nprocs):
    eval_chromosomes = chromosomes[first_chromosome:]
    max_procs = mp.cpu_count()
    if nprocs > max_procs:
        nprocs = max_procs
    pool = mp.Pool(nprocs)
    fitnesses = pool.map(fitness_func, [chromosome for chromosome in eval_chromosomes])
    pool.close()
    pool.join()
    return np.array(fitnesses)
