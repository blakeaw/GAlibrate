"""
Implementation of a finding the minimum of an N-dimensional Rastrigin objective
function,
    f(X) = 10*N + sum_(i=1)^N x_i**2 - 10*cos(2*pi*x_i) ,
on range [-5.12, 5.12] using galibrate.GAO.
The minimum is:
    X_0 = (0, 0, ..., 0), f(X_0) = 0
This example was adapted from the Rastrigin test objective function
description at:
https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.rastrigin
"""

import numpy as np

from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO


# Define the fitness function to minimize the N-dimensional Rastrigin function.
# minimum x = 0 and f(x) = 0
def rastrigin(position):
    """N-dimensional Rastrigin function.
    """
    N = len(position)
    xsq = position**2
    cos2pix = np.cos(2*np.pi*position)
    return 10*N + np.sum(xsq - 10*cos2pix)

def fitness(chromosome):
    return -rastrigin(chromosome)


if __name__ == '__main__':


    # Set up the list of sampled parameters: the range is (-5.12:5.12)
    n_params = 2
    sampled_parameters = [SampledParameter(name=p, loc=-5.12,width=10.24) for p in range(n_params)]

    # Set the active point population size
    population_size = 100

    # Construct the Genetic Algorithm-based Optimizer.
    gao = GAO(sampled_parameters, fitness, population_size,
              generations=50,
              mutation_rate=0.1)
    # run it
    best_theta, best_theta_fitness = gao.run()
    # print the best theta.
    print("Fittest theta {} with fitness value {} ".format(best_theta, best_theta_fitness))
    try:
        import matplotlib.pyplot as plt
        plt.plot(gao.best_fitness_per_generation)
        plt.show()
    except ImportError:
        pass
