"""
Implementation of a finding the minimum of an N-dimensional Cigar objective
function,
    f(X) = x_1**2 + 10**6 * sum_(i=2)^N x_i**2 ,
on range [-10, 10] using galibrate.GAO.
The minimum is:
    X_0 = (0, 0, ..., 0), f(X_0) = 0
This example was adapted from the Cigar test objective function
description at:
https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.cigar
"""

import numpy as np

from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO


# Define the fitness function to minimize the 'cigar' objective function.
# minimum is x=0 and f(x) = 0
def cigar(position):
    """N-dimensional Cigar function.
    """
    return position[0]**2 + 1e6*np.sum(position[1:]**2)

def fitness(chromosome):
    return -cigar(chromosome)


if __name__ == '__main__':

    # Set up the list of sampled parameters: the range is (-10:10)
    n_params = 6
    sampled_parameters = [SampledParameter(name=p, loc=-10.0,width=20.0) for p in range(n_params)]

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
