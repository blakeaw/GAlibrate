"""
Implementation of a finding the minimum of an 2-dimensional Himmelblau objective
function,
    f(x_1, x_2) =  (x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2 ,
on range [-6,6] using galibrate.GAO.
This function has four minima:
    X_1 = (3.0, 2.0), f(X_1) = 0
    X_2 = (-2.805118, 3.131312), f(X_2) = 0
    X_3 = (-3.779310, -3.283186), f(X_3) = 0
    X_4 = (3.584428, -1.848126), f(X_4) = 0
This example was adapted from the Himmelblau test objective function
description at:
https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.himmelblau
"""

import numpy as np

from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO


# Define the fitness function (minimize himmelblau function)
# minima:
# x1 = (3.0,2.0) and f(x1) = 0
# x2 = (-2.805118,3.131312) and f(x2) = 0
# x3 = (-3.779310,-3.283186) and f(x3) = 0
# x4 = (3.584428,-1.848126) and f(x4) = 0
def himmelblau(x_1, x_2):
    """2-dimensional Himmelblau function.
    """
    return (x_1**2 + x_2 - 11)**2 + (x_1 + x_2**2 - 7)**2

def fitness(chromosome):
    return -himmelblau(chromosome[0], chromosome[1])


if __name__ == '__main__':

    # Set up the list of sampled parameters: the range is (-6:6)
    params = ['x1', 'x2']
    sampled_parameters = [SampledParameter(name=p, loc=-6.0,width=12.0) for p in params]

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
