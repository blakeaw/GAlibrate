"""
Implementation of a finding the minimum of an N-dimensional Sphere objective
function,
    f(X) = sum_(i=1)^N x_i**2 ,
on range [-10, 10] using galibrate.GAO.
The minimum is:
    X_0 = (0, 0, ..., 0), f(X_0) = 0
This example was adapted from the Sphere test objective function
description at:
https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.sphere
"""

import numpy as np

from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO
from galibrate.benchmarks import sphere


# Define the fitness function to minimize the 'sphere' objective function.
# minimum is x=0 and f(x) = 0
def fitness(chromosome):
    return -sphere(chromosome)


if __name__ == "__main__":
    # Set up the list of sampled parameters: the range is (-10:10)
    parm_names = list(["x", "y", "z"])
    sampled_parameters = [
        SampledParameter(name=p, loc=-10.0, width=20.0) for p in parm_names
    ]

    # Set the active point population size
    population_size = 100

    # Construct the Genetic Algorithm-based Optimizer.
    gao = GAO(
        sampled_parameters, fitness, population_size, generations=50, mutation_rate=0.1
    )
    # run it
    best_theta, best_theta_fitness = gao.run()
    # Resume the run for an additional 10 generations.
    best_theta, best_theta_fitness = gao.resume(generations=10)
    # print the best theta.
    print(
        "Fittest theta {} with fitness value {} ".format(best_theta, best_theta_fitness)
    )
    try:
        import matplotlib.pyplot as plt

        plt.plot(gao.best_fitness_per_generation)
        plt.show()
    except ImportError:
        pass
