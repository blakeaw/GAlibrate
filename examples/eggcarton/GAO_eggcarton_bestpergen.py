"""
Implementation of the 2-dimensional 'Egg Carton' problem and its sampling
using an implementation of Genetic algorithm-based optimization via GAlibrate.

Adapted from the pymultinest_demo.py at:
https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest_demo.py

The likelihood landscape has an egg carton-like shape; see slide 15 from:
http://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2016/Lecture14_MultiNest.pdf

"""

import numpy as np
from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO


# Number of parameters to sample is 2
ndim = 2

# Define the loglikelihood function
def fitness(theta):
    chi = (np.cos(theta)).prod()
    return (2. + chi)**5

if __name__ == '__main__':

    # Set up the list of sampled parameters: the prior is Uniform(0:10*pi) --
    # we are using a fixed uniform prior from scipy.stats
    sampled_parameters = [SampledParameter(name=i, loc=0.0, width=10.0*np.pi) for i in range(ndim)]

    # Set the active point population size
    population_size = 100

    # Construct the Genetic Algorithm-based Optimizer.
    gao = GAO(sampled_parameters, fitness, population_size,
              generations=100,
              mutation_rate=0.05)
    # run it
    best_theta, best_theta_fitness = gao.run()

    # print the best theta.
    print("Fittest theta {} with fitness value {} ".format(best_theta, best_theta_fitness))
    try:
        import matplotlib.pyplot as plt
        best_fitness_per_generation = gao.best_fitness_per_generation
        plt.plot(best_fitness_per_generation, marker='o')
        plt.show()
    except ImportError:
        pass
