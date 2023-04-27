"""
Implementation of a finding the parameters of line with three data
points that have uncertainty using galibrate.GAO. This is a 2 parameter
problem; i.e., with linear equation model
    y = mx + b,
where the slope, m, and intercept, b, are the free parameters.
Adapted from the Nestle 'Getting started' example at:
http://kylebarbary.com/nestle/
"""

import numpy as np
from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO


# Set up the data points that are being fitted.
data_x = np.array([1.0, 2.0, 3.0])
data_y = np.array([1.4, 1.7, 4.1])
data_yerr = np.array([0.2, 0.15, 0.2])


# Define the fitness function (minimize chi-squared value)
def fitness(chromosome):
    y = chromosome[0] * data_x + chromosome[1]
    chisq = np.sum(((data_y - y) / data_yerr) ** 2)
    return -chisq / 2.0


if __name__ == "__main__":
    # Set up the list of sampled parameters: the range is (-10:10)
    parm_names = list(["m", "b"])
    sampled_parameters = [
        SampledParameter(name=p, loc=-10.0, width=20.0) for p in parm_names
    ]

    # Set the active point population size
    population_size = 200

    # Construct the Genetic Algorithm-based Optimizer.
    gao = GAO(
        sampled_parameters, fitness, population_size, generations=100, mutation_rate=0.1
    )
    # run it
    best_theta, best_theta_fitness = gao.run()
    # print the best theta.
    print(
        "Fittest theta {} with fitness value {} ".format(best_theta, best_theta_fitness)
    )
    try:
        import matplotlib.pyplot as plt

        plt.errorbar(
            data_x, data_y, yerr=data_yerr, linestyle="", marker="s", label="Data"
        )
        x = np.arange(5)
        y = best_theta[0] * x + best_theta[1]
        plt.plot(x, y, linestyle="--", label="Fit")
        plt.legend(loc=0)
        # plt.plot(gao.best_fitness_per_generation)
        plt.show()
    except ImportError:
        pass
