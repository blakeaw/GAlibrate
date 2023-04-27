"""GAlibrate calibration of a double enzymatic PySB model.
The double enzymatic PySB model is named mm_two_paths_model.py and has
calibration data files product_data.npy and exp_sd.npy.
This example was adapted from:
https://github.com/LoLab-VU/pydyno/tree/master/pydyno/examples/double_enzymatic
"""

import os
import numpy as np
from pysb.simulator import ScipyOdeSimulator

from mm_two_paths_model import model

from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO



# Load the calibration data.
directory = os.path.dirname(__file__)
avg_data_path = os.path.join(directory, 'product_data.npy')
sd_data_path = os.path.join(directory, 'exp_sd.npy')
exp_avg = np.load(avg_data_path)
exp_sd = np.load(sd_data_path)
# PySB simulator timespane
tspan = np.linspace(0, 10, 51)
# PySB model simulator.
solver = ScipyOdeSimulator(model, tspan=tspan)
# Indices of parameters to include in the calibration.
idx_pars_calibrate = [0, 1, 2, 3, 4, 5]
# Mask for the PySB parameters included in the calibration.
rates_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]
# Nominal values of the parameters.
param_values = np.array([p.value for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
# Convert to log10 space.
xnominal = np.log10(nominal_values[rates_mask])
# Set up the sampled parameters for the GAO.
bounds_radius = 2
sampled_parameters = list()
for i,p in enumerate(model.parameters):
    if i in idx_pars_calibrate:
        sampled_parameters.append(SampledParameter(p.name, loc=np.log10(p.value)-2., width=4.))

try:
    import matplotlib.pyplot as plt
    def display(position):
        """Displays the product results for the fit and the calibration data.
        """
        Y = np.copy(position)
        param_values[rates_mask] = 10 ** Y
        sim = solver.run(param_values=param_values).all
        plt.plot(tspan, sim['Product'], label='Fit')
        plt.errorbar(tspan, exp_avg, yerr=exp_sd, label='Data', linestyle="", marker='s')
        plt.legend(loc=0)
        plt.show()
except ImportError:
    def display(position):
        pass


def cost(position):
    """Chi-squared cost function to fit double enzymatic model data.
    """
    Y = np.copy(position)
    param_values[rates_mask] = 10 ** Y
    sim = solver.run(param_values=param_values).all
    e1 = np.sum((exp_avg - sim['Product']) ** 2 / (2 * exp_sd)) / len(exp_avg)
    if np.isnan(e1):
        return np.inf
    return e1

def fitness(theta):
    """GAO fitness function.
    """
    return -cost(theta)

def run_galibrate():
    """Runs the GAlibrate-based calibration.
    """
    population_size = 50
    # Construct the GAO
    gao = GAO(sampled_parameters, fitness, population_size,
              generations=50, mutation_rate=0.1)
    best_theta, best_theta_fitness = gao.run(nprocs=1)
    print("best_theta: ",best_theta)
    print("best_theta_fitness: ", best_theta_fitness)
    display(best_theta)
    np.save('best_theta_gao.npy', best_theta)

if __name__ == '__main__':
    run_galibrate()
