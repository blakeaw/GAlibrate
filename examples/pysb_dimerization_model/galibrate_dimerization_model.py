'''
GAlibrate GAO run script for dimerization_model.py
'''
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from scipy.stats import norm
from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO
from dimerization_model import model

# Initialize PySB solver object for running simulations.
# Simulation timespan should match experimental data.
tspan = np.linspace(0,1, num=51)
solver = ScipyOdeSimulator(model, tspan=tspan, integrator='lsoda')
parameters_idxs = [0, 1]
rates_mask = [True, True, False]
param_values = np.array([p.value for p in model.parameters])

# USER must add commands to import/load any experimental
# data for use in the likelihood function!
experiments_avg = np.load('dimerization_model_dimer_data.npy')
experiments_sd = np.load('dimerization_model_dimer_sd.npy')
like_data = norm(loc=experiments_avg, scale=10.0*experiments_sd)

#@numba.jit
def fitness(position):
    Y=np.copy(position)
    param_values[rates_mask] = 10 ** Y
    sim = solver.run(param_values=param_values).all
    #    return -np.inf
    # sim = solver.run(param_values=param_values).all
    logp_data = np.sum(like_data.logpdf(sim['A_dimer']))
    if np.isnan(logp_data):
        logp_data = -np.inf
    return logp_data

if __name__ == '__main__':
    sampled_parameters = list()
    sp_kf = SampledParameter('kf', loc=np.log10(0.001)-0.5, width=1.)
    sampled_parameters.append(sp_kf)
    sp_kr = SampledParameter('kr', loc=np.log10(1.0)-0.5, width=1.)
    sampled_parameters.append(sp_kr)
    # Setup the Nested Sampling run
    n_params = len(sampled_parameters)
    population_size = 10
    # Construct the GAO
    gao = GAO(sampled_parameters, fitness, population_size,
              generations=10, mutation_rate=0.05)
    # run it
    best_theta, best_theta_fitness = gao.run()
    print("best_theta: ",best_theta)
    print("best_theta_fitness: ", best_theta_fitness)
