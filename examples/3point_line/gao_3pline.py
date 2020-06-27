"""
Implementation of a finding the parameters of line with three data
points that have uncertainty using Nestle via Gleipnir. This is a 2 parameter
problem.

Adapted from the Nestle 'Getting started' example at:
http://kylebarbary.com/nestle/
"""

import numpy as np
from scipy.stats import uniform
from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO



# Setupp the data points that are being fitted.
data_x = np.array([1., 2., 3.])
data_y = np.array([1.4, 1.7, 4.1])
data_yerr = np.array([0.2, 0.15, 0.2])

# Define the fitness function
def fitness(chromosome):
    y = chromosome[1] * data_x + chromosome[0]
    chisq = np.sum(((data_y - y) / data_yerr)**2)
    if np.isnan(chisq):
        return -np.inf
    return -chisq / 2.

if __name__ == '__main__':

    # Set up the list of sampled parameters: the prior is Uniform(-5:5) --
    # we are using a fixed uniform prior from scipy.stats
    parm_names = list(['m', 'b'])
    for i in range(998):
        parm_names.append(str(i))
    sampled_parameters = [SampledParameter(name=p, loc=-5.0, width=10.0) for p in parm_names]

    # Set the active point population size
    population_size = 20000
    # Setup the Nested Sampling run
    n_params = len(sampled_parameters)
    print("Sampling a total of {} parameters".format(n_params))
    #population_size = 10
    print("Will use GA population size of {}".format(population_size))
    # Construct the Nested Sampler
    gao = GAO(sampled_parameters,
             fitness,
             population_size,
             generations = 1000,
             mutation_rate = 0.1)
    #print(PCNS.likelihood(np.array([1.0])))
    # run it
    gao.run(verbose=True)
    quit()
    # Print the output
    print("log_evidence: {} +- {} ".format(log_evidence, log_evidence_error))
    best_fit_l = NNS.best_fit_likelihood()
    print("Max likelihood parms: ", best_fit_l)
    best_fit_p, fit_error = NNS.best_fit_posterior()
    print("Max posterior weight parms ", best_fit_p)
    print("Max posterior weight parms error ", fit_error)
    # Information criteria
    # Akaike
    aic = NNS.akaike_ic()
    # Bayesian
    bic = NNS.bayesian_ic(3)
    # Deviance
    dic = NNS.deviance_ic()
    print("AIC ",aic, " BIC ", bic, " DIC ",dic)
    #try plotting a marginal distribution
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Get the posterior distributions -- the posteriors are return as dictionary
        # keyed to the names of the sampled paramters. Each element is a histogram
        # estimate of the marginal distribution, including the heights and centers.
        posteriors = NNS.posteriors()
        # Lets look at the first paramter
        marginal, edges, centers = posteriors[list(posteriors.keys())[0]]
        # Plot with seaborn
        sns.distplot(centers, bins=edges, hist_kws={'weights':marginal})
        # Uncomment next line to plot with plt.hist:
        # plt.hist(centers, bins=edges, weights=marginal)
        plt.show()
    except ImportError:
        pass
