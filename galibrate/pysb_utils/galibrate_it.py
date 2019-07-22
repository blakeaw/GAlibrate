"""Tools to create GAO objects in order to optimize PySB model parameters.

This module defines the GaoIt and GAlibrateIt classes. GaoIt is used to flag/log
PySB model parameters for sampling during the optimization. With a GaoIt instance
users can optionally set the loc and width of each parameter. GAlibrateIt is used
generate an instance of the GAO object for a PySB. It automatically pulls out
kinetic parameters, or can get the parameters from a GaoIt instance, and can
automatically define a fitness function for the model and a given timeseries
dataset.

"""

import importlib
import os.path
try:
    import pysb
except ImportError:
    pass
from scipy.stats import norm
import numpy as np
from galibrate.sampled_parameter import SampledParameter
from galibrate import GAO


def is_numbers(inputString):
    return all(char.isdigit() for char in inputString)

def parse_directive(directive, priors, no_sample, Keq_sample):
    words = directive.split()
    if words[1] == 'no-sample':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]
        no_sample.append(par)
    elif words[1] == 'sample_Keq':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par_f = model.parameters[par_idx].name
        if is_numbers(words[3]):
            par_idx = int(words[3])
            par_r = model.parameters[par_idx].name
        if is_numbers(words[4]):
            par_idx = int(words[4])
            par_rep = model.parameters[par_idx].name
        no_sample.append(par_rep)
        Keq_sample.append((par_f, par_r, par_rep))
    return

def prune_no_samples(parameters, no_sample):
    pruned_pars = [parameter for parameter in parameters if parameter[0].name not in no_sample]

    return pruned_pars

def update_with_Keq_samples(parameters, Keq_sample):
    k_reps = [sample[2] for sample in Keq_sample]
    pruned_pars = [parameter for parameter in parameters if parameter[0].name not in k_reps]
    return pruned_pars

def write_uniform_param(p_name, p_val):
    line = "sp_{} = SampledParameter(\'{}\', loc=np.log10({})-1.0, width=2.0)\n".format(p_name, p_name, p_val)
    return line

class GaoIt(object):
    """Container to store parameters and priors for sampling.
    This object is geared towards flagging PySB model parameters at the level of
    model definition for genetic algorithm optimization. However, it could be used outside
    model definition.

    Attributes:
        parms (dict of :obj:): A dictionary keyed to the parameter names. The
            values are the parameter search space meta-parameters as given in
            the call function.

    """

    def __init__(self):
        self.parms = dict()
        return

    def __call__(self, parameter, loc=None, width=None):
        """Add a parameter to the list.

        Args:
            parameter (:obj:pysb.Parameter): The parameter to be registered for
                genetic algorithm optimization.
            loc (float): The lower boundary for the parameter search space.
                Should be defined using log10 scale. Default: None
                If None, loc will be set to numpy.log10(parameter.value)-2.0.
            width (float): The width of the parameter search space. Should be
                defined using log10 scale. Default: None
                if None, width will be set to 4.0 (i.e., 4 orders of magnitude).
        """
        if loc is None:
            if width is None:
                loc = np.log10(parameter.value)-2.
            else:
                loc = np.log10(parameter.value) - (width/2.)
        if width is None:
            width = 4.
        self.parms[parameter.name] = tuple((loc, width))
        return parameter

    def __getitem__(self, key):
        return self.parms[key]

    def __setitem__(self, key, loc_width):
        self.parms[key] = loc_width

    def __delitem__(self, key):
        del self.parm[key]

    def __contains__(self, key):
        return (key in self.names)

    def __iadd__(self, parm):
        self.__call__(parm)
        return self

    def __isub__(self, parm):
        try:
            name = parm.name
            self.__delitem__(name)
        except TypeError:
            self.__delitem__(parm)
        return self

    def names(self):
        return list(self.parms.keys())

    def keys(self):
        return self.parms.keys()

    def mask(self, model_parameters):
        names = self.names()
        return [(parm.name in names) for parm in model_parameters]

    def locs(self):
        return np.array([self.parm[name][0] for name in self.parm.keys()])

    def widths(self):
        return np.array([self.parm[name][1] for name in self.parm.keys()])

    def sampled_parameters(self):
        return [SampledParameter(name, self.parm[name][0], self.parm[name][1]) for name in self.keys()]

    def add_all_kinetic_params(self, pysb_model):
        for rule in pysb_model.rules:
            if rule.rate_forward:
                 self.__call__(rule.rate_forward)
            if rule.rate_reverse:
                 self.__call__(rule.rate_reverse)
        return

    def add_all_nonkinetic_params(self, pysb_model):
        kinetic_params = list()
        for rule in pysb_model.rules:
            if rule.rate_forward:
                 kinetic_params.append(rule.rate_forward)
            if rule.rate_reverse:
                 kinetic_params.append(rule.rate_reverse)
        for param in pysb_model.paramters:
            if param not in kinetic_params:
                self.__call__(param)
        return

    def add_by_name(self, pysb_model, name_or_list):
        if isinstance(name_or_list, (list, tuple)):
            for param in pysb_model.parameters:
                if param.name in name_or_list:
                    self.__call__(param)
        else:
            for param in pysb_model.parameters:
                if param.name == name_or_list:
                    self.__call__(param)
        return

class GAlibrateIt(object):
    """Create instances of GAO objects for PySB models.

    Args:
        model (pysb.Model): The instance of the PySB model that you want to run
            genetic algorithm optimization on.
        observable_data (dict of tuple): Defines the observable data to
            use when computing the fitness function. It is a dictionary
            keyed to the model Observables (or species names) that the
            data corresponds to. Each element is a 3 item tuple of format:
            (:numpy.array:data, None or :numpy.array:data_standard_deviations,
            None or :list like:time_idxs or :list like:time_mask).
        timespan (numpy.array): The timespan for model simulations.
        solver (:obj:): The ODE solver to use when running model simulations.
            Defaults to pysb.simulator.ScipyOdeSimulator.
        solver_kwargs (dict): Dictionary of optional keyword arguments to
            pass to the solver when it is initialized. Defaults to dict().
        gao_it (:obj:galibrate.pysb_utils.galirbate_it.GaoIt): An user-defined
            instance of the GaoIt class with the data about the parameters to
            be sampled. If None, the default parameters to be sampled
            are the kinetic rate parameters with ranges of four orders of
            magnitude. Default: None

    Attributes:
        model
        observable_data
        timespan
        solver
        solver_kwargs

    """
    def __init__(self, model, observable_data, timespan,
                 solver=pysb.simulator.ScipyOdeSimulator,
                 solver_kwargs=None, gao_it=None):
        """Inits the GAlibrateIt."""
        if solver_kwargs is None:
            solver_kwargs = dict()
        self.model = model
        self.observable_data = observable_data
        self.timespan = timespan
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        # self.ns_version = None
        self._gao_kwargs = None
        self._like_data = dict()
        self._data = dict()
        self._data_mask = dict()
        for observable_key in observable_data.keys():
            self._like_data[observable_key] = norm(loc=observable_data[observable_key][0],
                                               scale=observable_data[observable_key][1])
            self._data[observable_key] = observable_data[observable_key][0]
            self._data_mask[observable_key] = observable_data[observable_key][2]
            # print(observable_data[observable_key][2])
            if observable_data[observable_key][2] is None:
                self._data_mask[observable_key] = range(len(self.timespan))
        self._model_solver = solver(self.model, tspan=self.timespan, **solver_kwargs)
        if gao_it is not None:
            parm_mask = gao_it.mask(model.parameters)
            self._sampled_parameters = [SampledParameter(parm.name, *gao_it[parm.name]) for i,parm in enumerate(model.parameters) if parm_mask[i]]
            self._rate_mask = parm_mask
        else:
            gao_it = GaoIt()
            for rule in model.rules:
                if rule.rate_forward:
                     gao_it(rule.rate_forward)
                if rule.rate_reverse:
                     gao_it(rule.rate_reverse)
            parm_mask = gao_it.mask(model.parameters)
            self._sampled_parameters = [SampledParameter(parm.name, *gao_it[parm.name]) for i,parm in enumerate(model.parameters) if parm_mask[i]]
            self._rate_mask = parm_mask

        self._param_values = np.array([param.value for param in model.parameters])
        return


    def norm_logpdf_fitness(self, position):
        """Compute the fitness using the normal distribution estimator.

        Args:
            position (numpy.array): The parameter vector the compute fitness
                of.

        Returns:
            float: The natural logarithm of the likelihood estimate.

        """
        Y = np.copy(position)
        params = self._param_values.copy()
        params[self._rate_mask] = 10**Y
        sim = self._model_solver.run(param_values=[params]).all
        logl = 0.
        for observable in self._like_data.keys():
            sim_vals = sim[observable][self._data_mask[observable]]
            logl += np.sum(self._like_data[observable].logpdf(sim_vals))
        if np.isnan(logl):
            return -np.inf
        return logl

    def mse_fitness(self, position):
        """Compute the fitness using the negative mean squared error estimator.

        Args:
            position (numpy.array): The parameter vector the compute fitness of.

        Returns:
            float: The natural logarithm of the likelihood estimate.

        """
        Y = np.copy(position)
        params = self._param_values.copy()
        params[self._rate_mask] = 10**Y
        sim = self._model_solver.run(param_values=[params]).all
        logl = 0.0
        for observable in self._like_data.keys():
            sim_vals = sim[observable][self._data_mask[observable]]
            logl -= np.mean((self._data[observable]-sim_vals)**2)
        if np.isnan(logl):
            return -np.inf
        return logl

    def sse_fitness(self, position):
        """Compute the fitness using the negative sum of squared errors estimator.

        Args:
            position (numpy.array): The parameter vector the compute fitness
                of.

        Returns:
            float: The natural logarithm of the likelihood estimate.

        """
        Y = np.copy(position)
        params = self._param_values.copy()
        params[self._rate_mask] = 10**Y
        sim = self._model_solver.run(param_values=[params]).all
        logl = 0.0
        for observable in self._like_data.keys():
            sim_vals = sim[observable][self._data_mask[observable]]
            logl -= np.sum((self._data[observable]-sim_vals)**2)
        if np.isnan(logl):
            return -np.inf
        return logl

    def __call__(self, gao_population_size=1000, gao_kwargs=None,
                 fitness_type='norm_logpdf'):
        """Call the NestedSampleIt instance to construct to instance of the NestedSampling object.

        Args:
                gao_population_size (int): Set the size of the active population
                    of sample points to use during genetic algorithm optimization runs.
                    Defaults to 1000.
                gao_kwargs (dict): Dictionary of any additional optional keyword
                    arguments to pass to NestedSampling object constructor.
                    Defaults to dict().
                fitness_type (str): Define the type of fitness estimator
                    to use. Options are 'norm_logpdf'=>Compute the fitness using
                    the normal distribution estimator, 'mse'=>Compute the
                    fitness using the negative mean squared error estimator,
                    'sse'=>Compute the fitness using the negative sum of
                     squared errors estimator. Defaults to 'norm_logpdf'.

        Returns:
            type: Description of returned object.

        """
        if gao_kwargs is None:
            gao_kwargs = dict()
        # self.ns_version = ns_version
        self._gao_kwargs = gao_kwargs
        population_size = gao_population_size
        if fitness_type == 'mse':
            fitness = self.mse_fitness
        elif fitness_type == 'sse':
            fitness = self.sse_fitness
        else:
            fitness = self.norm_logpdf_fitness

        # Construct the GAO
        gao = GAO(self._sampled_parameters,
                  fitness,
                  population_size,
                  **gao_kwargs)

        return gao


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', type=str, help='The model.py file that you want run NS on.')
    parser.add_argument('output_path', metavar='output_path', type=str, help='The file location where you want to output the NS run script.')
    args = parser.parse_args()
    # get the input script from command line imputs
    model_file = os.path.abspath(args.model_file)
    # get the location to dump the output
    output_path = os.path.abspath(args.output_path)

    print("Using model from file: {}".format(model_file))
    base=os.path.basename(model_file)
    model_module_name = os.path.splitext(base)[0]
    print("With name: {}".format(model_module_name))
    default_prior_shape = 'norm'
    print("The default prior shape is: {}".format(default_prior_shape))
    #print(model_file)

    #print(model_module_name)
    model_module = importlib.import_module(model_module_name)
    model = getattr(model_module, 'model')
    #print(model)
    priors = dict()
    no_sample = list()
    Keq_sample = list()
    #Read the file and parse any #GALIBRATE_IT directives
    print("Parsing the model for any #GALIBRATE_IT directives...")
    with open(model_file, 'r') as file_obj:
        for line in file_obj:
            words = line.split()
            if len(words) > 1:
                if words[0] == '#GALIBRATE_IT':
                    parse_directive(line, priors, no_sample, Keq_sample)

    #now we need to extract a list of kinetic parameters
    parameters = list()
    print("Inspecting the model and pulling out kinetic parameters...")
    for rule in model.rules:
        print(rule.rate_forward, rule.rate_reverse)
        #print(rule_keys)
        if rule.rate_forward:
            param = rule.rate_forward
            #print(param)
            parameters.append([param,'f'])
        if rule.rate_reverse:
            param = rule.rate_reverse
            #print(param)
            parameters.append([param, 'r'])
    #print(no_sample)
    parameters = prune_no_samples(parameters, no_sample)
    parameters = update_with_Keq_samples(parameters, Keq_sample)
    #print(parameters)
    print("Found the following kinetic parameters:")
    print("{}".format(parameters))
    #default the priors to norm - i.e. normal distributions
    for parameter in parameters:
        name = parameter[0].name
        if name not in priors.keys():
            priors[name] = default_prior_shape

    # Obtain mask of sampled parameters to run simulation in the likelihood function
    parameters_idxs = [model.parameters.index(parameter[0]) for parameter in parameters]
    calibrate_mask = [i in parameters_idxs for i in range(len(model.parameters))]
    param_values = [p.value for p in model.parameters]

    out_file = open("galibrate_"+base, 'w')

    print("Writing to GAlibrate run script: galibrate_{}".format(base))
    out_file.write("\'\'\'\nGenerated by galibrate_it\n")
    out_file.write("GAlibrate run script for {} \n".format(base))
    out_file.write("\'\'\'")
    out_file.write("\n")
    out_file.write("from pysb.simulator import ScipyOdeSimulator\n")
    out_file.write("import numpy as np\n")
    out_file.write("from scipy.stats import norm,uniform\n")
    out_file.write("from galibrate import GAO\n")
    out_file.write("from galibrate.sampled_parameter import SampledParameter \n")

    #out_file.write("import inspect\n")
    #out_file.write("import os.path\n")
    out_file.write("from "+model_module_name+" import model\n")
    out_file.write("\n")

    out_file.write("# Initialize PySB solver object for running simulations.\n")
    out_file.write("# USER-Adjust: simulation timespan should match experimental data.\n")
    out_file.write("tspan = np.linspace(0,10, num=100)\n")
    out_file.write("solver = ScipyOdeSimulator(model, tspan=tspan)\n")
    out_file.write("parameters_idxs = " + str(parameters_idxs)+"\n")
    out_file.write("calibrate_mask = " + str(calibrate_mask)+"\n" )
    out_file.write("param_values = np.array([p.value for p in model.parameters])\n" )
    out_file.write("\n")
    out_file.write("# USER-Set: must add commands to import/load any experimental\n")
    out_file.write("# data for use in the likelihood function!\n")
    out_file.write("experiments_avg = np.load()\n")
    out_file.write("experiments_sd = np.load()\n")
    out_file.write("like_data = norm(loc=experiments_avg, scale=experiments_sd)\n")
    out_file.write("# USER-Set: must appropriately update fitness function!\n")
    out_file.write("def fitness(position):\n")
    out_file.write("    Y=np.copy(position)\n")
    out_file.write("    param_values[calibrate_mask] = 10 ** Y\n")
    out_file.write("    sim = solver.run(param_values=param_values).all\n")
    out_file.write("    logp_data = np.sum(like_data.logpdf(sim['observable']))\n")
    out_file.write("    if np.isnan(logp_data):\n")
    out_file.write("        logp_data = -np.inf\n")
    out_file.write("    return logp_data\n")
    out_file.write("\n")
    #write the sampled params lines
    out_file.write("# Setup the Genetic Algorithm optimization run\n \n")
    out_file.write("# Setup the parameters that being sampled. \n")
    out_file.write("sampled_parameters = list()\n")
    for parameter in parameters:
        name = parameter[0].name
        value = parameter[0].value
        prior_shape = 'uniform'
        print("Will sample parameter {} with around {}".format(name, value))
        if prior_shape == 'uniform':
            line = write_uniform_param(name, value)
            ps_name = line.split()[0]
            out_file.write(line)
            out_file.write("sampled_parameters.append({})\n".format(ps_name))

    out_file.write("n_params = len(sampled_parameters)\n")
    out_file.write("# Set the population size. \n")
    out_file.write("population_size = 10*n_params\n \n")
    out_file.write("# Construct the optimizer \n")
    out_file.write("gao = GAO(sampled_parameters,\n")
    out_file.write("          fitness, population_size,\n")
    out_file.write("          generations=1000,\n")
    out_file.write("          mutation_rate=0.05)\n \n")
    out_file.write("# run it\n")
    out_file.write("fittest_theta, fitness_val = gao.run()\n")
    out_file.write("print(\"Best parameters: \",fittest_theta)\n")
    out_file.write("print(\"Best parameters fitness: \", fitness_val)\n")


    out_file.close()
    print("galibrate_it is complete!")
    print("END OF LINE.")
