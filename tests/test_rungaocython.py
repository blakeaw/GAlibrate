import numpy as np
import pyximport
# Added the setup_args with include_dirs for numpy so it pyximport can build
# the code on Windows machine. 
pyximport.install(language_level=3, setup_args={"include_dirs":np.get_include()})
from galibrate import run_gao_cython

from galibrate.sampled_parameter import SampledParameter
from galibrate.benchmarks import sphere


# Define the fitness function to minimize the 'sphere' objective function.
# minimum is x=0 and f(x) = 0
def fitness(chromosome):
    return -sphere(chromosome)


# Set up the list of sampled parameters: the range is (-10:10)
parm_names = list(["x", "y", "z"])
sampled_parameters = [
    SampledParameter(name=p, loc=-10.0, width=20.0) for p in parm_names
]
min_point = np.zeros(3)
locs_sphere = np.zeros(3) - 10.0
widths_sphere = np.zeros(3) + 20.0

# Set the active point population size
population_size = 20
generations = 10
additional = 10
mutation_rate_sphere = 0.2
SHARED = dict()


def test_rungaocython_run_gao():
    new_population, best_pg = run_gao_cython.run_gao(
        population_size,
        len(parm_names),
        locs_sphere,
        widths_sphere,
        generations,
        mutation_rate_sphere,
        fitness,
        1,
    )
    assert new_population.shape == (population_size, len(parm_names))
    SHARED['population'] = new_population
    new_population, best_pg = run_gao_cython.run_gao(
        population_size,
        len(parm_names),
        locs_sphere,
        widths_sphere,
        generations,
        mutation_rate_sphere,
        fitness,
        2,
    )
    assert new_population.shape == (population_size, len(parm_names))

def test_rungaocython_continue_gao():
    fitnesses = np.array([fitness(individual) for individual in SHARED['population']])
    new_population, best_pg = run_gao_cython.continue_gao(
        population_size,
        len(parm_names),
        SHARED['population'],
        fitnesses,
        locs_sphere,
        widths_sphere,
        additional,
        mutation_rate_sphere,
        fitness,
        1,
    )
    assert new_population.shape == (population_size, len(parm_names))
    new_population, best_pg = run_gao_cython.continue_gao(
        population_size,
        len(parm_names),
        SHARED['population'],
        fitnesses,
        locs_sphere,
        widths_sphere,
        additional,
        mutation_rate_sphere,
        fitness,
        2,
    )
    assert new_population.shape == (population_size, len(parm_names))

if __name__ == "__main__":
    test_rungaocython_run_gao()
    test_rungaocython_continue_gao()