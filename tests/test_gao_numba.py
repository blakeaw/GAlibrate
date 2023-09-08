import numpy as np

from galibrate.sampled_parameter import SampledParameter

# from galibrate import run_gao_numba
from galibrate import gao

gao._set_run_gao_numba()
# gao.run_gao = run_gao_numba
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


# Set the active point population size
population_size = 1000
generations = 200
additional = 10
mutation_rate = 0.2
SHARED = dict()


def test_initialization():
    # Construct the Genetic Algorithm-based Optimizer.
    gao_loc = gao.GAO(
        sampled_parameters,
        fitness,
        population_size,
        generations=generations,
        mutation_rate=mutation_rate,
    )
    SHARED["gao"] = gao_loc


def test_attributes():
    assert SHARED["gao"].fitness_function == fitness
    assert SHARED["gao"].sampled_parameters == sampled_parameters
    assert SHARED["gao"].population_size == population_size
    assert SHARED["gao"].generations == generations
    assert np.isclose(SHARED["gao"].mutation_rate, mutation_rate)


def test_run():
    best_theta, best_theta_fitness = SHARED["gao"].run()
    assert np.isclose(best_theta_fitness, 0.0)
    assert np.allclose(min_point, best_theta)


def test_resume():
    best_theta, best_theta_fitness = SHARED["gao"].resume(generations=additional)
    assert np.isclose(best_theta_fitness, 0.0)
    assert np.allclose(min_point, best_theta)


def test_property_best():
    best_theta, best_theta_fitness = SHARED["gao"].best
    assert np.isclose(best_theta_fitness, 0.0)
    assert np.allclose(min_point, best_theta)


def test_property_best_fitness_per_generation():
    bests = SHARED["gao"].best_fitness_per_generation
    assert bests is not None
    assert len(bests) == (generations + additional + 2)
    assert np.isclose(bests[-1], 0.0)
    assert bests[0] < 0.0


def test_property_total_generations():
    total = SHARED["gao"].total_generations
    assert total == (generations + additional)


def test_property_final_population():
    final = SHARED["gao"].final_population
    assert final is not None
    assert len(final) == population_size
    assert len(final[0]) == len(sampled_parameters)


def test_property_final_population_fitness():
    final = SHARED["gao"].final_population_fitness
    assert final is not None
    assert len(final) == population_size
    assert np.isclose(np.max(final), 0.0)


def test_run_parallel():
    # Construct the Genetic Algorithm-based Optimizer.
    gao_loc = gao.GAO(
        sampled_parameters,
        fitness,
        population_size,
        generations=1,
        mutation_rate=mutation_rate,
    )
    SHARED["gao_par"] = gao_loc
    best_theta, best_theta_fitness = SHARED["gao_par"].run(nprocs=2)


def test_resume_parallel():
    best_theta, best_theta_fitness = SHARED["gao_par"].resume(generations=1, nprocs=2)


if __name__ == "__main__":
    test_initialization()
    test_attributes()
    test_run()
    test_resume()
    test_property_best()
    test_property_best_fitness_per_generation()
    test_property_total_generations()
    test_property_final_population()
    test_property_final_population_fitness()
    test_run_parallel()
    test_resume_parallel()
    # # Construct the Genetic Algorithm-based Optimizer.
    # gao = GAO(
    #     sampled_parameters, fitness, population_size, generations=100, mutation_rate=0.1
    # )
    # # run it
    # best_theta, best_theta_fitness = gao.run()
    # # print the best theta.
    # print(
    #     "Fittest theta {} with fitness value {} ".format(best_theta, best_theta_fitness)
    # )
    # try:
    #     import matplotlib.pyplot as plt

    #     plt.plot(gao.best_fitness_per_generation)
    #     plt.show()
    # except ImportError:
    #     pass
