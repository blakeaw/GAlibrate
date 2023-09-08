import numpy as np
from galibrate import run_gao_numba

pop_size = 8
n_params = 2
locs = np.zeros(n_params)
widths = np.ones(n_params)


def test_rungaonumba_random_population():
    rand_pop = run_gao_numba.random_population(pop_size, n_params, locs, widths)
    assert rand_pop.shape == (pop_size, n_params)
    assert (rand_pop < 1.0).all()
    assert (rand_pop > 0.0).all()


def test_rungaonumba_choose_mating_pairs():
    fit_idx = np.zeros((pop_size, 2))
    fit_idx[:, 1] = np.arange(pop_size)
    fit_idx[:, 0] = np.arange(pop_size)
    survivors = fit_idx[: int(pop_size / 2)]
    mating_pairs = run_gao_numba.choose_mating_pairs(survivors, pop_size)
    assert mating_pairs.shape == (int(pop_size / 4), 2)
    assert (mating_pairs < pop_size).all()
    assert (mating_pairs > -1).all()


def test_rungaonumba_crossover():
    chromosome_1 = np.zeros(n_params)
    chromosome_2 = np.ones(n_params)
    children = run_gao_numba.crossover(chromosome_1, chromosome_2, n_params)
    assert children.shape == (2, n_params)
    assert (children >= 0.0).all()
    assert (children <= 1.0).all()


def test_rungaonumba_mutation():
    rand_pop = run_gao_numba.random_population(pop_size, n_params, locs, widths)
    mutation_rate = 1.1
    # Shift the locs up by a small amount to make
    # sure all mutants should be different than the originals.
    locs_adjusted = locs + 0.1
    population = rand_pop.copy()
    run_gao_numba.mutation(
        population, locs_adjusted, widths, pop_size, n_params, mutation_rate
    )
    assert not np.allclose(rand_pop, population)
    # Now all mutants should be the same as the originals.
    mutation_rate = 0.0
    population = rand_pop.copy()
    run_gao_numba.mutation(population, locs, widths, pop_size, n_params, mutation_rate)
    assert np.allclose(rand_pop, population)


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


def test_rungaonumba_run_gao():
    new_population, best_pg = run_gao_numba.run_gao(
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
    new_population, best_pg = run_gao_numba.run_gao(
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

def test_rungaonumba_continue_gao():
    fitnesses = np.array([fitness(individual) for individual in SHARED['population']])
    new_population, best_pg = run_gao_numba.continue_gao(
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
    new_population, best_pg = run_gao_numba.continue_gao(
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
    test_rungaonumba_random_population()
    test_rungaonumba_choose_mating_pairs()
    test_rungaonumba_crossover()
    test_rungaonumba_mutation()
    test_rungaonumba_run_gao()
    test_rungaonumba_continue_gao()
