#=
Genetic algorithm functions adapted from their Python
versions in run_gao_numba. Includes all the 
functions decorated with numba.njit. 
=#

function _fill_fitness_idxs(pop_size, fitnesses, fitnesses_idxs)
    for i_mp = 1:pop_size
        fitnesses_idxs[i_mp, 1] = fitnesses[i_mp]
        # Subtract 1 for Python indexing.
        fitnesses_idxs[i_mp, 2] = i_mp - 1
    end
    return fitnesses_idxs
end    

function _move_over_survivors(pop_size, survivors, chromosomes, new_chromosome)
    for i_mp = 1:floor(Int64, pop_size/2)
        new_chromosome[i_mp] = chromosomes[floor(Int64, survivors[i_mp][1]), :]
    end    
    return new_chromosome
end

function random_population(pop_size, n_sp, locs, widths)
    u = rand(pop_size, n_sp)
    chromosomes = zeros(pop_size, n_sp)
    println(chromosomes)
    for i = 1:pop_size
        for j = 1:n_sp
            chromosomes[i, j] = locs[j] + u[i, j] * widths[j]
        end
    end
    return chromosomes
end