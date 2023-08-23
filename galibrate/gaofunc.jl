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
        idx = floor(Int64, survivors[i_mp, 2]) + 1
        new_chromosome[i_mp, :] = chromosomes[idx, :]
    end    
    return new_chromosome
end

function _copy_survivor_fitnesses(pop_size, survivors, fitness_array)
    stop = floor(Int64, pop_size/2)
    for i = 1:stop
        fitness_array[i] = survivors[i, 1]
    end    
    return fitness_array
end    

function crossover(c1, c2, n_sp)

    crossover_point = floor(Int64, n_sp * rand()) + 1
    crossover_beta = rand()
    children = zeros(2, n_sp)
    x1 = c1[crossover_point]
    x2 = c2[crossover_point]
    x1_c = (1. - crossover_beta)*x1 + crossover_beta*x2
    x2_c = (1. - crossover_beta)*x2 + crossover_beta*x1
    children[1, :] = c1[:]
    children[2, :] = c2[:]
    children[1, :][crossover_point] = x1_c
    children[2, :][crossover_point] = x2_c

    return children
end    

function _generate_children(pop_size, n_sp, i_n_new, mating_pairs, chromosomes, new_chromosome)
    stop = floor(Int64, pop_size/4)
    for i_mp = 1:stop
        i_mate1_idx = mating_pairs[i_mp, 1] + 1
        i_mate2_idx = mating_pairs[i_mp, 2] + 1
        chromosome1 = chromosomes[i_mate1_idx,:]
        chromosome2 = chromosomes[i_mate2_idx,:]
        # Crossover and update the chromosomes
        children = crossover(chromosome1, chromosome2, n_sp)
        child1 = children[1, :]
        child2 = children[2, :]
        new_chromosome[i_n_new, :] = child1
        i_n_new = i_n_new + 1
        new_chromosome[i_n_new, :] = child2
        i_n_new = i_n_new + 1
    end    
    return new_chromosome
end

function random_population(pop_size, n_sp, locs, widths)
    u = rand(pop_size, n_sp)
    chromosomes = zeros(pop_size, n_sp)
    #println(chromosomes)
    for i = 1:pop_size
        for j = 1:n_sp
            chromosomes[i, j] = locs[j] + u[i, j] * widths[j]
        end
    end
    return chromosomes
end

function _set_mating_pairs(pop_size, mating_pairs, pre_mating_pairs, survivors)
    e0 = 1
    e1 = 2
    stop = floor(Int64, pop_size / 4)
    for i_hps = 1:stop
        mating_pairs[i_hps, e0] = floor(Int64, survivors[pre_mating_pairs[i_hps, e0] + 1, e1])
        mating_pairs[i_hps, e1] = floor(Int64, survivors[pre_mating_pairs[i_hps, e1] + 1, e1])
    end    
    return mating_pairs
end    

function mutation(chromosomes, locs, widths, pop_size, n_sp, mutation_rate)


    half_pop_size = floor(Int64, pop_size/2) + 1

    for i = half_pop_size:pop_size
        for j = 1:n_sp
            u = rand()
            if u < mutation_rate
                v = rand()
                chromosomes[i, j] = locs[j] + widths[j]*v
            end    
        end
    end            
    return chromosomes
end    
