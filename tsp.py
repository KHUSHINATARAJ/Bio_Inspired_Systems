import random

# Distance matrix for 4 cities
distances = [
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 3],
    [10, 4, 3, 0]
]

# Fitness = total distance
def fitness(route):
    return sum(distances[route[i]][route[i+1]] for i in range(len(route)-1)) + distances[route[-1]][route[0]]

# Simple GA
def genetic_algorithm(generations=20, pop_size=4):
    # Step 1: random population
    population = [random.sample(range(4), 4) for _ in range(pop_size)]
    
    for g in range(generations):
        population.sort(key=fitness)
        best = population[0]
        print(f"Gen {g+1}: {best} -> {fitness(best)}")

        # Step 2: crossover (take half from best two)
        parent1, parent2 = population[0], population[1]
        cut = 2
        child = parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]

        # Step 3: mutation (swap)
        if random.random() < 0.3:
            i, j = random.sample(range(4), 2)
            child[i], child[j] = child[j], child[i]

        # Step 4: new population
        population[-1] = child  

    return min(population, key=fitness)

# Run
best = genetic_algorithm()
print("\nBest Route Found:", best, "with Distance:", fitness(best))
