import random

# Fitness Function: how good is a solution?
def fitness(x):
    return x**2

# Create initial population (random numbers between 0 and 31)
def create_population(size):
    return [random.randint(0, 31) for _ in range(size)]

# Selection: pick the best parents
def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

# Crossover: mix two parents to make a child
def crossover(parent1, parent2):
    # simple average crossover
    return (parent1 + parent2) // 2

# Mutation: randomly change the child a little
def mutate(child):
    if random.random() < 0.3:  # 30% chance
        child = random.randint(0, 31)
    return child

# Genetic Algorithm
def genetic_algorithm(generations=10, population_size=6):
    population = create_population(population_size)
    print(f"Initial Population: {population}")

    for g in range(generations):
        # Evaluate fitness
        best = max(population, key=fitness)
        print(f"Generation {g+1} - Best: {best} Fitness: {fitness(best)}")

        # Select parents
        parents = selection(population)

        # Create new population
        new_population = []
        while len(new_population) < population_size:
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)

        population = new_population

    best = max(population, key=fitness)
    return best, fitness(best)

# -------------------------
# Test Case
# -------------------------

best_solution, best_score = genetic_algorithm(generations=10, population_size=6)
print("\nFinal Best Solution:", best_solution)
print("Final Best Fitness:", best_score)
