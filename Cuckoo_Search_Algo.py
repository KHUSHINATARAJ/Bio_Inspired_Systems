import numpy as np

# Fitness function (Sphere function)
def fitness(x):
    return sum(x**2)

# Simple Cuckoo Search
def cuckoo_search(n=10, d=2, lb=-5, ub=5, pa=0.25, max_gen=20):
    nests = np.random.uniform(lb, ub, (n, d))  # Random nests
    best = nests[0]
    best_val = fitness(best)

    for gen in range(max_gen):
        for i in range(n):
            # Generate new solution by random walk
            step = np.random.randn(d)
            new = nests[i] + step
            new = np.clip(new, lb, ub)  # Keep in bounds

            # Accept if better
            if fitness(new) < fitness(nests[i]):
                nests[i] = new

            # Update best
            if fitness(nests[i]) < best_val:
                best, best_val = nests[i], fitness(nests[i])

        # Abandon some nests
        abandon = np.random.rand(n) < pa
        nests[abandon] = np.random.uniform(lb, ub, (np.sum(abandon), d))

        print(f"Gen {gen+1}: Best = {best_val:.6f}")

    return best, best_val

# Run
best, val = cuckoo_search()
print("\nBest Solution:", best)
print("Best Value:", val)
