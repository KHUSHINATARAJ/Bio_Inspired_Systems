import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================
# Define states
# ==============================
EMPTY = 0   # ⬜ Empty / burnt cell
TREE  = 1   # 🌲 Tree
FIRE  = 2   # 🔥 Burning tree


# ==============================
# Initialize the forest grid
# ==============================
def initialize_forest(size=20, tree_density=0.8):
    """
    Create a forest grid with some trees and one random fire.
    """
    forest = np.zeros((size, size), dtype=int)
    
    # Fill with trees based on tree density (probability)
    forest[np.random.rand(size, size) < tree_density] = TREE
    
    # Pick one random tree to catch fire
    i, j = np.random.randint(0, size), np.random.randint(0, size)
    forest[i, j] = FIRE
    
    return forest


# ==============================
# Update the forest (parallel cellular rule)
# ==============================
def update_forest(forest):
    size = forest.shape[0]
    new_forest = forest.copy()
    
    for i in range(size):
        for j in range(size):
            if forest[i, j] == TREE:
                # If any neighbor is on fire, this tree catches fire
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if forest[ni, nj] == FIRE:
                                new_forest[i, j] = FIRE
                                break
            elif forest[i, j] == FIRE:
                # Burning tree becomes empty
                new_forest[i, j] = EMPTY
    return new_forest


# ==============================
# Display the forest
# ==============================
def display_forest(forest, step):
    plt.imshow(forest, cmap="YlOrRd", vmin=0, vmax=2)
    plt.title(f"Forest Fire Simulation - Step {step}")
    plt.axis("off")
    plt.pause(0.3)  # short pause for animation


# ==============================
# Run the simulation
# ==============================
def run_simulation(steps=20, size=20, tree_density=0.8):
    forest = initialize_forest(size, tree_density)
    plt.figure(figsize=(6,6))
    
    for step in range(steps):
        display_forest(forest, step)
        new_forest = update_forest(forest)
        
        # Stop early if no more fire
        if np.all((new_forest != FIRE)):
            display_forest(new_forest, step + 1)
            print(f"🔥 Fire stopped after {step + 1} steps.")
            break
        
        forest = new_forest
    
    plt.show()


# ==============================
# Run the whole simulation
# ==============================
if __name__ == "__main__":
    run_simulation(steps=30, size=25, tree_density=0.75)
