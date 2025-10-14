import numpy as np
import cv2
import matplotlib.pyplot as plt
def fitness(threshold, image):
    threshold = int(threshold[0])  # GWO works with arrays, threshold is an array of size 1

    # Separate pixels into two groups
    foreground = image[image > threshold]
    background = image[image <= threshold]

    # Calculate weights
    w0 = len(background) / image.size
    w1 = len(foreground) / image.size

    # Means
    if len(background) == 0 or len(foreground) == 0:
        return 0  # Invalid threshold
    u0 = np.mean(background)
    u1 = np.mean(foreground)
    
    # Between-class variance
    return w0 * w1 * (u0 - u1) ** 2
class GreyWolfOptimizer:
    def __init__(self, fitness_func, num_wolves=5, max_iter=50, lb=0, ub=255):
        self.fitness_func = fitness_func
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.lb = lb  # lower bound of search space (threshold 0)
        self.ub = ub  # upper bound of search space (threshold 255)

        # Initialize wolf positions randomly
        self.positions = np.random.uniform(lb, ub, (num_wolves, 1))

        # Initialize alpha, beta, delta wolves
        self.alpha_pos = None
        self.alpha_score = -np.inf
        self.beta_pos = None
        self.beta_score = -np.inf
        self.delta_pos = None
        self.delta_score = -np.inf

    def optimize(self, image):
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                # Calculate fitness
                fitness_val = self.fitness_func(self.positions[i], image)

                # Update alpha, beta, delta
                if fitness_val > self.alpha_score:
                    self.alpha_score = fitness_val
                    self.alpha_pos = self.positions[i].copy()
                elif fitness_val > self.beta_score:
                    self.beta_score = fitness_val
                    self.beta_pos = self.positions[i].copy()
                elif fitness_val > self.delta_score:
                    self.delta_score = fitness_val
                    self.delta_pos = self.positions[i].copy()

            a = 2 - iter * (2 / self.max_iter)  # linearly decreased from 2 to 0

            for i in range(self.num_wolves):
                for j in range(1):  # dimension 1 here (threshold)
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i][j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i][j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i][j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    # Update position
                    self.positions[i][j] = (X1 + X2 + X3) / 3

        return self.alpha_pos[0], self.alpha_score
# Load grayscale image
image = cv2.imread('sample_grayscale.jpg', cv2.IMREAD_GRAYSCALE)

# Instantiate and optimize
gwo = GreyWolfOptimizer(fitness)
best_threshold, best_score = gwo.optimize(image)

print(f"Optimal Threshold found: {best_threshold:.2f}")
print(f"Fitness Value (Between-class variance): {best_score:.4f}")

# Apply threshold to segment image
_, segmented = cv2.threshold(image, int(best_threshold), 255, cv2.THRESH_BINARY)

# Display results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Original Grayscale Image")
plt.imshow(image, cmap='gray')
plt.subplot(1,2,2)
plt.title(f"Segmented Image (Threshold = {best_threshold:.2f})")
plt.imshow(segmented, cmap='gray')
plt.show()
