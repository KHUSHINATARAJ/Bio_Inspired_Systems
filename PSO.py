import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 4
output_size = 1

def total_weights():
    return (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size

def forward_pass(weights, x):
    idx = 0
    w1 = weights[idx:idx + input_size * hidden_size].reshape((input_size, hidden_size))
    idx += input_size * hidden_size
    b1 = weights[idx:idx + hidden_size].reshape((1, hidden_size))
    idx += hidden_size
    w2 = weights[idx:idx + hidden_size * output_size].reshape((hidden_size, output_size))
    idx += hidden_size * output_size
    b2 = weights[idx:].reshape((1, output_size))

    z1 = x @ w1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ w2 + b2
    a2 = 1 / (1 + np.exp(-z2))  
    
    return a2

def loss_function(weights):
    preds = forward_pass(weights, X)
    return np.mean((preds - y) ** 2)

num_particles = 30
dimensions = total_weights()
max_iter = 200
w = 0.7     
c1 = 1.5     
c2 = 1.5     

positions = np.random.uniform(-1, 1, (num_particles, dimensions))
velocities = np.zeros_like(positions)
pbest_positions = np.copy(positions)
pbest_scores = np.array([loss_function(p) for p in positions])
gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index]

for t in range(max_iter):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )
        positions[i] += velocities[i]

        score = loss_function(positions[i])
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i]

    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index]

    if t % 20 == 0 or t == max_iter - 1:
        print(f"Iteration {t+1}/{max_iter} - Loss: {pbest_scores[gbest_index]:.6f}")

predictions = forward_pass(gbest_position, X)
print("\nFinal predictions on XOR:")
print(np.round(predictions, 3))
print("\nTrue values:")
print(y)
