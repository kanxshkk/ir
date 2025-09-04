import numpy as np

# Define the function and its gradients
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2 * x, 2 * y])

# Parameters
eta = 0.1  # Learning rate
epsilon = 1e-8  # Convergence threshold
x = -0.1659  # Initial x
y = 0.4406   # Initial y
bounds = [-1, 1]  # Bounds for x and y

# Iteration counter
iteration = 0
max_iterations = 10000  # Safety limit

while True:
    # Compute gradients
    grad = grad_f(x, y)

    # Update parameters (SGD update)
    x -= eta * grad[0]
    y -= eta * grad[1]

    # Enforce bounds
    x = np.clip(x, bounds[0], bounds[1])
    y = np.clip(y, bounds[0], bounds[1])

    # Increment iteration
    iteration += 1

    # Print progress at each iteration
    print(f"Iteration {iteration}: x = {x:.8f}, y = {y:.8f}, f(x, y) = {f(x, y):.8f}")

    # Check convergence
    if np.linalg.norm(grad) < epsilon or iteration >= max_iterations:
        break

# Output results
print(f"\nFinal x: {x:.8f}, y: {y:.8f}")
print(f"Function value: {f(x, y):.8f}")
print(f"Total Iterations: {iteration}")
