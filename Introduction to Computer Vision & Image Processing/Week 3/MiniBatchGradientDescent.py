
import numpy as np

# Define a simple quadratic function for demonstration
def quadratic_function(x):
    return x**2

# Gradient of the quadratic function
def gradient_quadratic(x):
    return 2*x

# Gradient descent function
def gradient_descent(gradient_func, initial_x, learning_rate, num_iterations):
    x = initial_x
    for _ in range(num_iterations):
        gradient = gradient_func(x)
        x = x - learning_rate * gradient
    return x

# Mini-batch gradient descent function
def mini_batch_gradient_descent(gradient_func, initial_x, learning_rate, num_iterations, batch_size):
    x = initial_x
    for _ in range(num_iterations):
        batch_data = np.random.uniform(-1, 1, batch_size)
        gradient = np.mean(gradient_func(batch_data))
        x = x - learning_rate * gradient
    return x

# Set initial parameters
initial_x = 10
learning_rate = 0.1
num_iterations = 10000

# Run gradient descent
final_x_gradient_descent = gradient_descent(gradient_quadratic, initial_x, learning_rate, num_iterations)

# Run mini-batch gradient descent
batch_size = 10
final_x_mini_batch_gradient_descent = mini_batch_gradient_descent(gradient_quadratic, initial_x, learning_rate, num_iterations, batch_size)

print("Result of Gradient Descent:", final_x_gradient_descent)
print("Result of Mini-Batch Gradient Descent:", final_x_mini_batch_gradient_descent)