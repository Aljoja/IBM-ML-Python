import numpy as np
import matplotlib.pyplot as plt
import torch

class plot_diagram():
    def __init__(self, X, Y, w, stop):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X, w_val), Y).item() for w_val in self.parameter_values] 
        w.data = start

    def __call__(self, Yhat, w, error, n, ax):
        self.error.append(error)
        self.parameter.append(w.data.item())
        
        # Plot data space and estimated line
        ax[0].cla()
        ax[0].plot(self.X, Yhat.detach().numpy(), label=f'Iteration {n}')
        ax[0].plot(self.X, self.Y, 'ro')
        ax[0].set_ylim(-20, 20)
        ax[0].set_title("Data Space and Estimated Line")
        ax[0].legend()

        # Plot loss function and parameter space
        ax[1].cla()
        ax[1].plot(self.parameter_values.numpy(), self.Loss_function, label='Loss Function')
        ax[1].plot(self.parameter, self.error, 'ro', label='Training Error')
        ax[1].set_title("Parameter Space and Loss Function")
        ax[1].legend()

def forward(x, w):
    return w * x

def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

# Create the f(X) with a slope of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

plt.figure(1)
plt.plot(X.numpy(), f.numpy(), label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

Y = f + 0.1 * torch.randn(X.size())

plt.figure(2)
plt.plot(X.numpy(), Y.numpy(), 'rx', label='Y')
plt.plot(X.numpy(), f.numpy(), label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

def train_model(w_init, lr, iterations, axs):
    w = torch.tensor(w_init, requires_grad=True)
    LOSS = []
    gradient_plot = plot_diagram(X, Y, w, stop=5)
    
    for epoch in range(iterations):
        Yhat = forward(X, w)
        loss = criterion(Yhat, Y)
        gradient_plot(Yhat, w, loss.item(), epoch, axs[epoch])
        LOSS.append(loss.item())
        loss.backward()
        w.data = w.data - lr * w.grad.data
        w.grad.data.zero_()
    
    return LOSS

# Different initial weights to analyze
initial_weights = [-10.0, -5.0, 0.0, 5.0]
iterations = 4

# Plot settings
fig, axs = plt.subplots(len(initial_weights), iterations, figsize=(20, 20))

# Run the analysis for different weights
for i, w_init in enumerate(initial_weights):
    LOSS = train_model(w_init, lr=0.1, iterations=iterations, axs=axs[i])
    for j in range(iterations):
        axs[i, j].set_title(f'Initial Weight: {w_init}, Iteration: {j}')
        axs[i, j].set_xlabel("Epoch/Iterations")
        axs[i, j].set_ylabel("Cost")

plt.tight_layout()
plt.show()