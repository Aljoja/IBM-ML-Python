import numpy as np
import matplotlib.pyplot as plt
import torch

"The class for plotting"

class plot_diagram():
    """
    A class used to plot diagrams for data and model parameters during training.

    Attributes
    ----------
    X : numpy.ndarray
        The input data as a numpy array.
    Y : numpy.ndarray
        The target data as a numpy array.
    parameter_values : torch.Tensor
        The range of parameter values for plotting.
    Loss_function : list
        The list of loss values corresponding to each parameter value.
    error : list
        The list of error values during training.
    parameter : list
        The list of parameter values during training.
    """

    def __init__(self, X, Y, w, stop, go=False):
        """
        Initializes the plot_diagram with input data, target data, and initial parameters.

        Parameters
        ----------
        X : torch.Tensor
            The input data as a PyTorch tensor.
        Y : torch.Tensor
            The target data as a PyTorch tensor.
        w : torch.nn.Parameter
            The initial parameter for the model.
        stop : int or float
            The stopping value for the parameter range.
        go : bool, optional
            A flag for additional functionality (default is False).
        """
        start = w.data
        self.error = []
        self.parameter = []
        print(type(X.numpy()))
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values] 
        w.data = start

    def __call__(self, Yhat, w, error, n):
        """
        Executes the plotting process for the current iteration.

        Parameters
        ----------
        Yhat : torch.Tensor
            The predicted values from the model.
        w : torch.nn.Parameter
            The current parameter of the model.
        error : float
            The error value for the current iteration.
        n : int
            The current iteration number.
        """
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))

        # Convert lists to PyTorch tensors
        parameter_values_tensor = torch.tensor(self.parameter_values)
        loss_function_tensor = torch.tensor(self.Loss_function)

        # Plot using the tensors
        plt.plot(parameter_values_tensor.numpy(), loss_function_tensor.numpy())
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()

    def __del__(self):
        """
        Destructor to close all plot figures.
        """
        plt.close('all')



# Create the f(X) with a slope of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

# Plot the line with blue
plt.figure()
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


# Add some noise to f(X) and save it in Y
Y = f + 0.1 * torch.randn(X.size())

# Plot the data points
plt.figure()
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Create forward function for prediction
def forward(x, W):
    return W * x

# Create the MSE (Mean Squared Error) function for evaluate the result.
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

# Create Learning Rate and an empty list to record the loss for each iteration
lr = 0.1
LOSS1 = []
LOSS2 = []
LOSS3 = []

# Create some model parameters
ws = [torch.tensor(-20.0, requires_grad = True),
      torch.tensor(-15.0, requires_grad = True),
      torch.tensor(-10.0, requires_grad = True)]

stop = [20, 15, 5]
iter = 4
rows, cols = 5, 2
count = 0

# Define loop to train the 3 models
for w in ws:

    error = []
    parameter = []
    plt.figure(figsize = (12, 10))
    add = 1
    count += 1
    parameter_values = torch.arange(torch.tensor(-25.0), 20)
    Loss_function = [criterion(forward(X, W), Y) for W in parameter_values] 

    for epoch in range (iter):
        
        # make the prediction
        Yhat = forward(X, w)
        
        # calculate the iteration
        loss = criterion(Yhat,Y)
        
        # save the error and w
        error.append(loss.item())
        parameter.append(w.data)

        # Left plot
        plt.subplot(rows, cols, epoch + add)
        plt.plot(X.detach().numpy(), Yhat.detach().numpy())
        plt.plot(X.detach().numpy(), Y.detach().numpy(), 'ro')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Estimated line at iteration " + str(epoch))
        # plt.ylim(-25, 20)



        # Convert lists to PyTorch tensors
        parameter_values_tensor = torch.tensor(parameter_values)
        loss_function_tensor = torch.tensor(Loss_function)

        # Right plot
        # Plot using the tensors
        plt.subplot(rows, cols, epoch + add + 1)
        plt.plot(parameter_values_tensor.numpy(), loss_function_tensor.numpy())
        plt.plot(parameter, error, 'ro')
        plt.xlabel("w")
        plt.ylabel("L(w)")
        plt.title("Data Space Iteration " + str(epoch) + " w = " + str(w.data))


        # store the loss into list
        if count == 1:
            LOSS1.append(loss.item())
        elif count == 2:
            LOSS2.append(loss.item())
        elif count == 3:
            LOSS3.append(loss.item())

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # updata parameters
        w.data = w.data - lr * w.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()

        # Update addition of plots
        add += 1

    plt.tight_layout()


# Plot the loss for each iteration
plt.figure()
plt.plot(LOSS1, label = 'Loss $w = -20$')
plt.plot(LOSS2, label = 'Loss $w = -15$')
plt.plot(LOSS3, label = 'Loss $w = -10$')
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.legend()



plt.show()