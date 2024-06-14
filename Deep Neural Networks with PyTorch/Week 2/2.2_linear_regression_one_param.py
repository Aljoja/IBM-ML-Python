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
plt.figure(1)
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


# Add some noise to f(X) and save it in Y
Y = f + 0.1 * torch.randn(X.size())

# Plot the data points
plt.figure(2)
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'Y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Create forward function for prediction
def forward(x):
    return w * x

# Create the MSE (Mean Squared Error) function for evaluate the result.
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

# Create Learning Rate and an empty list to record the loss for each iteration
lr = 0.1
LOSS = []

# Create a model parameter
ws = [torch.tensor(-10.0, requires_grad = True),
      torch.tensor(0.0, requires_grad = True),
      torch.tensor(10.0, requires_grad = True)]


# Create a 'plot_diagram' object to visualize the data space and the parameter space for each iteration during training:
# gradient_plot = plot_diagram(X, Y, w, stop = 5, go = True)


# Define a function for train the model
for w in ws:
    def train_model(iter):
        error = []
        parameter = []
        count = 1
        for epoch in range (iter):
            
            # make the prediction as we learned in the last lab
            Yhat = forward(X)
            
            # calculate the iteration
            loss = criterion(Yhat,Y)
            
            # plot the diagram for us to have a better idea
            # gradient_plot(Yhat, w, loss.item(), epoch)
            
            error.append(loss.item())
            parameter.append(w.data)
            n1 = str(iter) + str(2) + str(count)
            plt.subplot(int(n1))
            plt.plot(X, Yhat.detach().numpy())
            plt.plot(X, Y, 'ro')
            plt.xlabel("A")
            # plt.ylim(-20, 20)
            count += 1
            n2 = str(iter) + str(2) + str(count)
            plt.subplot(int(n2))
            plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(epoch))

            # Convert lists to PyTorch tensors
            parameter_values = torch.arange(0, 5)
            Loss_function = [criterion(forward(X), Y) for w.data in parameter_values] 
            parameter_values_tensor = torch.tensor(parameter_values)
            loss_function_tensor = torch.tensor(Loss_function)

            # Plot using the tensors
            plt.plot(parameter_values_tensor.numpy(), loss_function_tensor.numpy())
            plt.plot(parameter, error, 'ro')
            plt.xlabel("B")


            # store the loss into list
            LOSS.append(loss.item())
            
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            # updata parameters
            w.data = w.data - lr * w.grad.data
            
            # zero the gradients before running the backward pass
            w.grad.data.zero_()

        # Plot the loss for each iteration
        plt.plot(LOSS)
        plt.tight_layout()
        plt.xlabel("Epoch/Iterations")
        plt.ylabel("Cost")


# Give 4 iterations for training the model here.
train_model(4)



plt.show()