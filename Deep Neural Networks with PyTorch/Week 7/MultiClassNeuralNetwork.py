import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
'Neural Network Architecture'
This defines a simple neural network with one hidden layer. 
The forward method describes how data flows through the network.
"""

class MultiClassNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



"""
'Hyperparameters'
These set the network architecture and training parameters.
"""

input_size = 10
hidden_size = 50
num_classes = 3
learning_rate = 0.01
num_epochs = 100
batch_size = 32


"""
'Data Generation'
This creates dummy data for training. 
In a real scenario, you'd load your actual dataset here.
"""

X = torch.randn(1000, input_size)
y = torch.randint(0, num_classes, (1000,))

"Let's plot the dummy data"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert to numpy for PCA
X_np = X.numpy()
y_np = y.numpy()

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_np)

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_np, cmap='viridis', alpha=0.6)

# Add a color bar
plt.colorbar(scatter)

# Set labels and title
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Visualization of Multi-Class Dummy Data')

# Add a legend
handles, labels = scatter.legend_elements()
legend = plt.legend(handles, [f'Class {i}' for i in range(num_classes)], loc="upper right", title="Classes")

plt.show()

"""
'DataLoader'
This creates a DataLoader to handle batching and shuffling of the data during training.
"""

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


"""
'Model Initialization'
This initializes the model, loss function (CrossEntropyLoss for multi-class classification), and optimizer.
"""

model = MultiClassNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


"""
'Training Loop'
This is the main training loop. 
It iterates through the data in batches, computes the loss, and updates the model parameters.
"""

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


"""
'Testing'
This section demonstrates how to use the trained model for predictions on new data.
This code provides a basic framework for multi-class classification with PyTorch. 
In a real-world scenario, you'd replace the dummy data with your actual dataset, 
possibly add more layers to the network, implement validation to prevent overfitting, 
and include metrics to evaluate the model's performance.
"""

with torch.no_grad():
    test_X = torch.randn(100, input_size)
    outputs = model(test_X)
    _, predicted = torch.max(outputs, 1)
    print('Example predictions:', predicted[:10])
