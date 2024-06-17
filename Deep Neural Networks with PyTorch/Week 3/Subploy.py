# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 15:03:52 2024

@author: amurd
"""

import matplotlib.pyplot as plt
import numpy as np

# Example data
data = [np.random.rand(10, 2) for _ in range(10)]  # Replace with your actual data

# Create a figure
plt.figure(figsize=(15, 10))

# Number of rows and columns
rows, cols = 5, 2

# Iterate over the data and plot each in a different subplot
for i, dataset in enumerate(data):
    plt.subplot(rows, cols, i + 1)  # 5 rows, 2 columns, subplot number i+1
    plt.plot(dataset[:, 0], dataset[:, 1], 'o-')  # Replace with your actual plotting logic
    plt.title(f"Plot {i + 1}")
    plt.xlabel("X-axis label")
    plt.ylabel("Y-axis label")

# Adjust layout
plt.tight_layout()
plt.show()