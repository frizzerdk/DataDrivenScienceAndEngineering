import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import pyreadr

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read the RDA file
result = pyreadr.read_r('Inputs/Hald.Cement.rda')

# The result is a dictionary where keys are dataframe names
data = result[list(result.keys())[0]]

# Extract matrix A and vector b to numpy arrays
A = data[['x1', 'x2', 'x3', 'x4']].values
b = data['Y'].values.reshape(-1, 1)

print(f"A: {A}")
print(f"b: {b}")

U, S, VT = np.linalg.svd(A, full_matrices=0)

x_hat = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

print(f"x_hat: {x_hat}")

plt.plot(b, color='k', linewidth=2, label='Heat data')
plt.plot(A@x_hat, color='r', linewidth=2, label='Estimated heat data')
plt.legend()
plt.show()





