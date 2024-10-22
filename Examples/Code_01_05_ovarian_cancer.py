import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#make axis
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Load group data from CSV
grp = pd.read_csv('Inputs/ovariancancer_grp.csv', header=None).values.flatten()

# Load observations data from CSV
obs = pd.read_csv('Inputs/ovariancancer_obs.csv', header=None).values

print(f'Group: {grp}')
print(f'Observations: {obs}')

U , S, VT = np.linalg.svd(obs, full_matrices=0)

for j in range(obs.shape[0]):
    x = VT[0,:] @ obs[j,:].T
    y = VT[1,:] @ obs[j,:].T
    z = VT[2,:] @ obs[j,:].T

    if grp[j] == 1:
        ax.scatter(x, y, z, marker = 'x', color='r')
    else:
        ax.scatter(x, y, z, marker = 'o', color='b')

plt.show()
